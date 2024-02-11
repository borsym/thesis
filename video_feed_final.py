import open3d as o3d
import cv2
import threading
import time
import numpy as np
import json
import os
import glob
import time
from queue import Queue

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmengine import init_default_scope
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances

from ultralytics import YOLO

import matplotlib.pyplot as plt


def calibrate_intrinsics(image0, image1, chessboard_size=(9, 6), square_size=0.07):
    square_size = square_size  # 70 mm
    # Number of inner corners (width, height)
    chessboard_size = chessboard_size

    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world coordinates
    img_points0 = []  # 2D points in camera 1 image
    img_points1 = []  # 2D points in camera 2 image

    # Create a list of object points for the chessboard
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    obj_points.append(objp)
    # Loop through your image pairs (images from each camera)
    print('Gathering Intrinsic Camera Parameters...')
    img = None
    # Replace '1' with the number of image pairs you have
    for i, img in enumerate([image0, image1]):
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(cv2.cvtColor(
                img, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)

            if i == 0:
                img_points0.append(corners)
            elif i == 1:
                img_points1.append(corners)
    # Calibrate each camera

    ret0, mtx0_int, dist0_int, rvecs0, tvecs0 = cv2.calibrateCamera(obj_points, img_points0, img.shape[::-1][1:], None,
                                                                    None)
    ret1, mtx1_int, dist1_int, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points, img_points1, img.shape[::-1][1:], None,
                                                                    None)
    return mtx0_int, dist0_int, mtx1_int, dist1_int


def calibrate_extrinsics(camera_int_params, chessboard_size=(9, 6), square_size=70):
    mtx0_int = camera_int_params[0]
    mtx1_int = camera_int_params[2]
    dist0_int = camera_int_params[1]
    dist1_int = camera_int_params[3]

    # Chessboard settings
    # Number of inner corners per a chessboard row and column (8x6 board)
    chessboard_size = chessboard_size
    square_size = square_size  # Size of a chessboard square in mm

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints_left = []  # 2d points in image plane for left camera
    imgpoints_right = []  # 2d points in image plane for right camera

    # Load images for left and right camera
    images_left = glob.glob('extrinsic/image_0_*1.png')  # Update path
    images_right = glob.glob('extrinsic/image_1_*1.png')  # Update path
    gray_left = None
    # for left_img, right_img in tqdm.tqdm(zip(sorted(images_left), sorted(images_right)), desc="Camera Stereo Calibration for Extrinsic Parameters"):
    for left_img, right_img in zip(sorted(images_left), sorted(images_right)):
        print(f'left img: {left_img} \t\t right img: {right_img}')
        img_left = cv2.imread(left_img)
        img_right = cv2.imread(right_img)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, chessboard_size, None)

        # If found, add object points and image points
        if ret_left and ret_right:
            objpoints.append(objp)
            corners2_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2_left)
            imgpoints_right.append(corners2_right)

    # Stereo calibration
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx0_int, dist0_int, mtx1_int, dist1_int, gray_left.shape[::-1])

    print("Rotation matrix:\n", R)
    print("Translation vector:\n", T)
    return R, T


class humanDetector:
    def __init__(self, camera_int_params, camera_ext_params):
        self.camera_int_params = camera_int_params
        self.camera_ext_params = camera_ext_params
        init_default_scope('mmdet')
        det_config = 'D:/thesis/realtime_update/weights/rtmdet_nano_320-8xb32_coco-person.py'
        det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'

        model_cfg = 'D:/thesis/realtime_update/weights/rtmpose-m_8xb256-420e_coco-256x192.py'
        ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth '
        device = 'cuda'
        init_default_scope('mmpose')
        self.model = init_pose_estimator(model_cfg, ckpt, device=device,
                                         cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        # Initialize detection model
        init_default_scope('mmdet')
        self.detection_model = init_detector(
            det_config, det_checkpoint, device=device)

        self.mtx0_int = np.asarray(camera_int_params['cam0_K'])
        self.mtx1_int = np.asarray(camera_int_params['cam1_K'])
        self.dist0_int = np.asarray(camera_int_params['cam0_dist'])
        self.dist1_int = np.asarray(camera_int_params['cam1_dist'])

        self.R = np.asarray(camera_ext_params['R'])
        self.T = np.asarray(camera_ext_params['T'])

        self.Proj_matrix0 = np.dot(
            self.mtx0_int, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.Proj_matrix1 = np.dot(
            self.mtx1_int, np.hstack((self.R, self.T.reshape(-1, 1))))

        ## ezt kulon kell szedni ##

        mesh = o3d.io.read_triangle_mesh(
            "D:/thesis/realtime_update/meshes/beanbag.ply")
        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for _ in range(len(mesh.vertices))])
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()

        self.skeleton_cloud = o3d.geometry.PointCloud()
        keypoint_color = [0, 1, 0]
        self.skeleton_cloud.paint_uniform_color(keypoint_color)

        self.connections = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
                            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        self.lines = o3d.geometry.LineSet()
        self.lines.points = self.skeleton_cloud.points
        self.lines.lines = o3d.utility.Vector2iVector(self.connections)
        self.lines.paint_uniform_color([0, 1, 0])

        self.visualizer.add_geometry(mesh)
        self.visualizer.add_geometry(self.lines)
        self.visualizer.add_geometry(self.skeleton_cloud)

        render_option = self.visualizer.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = False

        self.sphere_list = []
        for _, _ in enumerate(range(17)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.05)  # Adjust radius for size
            sphere.translate([0, 0, 0])
            sphere.paint_uniform_color([1, 0, 0])
            self.sphere_list.append(sphere)
            self.visualizer.add_geometry(sphere)

        time.sleep(10)

    def select_person(self, image):
        det_result = inference_detector(self.detection_model, image)
        pred_instance = det_result.pred_instances.cpu().numpy()

        # put the prediction to the last column  1511      224.36        1920      1189.2     0.80981]
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        # select human bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                       pred_instance.scores > 0.3)]

        return bboxes[nms(bboxes, 0.3), :4]

    def mark_kp(self, image, x, y, color=(0, 0, 255)):
        cv2.circle(image, (int(round(x)), int(round(y))),
                   10, color=color, thickness=cv2.FILLED)

    def mark_image(self, image, results, rgb):
        for pose in results[0].pred_instances.keypoints:
            for kp in pose:
                x, y = kp
                self.mark_kp(image, x, y)
            for start_point, end_point in self.connections:
                cv2.line(image, [int(pose[start_point][0]), int(pose[start_point][1])], [int(
                    pose[end_point][0]), int(pose[end_point][1])], color=rgb, thickness=3)
        return image

    def inference(self, image0, image1):
        init_default_scope('mmdet')
        person_bboxes0 = self.select_person(image0)
        person_bboxes1 = self.select_person(image1)
        init_default_scope('mmpose')

        if person_bboxes0 == [] or person_bboxes1 == []:
            print("return")
            return None, None

        results0 = inference_topdown(self.model, image0, person_bboxes0)
        results1 = inference_topdown(self.model, image1, person_bboxes1)

        # return image
        self.mark_image(image0, results0, (0, 255, 0))
        self.mark_image(image1, results1, (0, 255, 0))

        return results0, results1, image0, image1

    def update_open3d(self):
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def get_pose_xyz(self, frame):
        pose_set = []
        for x, y, z in zip(frame[0], frame[1], frame[2]):
            # divide by 1000 if camera focal length is given in mm
            pose_set.append(np.asarray([x, y, z])/1000)
        return np.asarray(pose_set)

    def update_sphere_position(self, sphere, new_center):
        # Calculate the current center of the sphere
        current_center = np.mean(np.asarray(sphere.vertices), axis=0)
        # Calculate the shift required to move to the new center
        shift = new_center - current_center
        # Update the vertices
        np_vertices = np.asarray(sphere.vertices)
        sphere.vertices = o3d.utility.Vector3dVector(np_vertices + shift)
        # Update the mesh (important for visualization)
        sphere.compute_vertex_normals()
        return sphere

    def triangulation(self, results0, results1, P0, P1):
        left_pose = max(results0[0].pred_instances,
                        key=lambda item: np.mean(item.keypoint_scores))
        right_pose = max(results1[0].pred_instances,
                         key=lambda item: np.mean(item.keypoint_scores))

        points_4d_hom = cv2.triangulatePoints(P0, P1, np.asarray(left_pose.keypoints).squeeze().T,
                                              np.asarray(right_pose.keypoints).squeeze().T)

        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d = self.get_pose_xyz(points_3d)

        ### 3d open visualization jon ##
        self.skeleton_cloud.points = o3d.utility.Vector3dVector(points_3d)
        self.lines.points = o3d.utility.Vector3dVector(points_3d)

        self.visualizer.update_geometry(self.skeleton_cloud)
        self.visualizer.update_geometry(self.lines)

        for point_idx, point in enumerate(range(17)):
            if point_idx > len(self.skeleton_cloud.points):
                sphere = self.sphere_list[point_idx]
                self.update_sphere_position(sphere, np.array([0, 0, 0]))
            else:
                sphere = self.sphere_list[point_idx]
                self.update_sphere_position(sphere, np.array(
                    self.skeleton_cloud.points[point_idx]))
            self.visualizer.update_geometry(sphere)

        self.update_open3d()

        # pcd = o3d.geometry.PointCloud()
        # lines = o3d.utility.Vector2iVector(self.connections)
        # lines.paint_uniform_color([0, 1, 0])

        # # Set the points of the PointCloud object to the points_3d
        # pcd.points = o3d.utility.Vector3dVector(
        #     points_3d)  # Triangulate the 2D keypoints to 3D

        # o3d.visualization.draw_geometries([pcd])

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Scatter plot
        # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

        # # Drawing the lines based on the connection pattern
        # # for connection in self.connections:
        # #     start_point, end_point = connection
        # #     # Adjust indices in the connection list if necessary, assuming 0-based indexing here
        # #     start_point -= 1
        # #     end_point -= 1

        # #     # Check if start and end points are within the range of points_3d to avoid IndexError
        # #     if start_point < len(points_3d) and end_point < len(points_3d):
        # #         ax.plot([points_3d[start_point][0], points_3d[end_point][0]],
        # #                 [points_3d[start_point][1], points_3d[end_point][1]],
        # #                 [points_3d[start_point][2], points_3d[end_point][2]], color='blue')  # Choose color

        # # # Set labels
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # # Show plot
        # plt.show()

        # left_pose = max(results0[0].pred_instances, key=lambda item: np.mean(item.keypoint_scores))
        # right_pose = max(results1[0].pred_instances, key=lambda item: np.mean(item.keypoint_scores))
        # Convert the 2D keypoints to 3D

    def run(self):
        # Your code here
        pass


class robotDetector:
    def __init__(self) -> None:
        self.model = YOLO('D:/thesis/realtime_update/convert_images/best.pt')
        self.skeleton = self.get_skeleton()

    def get_skeleton(self):
        with open("D:/thesis/realtime_update/robot-keypoints-connection.json", 'r') as file:
            annotations_data = json.load(file)
        return annotations_data['skeleton']

    def inference(self, image):
        result = self.model(image)
        return result

    def get_keypoints(self, result):
        return result[0].keypoints.xy[0]

    def draw_keypoints(self, image, points):

        for kp in points:
            x, y = kp
            cv2.circle(image, (int(x), int(y)),
                       radius=5, color=(255, 0, 0), thickness=-1)
        return image

    def draw_skeleton(self, image, points):
        for link in self.skeleton:
            p1, p2 = int(link[0]), int(link[1])
            pt1, pt2 = points[p1 - 1], points[p2 - 1]
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(
                pt2[0]), int(pt2[1])), (255, 0, 255), 1)
        return image

    def draw(self, image, points):
        if len(points) <= 0:
            return image
        self.draw_keypoints(image, points)
        self.draw_skeleton(image, points)
        return image


class VideoManager:
    def __init__(self, camera_int_params, camera_ext_params):
        # Initialization code remains mostly the same...
        # Initialize your models and visualizer here as before
        # Open video files instead of starting camera threads
        self.mtx0_int = np.asarray(camera_int_params['cam0_K'])
        self.mtx1_int = np.asarray(camera_int_params['cam1_K'])
        self.dist0_int = np.asarray(camera_int_params['cam0_dist'])
        self.dist1_int = np.asarray(camera_int_params['cam1_dist'])

        self.R = np.asarray(camera_ext_params['R'])
        self.T = np.asarray(camera_ext_params['T'])

        self.Proj_matrix0 = np.dot(
            self.mtx0_int, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.Proj_matrix1 = np.dot(
            self.mtx1_int, np.hstack((self.R, self.T.reshape(-1, 1))))

        self.video0 = cv2.VideoCapture(
            'D:/thesis/realtime_update/recordings/Scenario1/Cam1/out.mp4')
        self.video1 = cv2.VideoCapture(
            'D:/thesis/realtime_update/recordings/Scenario1/Cam2/out.mp4')

        self.robot_detector = robotDetector()
        self.human_detector = humanDetector(
            camera_int_params, camera_ext_params)

    def run(self):
        while True:
            start_time = time.time()
            # Read frames from each video
            ret0, frame0 = self.video0.read()
            ret1, frame1 = self.video1.read()

            # Check if either frame could not be read -> end of video
            if not ret0 or not ret1:
                break

            # img0_undistorted = cv2.undistort(frame0, self.mtx0_int, self.dist0_int, None, self.mtx0_int)
            # img1_undistorted = cv2.undistort(frame1, self.mtx1_int, self.dist1_int, None, self.mtx1_int)

            # results0, results1 = neural_network_process(
            #     self.detection_model, self.model, image_list[0], image_list[1], self.connections)
            # run_triangulation(results0, results1, self.Proj_matrix0, self.Proj_matrix1, self.skeleton_cloud,
            #                   self.lines, self.sphere_list, self.vis)
            # self.update_cv2_windows()  # Update OpenCV windows here

            # robot detection
            # result_frame0 = self.robot_detector.inference(frame0)
            # result_frame1 = self.robot_detector.inference(frame1)

            # points0 = self.robot_detector.get_keypoints(result_frame0)
            # points1 = self.robot_detector.get_keypoints(result_frame1)

            # self.robot_detector.draw(frame0, points0)
            # self.robot_detector.draw(frame1, points1)

            # human detection
            result0, result1, frame0, frame1 = self.human_detector.inference(
                frame0, frame1)
            self.human_detector.triangulation(
                result0, result1, self.Proj_matrix0, self.Proj_matrix1)
            # self.update_cv2_windows2("Camera 0", frame0)
            # self.update_cv2_windows2("Camera 1", frame1)

            elapsed_time = time.time() - start_time
            # Calculate frames per second
            fps = 1 / elapsed_time

            # Print FPS every 10 frames
            print(f"FPS: {fps:.2f}")

        self.video0.release()
        self.video1.release()
        cv2.destroyAllWindows()

    def update_cv2_windows2(self, window_name, image):
        # Resize and show the frame as needed
        resized_frame = cv2.resize(image, (800, 500))
        cv2.imshow(window_name, resized_frame)
        cv2.waitKey(1)


# class CameraManager:
#     def __init__(self, camera_int_params, camera_ext_params):
#         self.camera_threads = []

#         init_default_scope('mmdet')
#         det_config = '/home/gables/Programs/mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
#         det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'

#         # model_cfg = '/home/gables/Programs/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py'
#         # ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth'

#         model_cfg = '/home/gables/Programs/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
#         ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth '
#         device = 'cuda'
#         init_default_scope('mmpose')
#         self.model = init_pose_estimator(model_cfg, ckpt, device=device,
#                                          cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

#         # Initialize detection model
#         init_default_scope('mmdet')
#         self.detection_model = init_detector(
#             det_config, det_checkpoint, device=device)
#         mesh = o3d.io.read_triangle_mesh("meshes/temp.ply")

#         self.mtx0_int = np.asarray(camera_int_params['cam0_K'])
#         self.mtx1_int = np.asarray(camera_int_params['cam1_K'])
#         self.dist0_int = np.asarray(camera_int_params['cam0_dist'])
#         self.dist1_int = np.asarray(camera_int_params['cam1_dist'])

#         self.R = np.asarray(camera_ext_params['R'])
#         self.T = np.asarray(camera_ext_params['T'])

#         self.Proj_matrix0 = np.dot(
#             self.mtx0_int, np.hstack((np.eye(3), np.zeros((3, 1)))))
#         self.Proj_matrix1 = np.dot(
#             self.mtx1_int, np.hstack((self.R, self.T.reshape(-1, 1))))

#     def start_cameras(self, num_cameras):
#         for i in range(num_cameras):
#             cv2.namedWindow(f'Camera {i}', cv2.WINDOW_NORMAL)
#             cv2.resizeWindow(f'Camera {i}', 800, 500)

#     def update_cv2_windows(self):
#         for index, queue in camera_queues.items():
#             if not queue.empty():
#                 frame = queue.get()
#                 window_name = f'Camera {index}'
#                 # Resize and show the frame as needed
#                 resized_frame = cv2.resize(frame, (800, 500))
#                 cv2.imshow(window_name, resized_frame)
#                 cv2.waitKey(1)

#     def update_cv2_windows2(self, window_name, image):
#         # Resize and show the frame as needed
#         resized_frame = cv2.resize(image, (800, 500))
#         cv2.imshow(window_name, resized_frame)
#         cv2.waitKey(1)

#     def run_advanced(self):
#         while True:
#             start_time = time.time()
#             image_list = []
#             for index, queue in camera_queues.items():
#                 image_list.append(queue.get())
#             if len(image_list) > 1:
#                 # img0_undistorted = cv2.undistort(image_list[0], self.mtx0_int, self.dist0_int, None, self.mtx0_int)
#                 # img1_undistorted = cv2.undistort(image_list[1], self.mtx1_int, self.dist1_int, None, self.mtx1_int)
#                 results0, results1 = neural_network_process(
#                     self.detection_model, self.model, image_list[0], image_list[1], self.connections)
#                 run_triangulation(results0, results1, self.Proj_matrix0, self.Proj_matrix1, self.skeleton_cloud,
#                                   self.lines, self.sphere_list, self.vis)
#             # self.update_cv2_windows()  # Update OpenCV windows here
#             self.update_cv2_windows2("Camera 0", image_list[0])
#             self.update_cv2_windows2("Camera 1", image_list[1])
#             self.vis.poll_events()
#             self.vis.update_renderer()

#             elapsed_time = time.time() - start_time
#             # Calculate frames per second
#             fps = 1 / elapsed_time

#             # Print FPS every 10 frames
#             print(f"FPS: {fps:.2f}")

#     def run(self):
#         self.start_cameras(2)  # Assuming two cameras
#         self.run_advanced()   # This will also handle OpenCV updates
#         self.stop_cameras()
#         cv2.destroyAllWindows()


def main():
    chessboard_size = (9, 6)
    square_size = 70  # one square size in mm

    if os.path.exists(f'intrinsic/intrinsics.json'):
        print(f'loading intrinsics from config...')
        with open('intrinsic/intrinsics.json', 'r') as json_file:
            camera_int_params = json.load(json_file)
    else:
        print(f"The file 'intrinsic/intrinsics.json' does not exist. Creating it...")
        door_cam_internal_img = cv2.imread(
            f'intrinsic/door.jpg')  # Load the image
        window_cam_internal_img = cv2.imread(
            f'intrinsic/window2.jpg')  # Load the image

        camera_int_params = calibrate_intrinsics(door_cam_internal_img, window_cam_internal_img, chessboard_size,
                                                 square_size)
        data = {'cam0_K': camera_int_params[0].tolist(),
                'cam0_dist': camera_int_params[1].tolist(),
                'cam1_K': camera_int_params[2].tolist(),
                'cam1_dist': camera_int_params[3].tolist()}
        with open('intrinsic/config.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        with open('intrinsic/config.json', 'r') as json_file:
            camera_int_params = json.load(json_file)

    if os.path.exists(f'extrinsic/extrinsics.json'):
        print(f'loading extrinsic from config...')
        with open('extrinsic/extrinsics.json', 'r') as json_file:
            camera_ext_params = json.load(json_file)
    else:
        print(f"The file 'extrinsic/config.json' does not exist. Creating it...")
        camera_ext_params = calibrate_extrinsics(
            camera_int_params, chessboard_size, square_size)
        data = {'R': camera_ext_params[0].tolist(),
                'T': camera_ext_params[1].tolist()}
        with open('extrinsic/config.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        with open('extrinsic/config.json', 'r') as json_file:
            camera_ext_params = json.load(json_file)

    print(f'camera_int_params:\n{camera_int_params}')
    print(f'camera_ext_params:\n{camera_ext_params}')

    return camera_int_params, camera_ext_params

    ### Camera calibration over ###

    # camera_manager = CameraManager(camera_int_params, camera_ext_params)
    # camera_manager.run()


if __name__ == "__main__":
    ins, ex = main()
    VideoManager = VideoManager(ins, ex)
    VideoManager.run()
    # main()
