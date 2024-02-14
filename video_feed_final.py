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
from scipy.spatial import procrustes


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


class Visualizer:
    def __init__(self) -> None:
        mesh = o3d.io.read_triangle_mesh(
            "D:/thesis/realtime_update/meshes/rotated_beanbag.ply")

        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for _ in range(len(mesh.vertices))])
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        self.robot_mesh = o3d.io.read_triangle_mesh(
            "D:/thesis/realtime_update/meshes/save4.ply")

        if not self.robot_mesh.has_vertex_colors():
            self.robot_mesh.vertex_colors = o3d.utility.Vector3dVector(
                [[0.5, 0.5, 0.5] for _ in range(len(self.robot_mesh.vertices))])
        if not self.robot_mesh.has_vertex_normals():
            self.robot_mesh.compute_vertex_normals()

        self.nerf_robot_keypoints = np.array([
            [-0.88645411, -0.44562197, 0.93118215],
            [-0.63839042, -0.37820864, -1.22604775],
            [0.83720541, -0.33930898, 1.09157586],
            [1.00591755, -0.24008346, -1.06751966],
            [-1.39356089, 0.32691932, -0.14874268],
            [-1.04946733, 0.22727919, -0.5598197],
            [-1.1274302, 0.16455293, 0.26233435],
            [-1.02111721, 1.14981174, 0.29777205],
            [-1.28275752, 1.59558785, -0.07786727],
            [-0.99985462, 1.190431, -0.446419],
            [0.09162557, 0.18090272, -0.83623338],
            [-0.03595006, 0.10247564, 0.74428666],
            [1.17964864, 0.3623569, -0.29758072],
            [1.11931801, 0.30521822, 0.43243515],
            [0.86416674, 1.27875948, 0.51748562],
            [0.89251685, 1.31221807, -0.29049325],
            [1.25816941, 1.35461175, 0.17728388]
        ])

        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window()

        self.robot_mesh_vertex_cloud = o3d.geometry.PointCloud()
        self.robot_mesh_vertex_cloud.paint_uniform_color([0, 1, 1])

        self.vertex_cloud_for_robot = np.array(
            self.robot_mesh.vertices)[::1000, :]

        self.person_skeleton_cloud = o3d.geometry.PointCloud()
        keypoint_color = [0, 1, 0]
        self.person_skeleton_cloud.paint_uniform_color(keypoint_color)
        self.connections = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
                            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        self.person_lines = o3d.geometry.LineSet()
        self.person_lines.points = self.person_skeleton_cloud.points
        self.person_lines.lines = o3d.utility.Vector2iVector(self.connections)
        self.person_lines.paint_uniform_color([0, 1, 0])

        self.robot_conections = self.get_skeleton_connection_robot()
        print("robot connection list", self.robot_conections)
        self.robot_skeleton_cloud = o3d.geometry.PointCloud()
        keypoint_color = [0, 0, 1]
        self.robot_skeleton_cloud.paint_uniform_color(keypoint_color)

        self.robot_lines = o3d.geometry.LineSet()
        self.robot_lines.points = self.robot_skeleton_cloud.points
        self.robot_lines.lines = o3d.utility.Vector2iVector(
            self.robot_conections)
        self.robot_lines.paint_uniform_color([0, 0, 1])

        self.visualizer.add_geometry(mesh)
        self.visualizer.add_geometry(self.robot_mesh_vertex_cloud)
        self.visualizer.add_geometry(self.person_lines)
        self.visualizer.add_geometry(self.person_skeleton_cloud)
        self.visualizer.add_geometry(self.robot_lines)
        self.visualizer.add_geometry(self.robot_skeleton_cloud)

        render_option = self.visualizer.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = False

        self.direction_human_arrow_shaft = o3d.geometry.LineSet()
        self.direction_human_arrow_head = o3d.geometry.TriangleMesh.create_cone(
            radius=0.04, height=0.1)
        self.direction_human_arrow_head.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrowhead
        self.direction_human_arrow_shaft.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrow shaft
        self.visualizer.add_geometry(self.direction_human_arrow_shaft)
        # self.visualizer.add_geometry(self.direction_human_arrow_head)

        self.direction_robot_arrow_shaft = o3d.geometry.LineSet()
        self.direction_robot_arrow_head = o3d.geometry.TriangleMesh.create_cone(
            radius=0.04, height=0.1)
        self.direction_robot_arrow_head.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrowhead
        self.direction_robot_arrow_shaft.paint_uniform_color(
            [1, 0, 0])  # Red color for the arrow shaft
        self.visualizer.add_geometry(self.direction_robot_arrow_shaft)
        # self.visualizer.add_geometry(self.direction_robot_arrow_head)

    def update_arrow(self, start_point, direction_vector, object_type):
        # Update the arrow's shaft
        scaling_factor = 5  # Adjust the scaling factor as needed
        end_point = start_point + scaling_factor * direction_vector
        points = [start_point, end_point]
        lines = [[0, 1]]  # LineSet uses indices into the points list
        if object_type == "human":
            self.direction_human_arrow_shaft.points = o3d.utility.Vector3dVector(
                points)
            self.direction_human_arrow_shaft.lines = o3d.utility.Vector2iVector(
                lines)
        elif object_type == "robot":
            self.direction_robot_arrow_shaft.points = o3d.utility.Vector3dVector(
                points)
            self.direction_robot_arrow_shaft.lines = o3d.utility.Vector2iVector(
                lines)

        # Update the arrow's head
        # Place the cone at the end of the shaft and rotate it to point in the direction of the vector
        transformation_matrix = self.get_arrow_transformation_matrix(
            end_point, direction_vector)  # TODO start helyett lehet end
        self.direction_human_arrow_head.transform(
            transformation_matrix)

        if object_type == "human":
            # Update the geometries
            self.visualizer.update_geometry(self.direction_human_arrow_shaft)
            self.visualizer.update_geometry(self.direction_human_arrow_head)
        elif object_type == "robot":
            # Update the geometries
            self.visualizer.update_geometry(self.direction_robot_arrow_shaft)
            self.visualizer.update_geometry(self.direction_robot_arrow_head)

    def get_arrow_transformation_matrix(self, end_point, direction_vector):
        # Normalize the direction vector
        direction = direction_vector / np.linalg.norm(direction_vector)

        # Create a rotation matrix that aligns the z-axis to the direction vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_angle = np.arccos(
            np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        rotation_axis_angle = rotation_axis * rotation_angle
        rotation_axis_angle = np.expand_dims(rotation_axis_angle, axis=1)

        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis_angle)

        # Create the transformation matrix with rotation and translation
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = end_point
        return transformation_matrix

    def get_skeleton_connection_robot(self):
        with open("D:/thesis/realtime_update/robot-keypoints-connection.json", 'r') as file:
            annotations_data = json.load(file)
        skeleton_connections = annotations_data['skeleton']
        # shift the indices by 1 to match the 0-based indexing in open3d
        skeleton_connections = [[x - 1, y - 1]
                                for x, y in skeleton_connections]
        return skeleton_connections

    def update_open3d(self):
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

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

    def align_mesh_to_keyponts(self, keypoints):
        mesh_vertex_cloud = self.vertex_cloud_for_robot

        centroid_robot_keypoints = np.mean(keypoints, axis=0)
        centroid_nerf_keypoints = np.mean(self.nerf_robot_keypoints, axis=0)

        centered_keypoints = keypoints - centroid_robot_keypoints
        centered_mesh_vertex_cloud = mesh_vertex_cloud - centroid_nerf_keypoints
        centered_nerf_keypoints = self.nerf_robot_keypoints - centroid_nerf_keypoints

        # Perform Procrustes analysis
        _, mtx2, _ = procrustes(centered_keypoints, centered_nerf_keypoints)

        transformation_matrix = np.linalg.lstsq(
            centered_nerf_keypoints, mtx2, rcond=None)[0]
        transformed_mesh_vertex_cloud = centered_mesh_vertex_cloud @ transformation_matrix.T

        # transformed_mesh_vertex_cloud *= 1.2

        # Translate the transformed mesh vertices to match the robot keypoints centroid
        transformed_mesh_vertex_cloud += centroid_robot_keypoints

        return transformed_mesh_vertex_cloud

        # return mesh

    def run_human(self, points_3d):

        ### 3d open visualization jon ##
        self.person_skeleton_cloud.points = o3d.utility.Vector3dVector(
            points_3d)
        self.person_lines.points = o3d.utility.Vector3dVector(points_3d)

        self.visualizer.update_geometry(self.person_skeleton_cloud)
        self.visualizer.update_geometry(self.person_lines)

        # for point_idx, point in enumerate(range(17)):
        #     if point_idx > len(self.person_skeleton_cloud.points):
        #         sphere = self.sphere_list[point_idx]
        #         self.update_sphere_position(sphere, np.array([0, 0, 0]))
        #     else:
        #         sphere = self.sphere_list[point_idx]
        #         self.update_sphere_position(sphere, np.array(
        #             self.person_skeleton_cloud.points[point_idx]))
        #     self.visualizer.update_geometry(sphere)

    def run_robot(self, points_3d):

        self.robot_skeleton_cloud.points = o3d.utility.Vector3dVector(
            points_3d)
        self.robot_lines.points = o3d.utility.Vector3dVector(points_3d)

        self.visualizer.update_geometry(self.robot_skeleton_cloud)
        self.visualizer.update_geometry(self.robot_lines)


class HumanDetector:
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

        self.connections = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
                            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

        self.all_points_3d = np.array([])
        self.previous_keypoints = None
        self.current_keypoints = None

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

    def get_pose_xyz(self, frame):
        pose_set = []
        for x, y, z in zip(frame[0], frame[1], frame[2]):
            # divide by 1000 if camera focal length is given in mm
            pose_set.append(np.asarray([x, y, z])/1000)
        return np.asarray(pose_set)

    def save_to_npy(self):
        np.save("all_points_3d.npy", np.array(
            self.all_points_3d, dtype=object))

    def triangulation(self, results0, results1, P0, P1):
        left_pose = max(results0[0].pred_instances,
                        key=lambda item: np.mean(item.keypoint_scores))
        right_pose = max(results1[0].pred_instances,
                         key=lambda item: np.mean(item.keypoint_scores))

        points_4d_hom = cv2.triangulatePoints(P0, P1, np.asarray(left_pose.keypoints).squeeze().T,
                                              np.asarray(right_pose.keypoints).squeeze().T)

        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d = self.get_pose_xyz(points_3d)
        # self.all_points_3d.append(np.array(points_3d))
        # self.save_to_npy()
        if self.current_keypoints is not None:
            self.previous_keypoints = self.current_keypoints
        self.current_keypoints = points_3d
        return points_3d


class RobotDetector:
    def __init__(self) -> None:
        self.model = YOLO('D:/thesis/realtime_update/weights/best.pt')
        self.skeleton = self.get_skeleton()
        self.previous_keypoints = None
        self.current_keypoints = None

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

    def get_pose_xyz(self, frame):
        pose_set = []
        for x, y, z in zip(frame[0], frame[1], frame[2]):
            # divide by 1000 if camera focal length is given in mm
            pose_set.append(np.asarray([x, y, z])/1000)
        return np.asarray(pose_set)

    def triangulation(self, results0, results1, P0, P1, points0, points1):
        if len(points0) == 0 or len(points1) == 0:
            return []

        left_pose = max(results0[0].keypoints,
                        key=lambda item: np.mean(item.conf.cpu().numpy()))
        right_pose = max(results1[0].keypoints,
                         key=lambda item: np.mean(item.conf.cpu().numpy()))

        points_4d_hom = cv2.triangulatePoints(P0, P1, np.asarray(left_pose.xy[0].cpu().numpy()).squeeze().T,
                                              np.asarray(right_pose.xy[0].cpu().numpy()).squeeze().T)

        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d = self.get_pose_xyz(points_3d)

        if self.current_keypoints is not None:
            self.previous_keypoints = self.current_keypoints
        self.current_keypoints = points_3d

        return points_3d


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

        self.open3d_visualizer = Visualizer()
        self.robot_detector = RobotDetector()
        self.human_detector = HumanDetector(
            camera_int_params, camera_ext_params)

    def run(self):
        # self.Visualizer.run()
        first_frame = True
        while True:
            start_time = time.time()
            # Read frames from each video
            ret0, frame0 = self.video0.read()
            ret1, frame1 = self.video1.read()

            # Check if either frame could not be read -> end of video
            if not ret0 or not ret1:
                break

            # ##### human detection #####
            result0, result1, frame0, frame1 = self.human_detector.inference(
                frame0, frame1)
            human_points_3d = self.human_detector.triangulation(
                result0, result1, self.Proj_matrix0, self.Proj_matrix1)

            if self.human_detector.previous_keypoints is not None:
                # Calculate the direction vector and start point
                direction_vector = np.mean(
                    self.human_detector.current_keypoints - self.human_detector.previous_keypoints, axis=0)
                start_point = (
                    self.human_detector.current_keypoints[5] + self.human_detector.current_keypoints[6]) / 2

                # Update the arrow with the calculated start point and direction vector
                self.open3d_visualizer.update_arrow(
                    start_point, direction_vector, "human")
            #### undistroztion ####
            frame0 = cv2.undistort(frame0, self.mtx0_int,
                                   self.dist0_int, None, self.mtx0_int)
            frame1 = cv2.undistort(frame1, self.mtx1_int,
                                   self.dist1_int, None, self.mtx1_int)

            ##### robot detection #####
            result_frame0 = self.robot_detector.inference(frame0)
            result_frame1 = self.robot_detector.inference(frame1)

            points0 = self.robot_detector.get_keypoints(result_frame0)
            points1 = self.robot_detector.get_keypoints(result_frame1)

            robot_points_3d = self.robot_detector.triangulation(
                result_frame0, result_frame1, self.Proj_matrix0, self.Proj_matrix1, points0, points1)

            if self.robot_detector.previous_keypoints is not None:
                direction_vector = np.mean(
                    self.robot_detector.current_keypoints - self.robot_detector.previous_keypoints, axis=0)
                start_point = (
                    self.robot_detector.current_keypoints[5] + self.robot_detector.current_keypoints[6]) / 2

                # Update the arrow with the calculated start point and direction vector
                self.open3d_visualizer.update_arrow(
                    start_point, direction_vector, "robot")

            # visualizing the 3d points

            if len(robot_points_3d) > 0 and not first_frame:
                transformed_vertices = self.open3d_visualizer.align_mesh_to_keyponts(
                    robot_points_3d)

                self.open3d_visualizer.robot_mesh_vertex_cloud.points = o3d.utility.Vector3dVector(
                    transformed_vertices)
                self.open3d_visualizer.robot_mesh_vertex_cloud.paint_uniform_color([
                                                                                   0, 1, 1])
                self.open3d_visualizer.visualizer.update_geometry(
                    self.open3d_visualizer.robot_mesh_vertex_cloud)
            else:
                first_frame = False
                self.open3d_visualizer.robot_mesh_vertex_cloud.paint_uniform_color([
                                                                                   0, 1, 1])
                self.open3d_visualizer.visualizer.add_geometry(
                    self.open3d_visualizer.robot_mesh_vertex_cloud)

            self.open3d_visualizer.run_robot(robot_points_3d)
            self.open3d_visualizer.run_human(human_points_3d)

            self.open3d_visualizer.update_open3d()

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


if __name__ == "__main__":
    ins, ex = main()
    VideoManager = VideoManager(ins, ex)
    VideoManager.run()
