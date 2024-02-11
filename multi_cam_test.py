from pypylon import pylon
import threading
import json
import time
import cv2
import glob
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmengine import init_default_scope
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
import open3d as o3d
import numpy as np
import queue

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CameraThread:
    def __init__(self, camera_index, shared_data, shared_data_lock, trigger_mode=True, name=""):
        self.camera_index = camera_index
        self.shared_data = shared_data
        self.shared_data_lock = shared_data_lock
        self.trigger_mode = trigger_mode
        self.running = False
        self.thread = threading.Thread(target=self.run)
        self.cam_name = name
        self.image_queue = queue.Queue()
        # print(self.cam_name)
        # cv2.namedWindow(self.cam_name)

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if len(devices) <= camera_index:
            raise RuntimeError(f"No camera found at index {camera_index}")

        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[camera_index]))
        self.camera.Open()
        self.configure_camera()

    def configure_camera(self):
        # Set the camera to hardware trigger mode if required
        if self.trigger_mode:
            self.camera.TriggerMode.SetValue("On")
            # Assuming "Line1" is the trigger source, adjust as needed
            self.camera.TriggerSource.SetValue("Line1")
            self.camera.PixelFormat.SetValue("RGB8")
            self.camera.StreamGrabber.MaxTransferSize = 4194304
            # Other trigger configurations as necessary
            # e.g., self.camera.TriggerActivation.SetValue("RisingEdge")

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_latest_image(self):
        print(f'{self.cam_name} queue length: {self.image_queue.qsize()}')
        latest_image = None
        while not self.image_queue.empty():
            latest_image = self.image_queue.get_nowait()
        return latest_image

    def run(self):
        self.camera.MaxNumBuffer = 1
        # self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,
                                   pylon.GrabLoop_ProvidedByUser)
        while self.running and self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            print(f'{self.cam_name} grabbing image...')
            if grabResult.GrabSucceeded():
                timestamp = time.time()
                # img = grabResult.Array  # Assuming this is how you get the image array
                img = cv2.cvtColor(grabResult.GetArray(), cv2.COLOR_BGR2RGB)
                with self.shared_data_lock:
                    self.shared_data['frames'][self.camera_index] = (img, timestamp)
                # cv2.imshow(self.cam_name, img)
                self.image_queue.put(img)
            grabResult.Release()

        self.camera.StopGrabbing()


def get_pose_xyz(frame):
    pose_set = []
    for x, y, z in zip(frame[0], frame[1], frame[2]):
        pose_set.append(np.asarray([x, y, z])/1000)  # divide by 1000 if camera focal length is given in mm
    return np.asarray(pose_set)


def mark_kp(img, point_2d, color=(255, 0, 0)):
    x = point_2d[0]
    y = point_2d[1]
#     print(f'{x} - {y}')
    cv2.circle(img,
                  (int(round(x)), int(round(y))),
                  4,
                  color=color,
                  thickness=cv2.FILLED)

def process_mmdet_results(detector_results, cat_id):
    filtered_bboxes = [bbox for label_id, bbox in zip(detector_results.pred_instances.labels, detector_results.pred_instances.bboxes) if label_id == cat_id]
    return filtered_bboxes

def neural_network_process(detection_model, model, image0, image1):
    # Object detection
    init_default_scope('mmdet')

    det_result = inference_detector(detection_model, image0)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    person_bboxes0 = bboxes[nms(bboxes, 0.3), :4]

    det_result = inference_detector(detection_model, image1)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    person_bboxes1 = bboxes[nms(bboxes, 0.3), :4]

    # mmdet_results0 = inference_detector(detection_model, image0)
    # person_bboxes0 = process_mmdet_results(mmdet_results0, cat_id=0)  # assuming category ID for person is 1
    # mmdet_results1 = inference_detector(detection_model, image1)
    # person_bboxes1 = process_mmdet_results(mmdet_results1, cat_id=0)  # assuming category ID for person is 1

    # Process the image through your neural network model
    init_default_scope('mmpose')
    # print(type(person_bboxes0))
    if person_bboxes0 == [] or person_bboxes1 == []:
        return None, None
    else:
        results0 = inference_topdown(model, image0, person_bboxes0)
        results1 = inference_topdown(model, image1, person_bboxes1)

        data_samples0 = merge_data_samples(results0)
        data_samples1 = merge_data_samples(results1)

        # visualizer.add_datasample(
        #     'result',
        #     image0,
        #     data_sample=data_samples0,
        #     draw_gt=False,
        #     draw_heatmap=False,
        #     draw_bbox=True,
        #     show_kpt_idx=False,
        #     skeleton_style='mmpose',
        #     show=True,
        #     wait_time=0,
        #     kpt_thr=0.3)

        # pred_instances_0 = results0[0].pred_instances
        # pred_instances_1 = results1[0].pred_instances
        #
        # # print(pred_instances_1)
        # for pose in pred_instances_0.keypoints:
        #     for kp in pose:
        #         mark_kp(image0, kp)
        # # cv2.imwrite('debug/valami0.png', image0)
        # #
        # for pose in pred_instances_1.keypoints:
        #     for kp in pose:
        #         mark_kp(image1, kp)
        # cv2.imwrite('debug/valami1.png', image1)
        # print(pred_instances.bbox_scores)
        # print(pred_instances.keypoint_scores)
        # print(pred_instances.keypoints_visible)

        # 3D triangulation
        # Process the image pair here

        # vis.poll_events()
        # vis.update_renderer()
        return results0, results1


def run_triangulation(results0, results1, P0, P1, skeleton_cloud, lines, sphere_list, vis):
    left_pose = max(results0[0].pred_instances, key=lambda item: np.mean(item.keypoint_scores))
    right_pose = max(results1[0].pred_instances, key=lambda item: np.mean(item.keypoint_scores))

    points_4d_hom = cv2.triangulatePoints(P0, P1, np.asarray(left_pose.keypoints).squeeze().T,
                                          np.asarray(right_pose.keypoints).squeeze().T)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]

    points_3d = get_pose_xyz(points_3d)

    # print(points_3d)
    skeleton_cloud.points = o3d.utility.Vector3dVector(points_3d)
    lines.points = o3d.utility.Vector3dVector(points_3d)
    vis.update_geometry(skeleton_cloud)  # Update skeleton's transformation
    vis.update_geometry(lines)

    for point_idx, point in enumerate(range(17)):
        if point_idx > len(skeleton_cloud.points):
            # temporarly remove sphere visibility
            sphere = sphere_list[point_idx]
            update_sphere_position(sphere, np.array([0, 0, 0]))
        else:
            sphere = sphere_list[point_idx]
            # sphere.translate(pcd.points[point_idx])
            update_sphere_position(sphere, np.array(skeleton_cloud.points[point_idx]))
        vis.update_geometry(sphere)

    # vis.poll_events()
    # vis.update_renderer()

def update_sphere_position(sphere, new_center):
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

class CameraManager:
    def __init__(self, camera_int_params, camera_ext_params):
        self.shared_data = {'frames': {}, 'last_processed_timestamps': {}}
        self.shared_data_lock = threading.Lock()

        # Paths to object detection model config and checkpoint
        init_default_scope('mmdet')
        det_config = '/home/gables/Programs/mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
        det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'


        # self.camera0 = CameraThread(0, self.shared_data, self.shared_data_lock, trigger_mode=True, name="Door Camera")
        # self.camera1 = CameraThread(1, self.shared_data, self.shared_data_lock, trigger_mode=True, name="Window Camera")

        # self.opencv_window = OpenCVWindow()

        # model_cfg = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        # ckpt = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        model_cfg = '/home/gables/Programs/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py'
        ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth'
        # model_cfg = '/home/gables/Programs/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
        device = 'cuda'
        init_default_scope('mmpose')
        self.model = init_pose_estimator(model_cfg, ckpt, device=device, cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))

        # Initialize detection model
        init_default_scope('mmdet')
        self.detection_model = init_detector(det_config, det_checkpoint, device=device)
        mesh = o3d.io.read_triangle_mesh("meshes/temp.ply")

        self.mtx0_int = np.asarray(camera_int_params['cam0_K'])
        self.mtx1_int = np.asarray(camera_int_params['cam1_K'])
        self.dist0_int = np.asarray(camera_int_params['cam0_dist'])
        self.dist1_int = np.asarray(camera_int_params['cam1_dist'])

        self.R = np.asarray(camera_ext_params['R'])
        self.T = np.asarray(camera_ext_params['T'])

        self.Proj_matrix0 = np.dot(self.mtx0_int, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.Proj_matrix1 = np.dot(self.mtx1_int, np.hstack((self.R, self.T.reshape(-1, 1))))

        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(len(mesh.vertices))])
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()


        self.skeleton_cloud = o3d.geometry.PointCloud()
        # skeleton_cloud.points = o3d.utility.Vector3dVector(get_pose_xyz(video_poses[0]))

        # Optional: Customize the appearance of the keypoints
        keypoint_color = [0, 1, 0]  # Red color for keypoints
        self.skeleton_cloud.paint_uniform_color(keypoint_color)

        connections = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
                       (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

        self.lines = o3d.geometry.LineSet()
        self.lines.points = self.skeleton_cloud.points
        self.lines.lines = o3d.utility.Vector2iVector(connections)
        self.lines.paint_uniform_color([0, 1, 0])

        # Add the mesh, point cloud, and spinning cube to the visualizer
        self.vis.add_geometry(mesh)
        self.vis.add_geometry(self.lines)
        self.vis.add_geometry(self.skeleton_cloud)

        render_option = self.vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = False

        # vid0_path = 'video_test/cam_0_1510.mp4'
        # vid1_path = 'video_test/cam_1_1510.mp4'
        # vid0_path = 'video_test/cam_0_1687.mp4'
        # vid1_path = 'video_test/cam_1_1687.mp4'
        #
        # processing = True
        # vid0 = cv2.VideoCapture(vid0_path)
        # vid1 = cv2.VideoCapture(vid1_path)
        # self.points3D_list = []
        # print('processing video')
        #
        # out0 = cv2.VideoWriter(f'debug/RTMPose_overlay_corner_1687.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1200))
        # out1 = cv2.VideoWriter(f'debug/RTMPose_overlay_stand_1687.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1200))

        # # build visualizer
        # self.model.cfg.visualizer.radius = 3
        # self.model.cfg.visualizer.alpha = 0.8
        # self.model.cfg.visualizer.line_width = 1
        # init_default_scope('mmpose')
        # visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        # # the dataset_meta is loaded from the checkpoint and
        # # then pass to the model in init_pose_estimator
        # visualizer.set_dataset_meta(
        #     self.model.dataset_meta, skeleton_style='mmpose')

        # while processing:
        #     ret0, frame_0 = vid0.read()
        #     ret1, frame_1 = vid1.read()
        #     if ret0 and ret1:
        #         results0, results1 = neural_network_process(self.detection_model, self.model, frame_0, frame_1)
        #         # out0.write(frame_0)
        #         # out1.write(frame_1)
        #         if results0 is not None and results1 is not None:
        #             left_pose = max(results0[0].pred_instances, key=lambda item: np.mean(item.keypoint_scores))
        #             right_pose = max(results1[0].pred_instances, key=lambda item: np.mean(item.keypoint_scores))
        #
        #             points_4d_hom = cv2.triangulatePoints(self.Proj_matrix0, self.Proj_matrix1, np.asarray(left_pose.keypoints).squeeze().T,
        #                                                   np.asarray(right_pose.keypoints).squeeze().T)
        #             points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        #
        #             points_3d = get_pose_xyz(points_3d)
        #             self.points3D_list.append(points_3d)
        #
        #             self.skeleton_cloud.points = o3d.utility.Vector3dVector(points_3d)
        #             self.lines.points = o3d.utility.Vector3dVector(points_3d)
        #             self.vis.update_geometry(self.skeleton_cloud)  # Update skeleton's transformation
        #             self.vis.update_geometry(self.lines)
        #             self.vis.poll_events()
        #             self.vis.update_renderer()
        #
        #         else:
        #             self.points3D_list.append([])
        #     else:
        #         processing = False
        #         # vid0.release()
        #         # vid1.release()
        #         np.save('demo_points3d_list_1687.npy', self.points3D_list)
        #         print('video processing completed')
        self.current_frame_id = 0
        self.sphere_list = []
        for point_idx, point in enumerate(range(17)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # Adjust radius for size
            sphere.translate([0, 0, 0])
            sphere.paint_uniform_color([1, 0, 0])
            self.sphere_list.append(sphere)
            self.vis.add_geometry(sphere)

        num_cameras = 2
        self.cameras = pylon.InstantCameraArray(num_cameras)
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        for i, camera in enumerate(self.cameras):
            camera.Attach(tlFactory.CreateDevice(devices[i]))

        self.cameras.Open()

        # Configure both cameras for hardware triggering
        for cam in self.cameras:
            cam.TriggerMode.SetValue("Off")
            # cam.TriggerSource.SetValue("Line1")
            cam.PixelFormat.SetValue("RGB8")
            # Configure additional settings if required

        for cam in self.cameras:
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    def start(self):
        pass
        # self.camera0.start()
        # self.camera1.start()
        # self.opencv_window.start()

    def stop(self):
        pass
        # self.camera0.stop()
        # self.camera1.stop()
        # self.camera0.camera.Close()
        # self.camera1.camera.Close()
        # self.opencv_window.stop()

    def process_frames(self):
        # cv2.namedWindow('Acquisition', cv2.WINDOW_NORMAL)
        while True:
            start_time = time.time()
            image_list = []
            for idx, cam in enumerate(self.cameras):
                # if cam.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException):
                #     cam.ExecuteSoftwareTrigger()

                # Retrieve the grabbed image
                grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Convert to OpenCV format if necessary
                    image = grabResult.GetArray()

                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                    # Process the image as needed
                    # ...
                    image_list.append(image)

                grabResult.Release()
            # results0, results1 = neural_network_process(self.detection_model, self.model, image_list[0], image_list[1])
            # run_triangulation(results0, results1, self.Proj_matrix0, self.Proj_matrix1, self.skeleton_cloud, self.lines,
            #                   self.sphere_list, self.vis)

            res_img = np.hstack([image_list[0], image_list[1]])
            print(res_img.shape)
            cv2.imshow('Acquisition', res_img)

            # elapsed_time = time.time() - start_time
            # # Calculate frames per second
            # fps = 1 / elapsed_time
            #
            # # Print FPS every 10 frames
            # print(f"FPS: {fps:.2f}")
            # cv2.imshow('Acquisition', np.hstack([image_list[0], image_list[1]]))

            # with self.shared_data_lock:
            #     frame_data = self.shared_data['frames']
            #     if 0 in frame_data and 1 in frame_data:
            #         img0, time0 = frame_data[0]
            #         img1, time1 = frame_data[1]
            #
            #         last_processed_time0 = self.shared_data['last_processed_timestamps'].get(0, 0)
            #         last_processed_time1 = self.shared_data['last_processed_timestamps'].get(1, 0)
            #
            #         # cam0_img = self.camera0.get_latest_image()
            #         # cam1_img = self.camera1.get_latest_image()
            #         # Process the images only if their timestamps have changed since the last processing
            #         if time0 != last_processed_time0 or time1 != last_processed_time1:
            #             # print(f'{time0} -- {self.current_frame_id}')
            #             self.current_frame_id +=1
            #             self.shared_data['last_processed_timestamps'][0] = time0
            #             self.shared_data['last_processed_timestamps'][1] = time1
            #
            #             # print(f'time0: {time0} -- time1: {time1}')
            #             # print(f'img0 shape {img0.shape}')
            #             # print(f'img1 shape {img1.shape}')
            #             # img0 = cv2.imread('video_test/image_0_800.png')
            #             # img1 = cv2.imread('video_test/image_1_800.png')
            #             # img0_undistorted = cv2.undistort(img0, self.mtx0_int, self.dist0_int, None, self.mtx0_int)
            #             # img1_undistorted = cv2.undistort(img1, self.mtx1_int, self.dist1_int, None, self.mtx1_int)
            #
            #             # cv2.imshow('Window', np.hstack([i.camera.RetrieveResult(5000).GetArray() for i in [self.camera0, self.camera1]]))
            #             results0, results1 = neural_network_process(self.detection_model, self.model, img0, img1)
            #             run_triangulation(results0, results1, self.Proj_matrix0, self.Proj_matrix1, self.skeleton_cloud, self.lines, self.sphere_list, self.vis)
            #
            #
            #             # self.opencv_window.update_image(cv2.resize(img0_undistorted, (320, 240), interpolation = cv2.INTER_AREA))
            #             # ...
            #         # else:
            #         #     print('no new image')

            self.vis.poll_events()
            self.vis.update_renderer()
            # time.sleep(0.03)  # Adjust the sleep time as necessary


def calibrate_intrinsics(image0, image1, chessboard_size = (9,6), square_size=0.07):
    square_size = square_size  # 70 mm
    chessboard_size = chessboard_size  # Number of inner corners (width, height)

    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world coordinates
    img_points0 = []  # 2D points in camera 1 image
    img_points1 = []  # 2D points in camera 2 image

    # Create a list of object points for the chessboard
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    obj_points.append(objp)
    # Loop through your image pairs (images from each camera)
    print('Gathering Intrinsic Camera Parameters...')
    img = None
    for i, img in enumerate([image0, image1]):  # Replace '1' with the number of image pairs you have
        # img = cv2.imread(f'cam{i}_{frame_id}_int.png')  # Load the image
        # print(f'{i}. img shape: {img.shape}')
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        # print(i, ret)
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), criteria)

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


def calibrate_extrinsics(camera_int_params, chessboard_size = (9,6), square_size = 70):
    mtx0_int = camera_int_params[0]
    mtx1_int = camera_int_params[2]
    dist0_int = camera_int_params[1]
    dist1_int = camera_int_params[3]

    # Chessboard settings
    chessboard_size = chessboard_size  # Number of inner corners per a chessboard row and column (8x6 board)
    square_size = square_size # Size of a chessboard square in mm

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints_left = []  # 2d points in image plane for left camera
    imgpoints_right = []  # 2d points in image plane for right camera

    # Load images for left and right camera
    # images_left = glob.glob('cam0_*1.png')  # Update path
    # images_right = glob.glob('cam1_*1.png')  # Update path

    images_left = glob.glob('extrinsic/image_0_*1.png')  # Update path
    images_left2 = glob.glob('extrinsic/image_0_*2.png')  # Update path
    images_left3 = glob.glob('extrinsic/image_0_*3.png')  # Update path
    images_left4 = glob.glob('extrinsic/image_0_*4.png')  # Update path
    images_right = glob.glob('extrinsic/image_1_*1.png')  # Update path
    images_right2 = glob.glob('extrinsic/image_1_*2.png')  # Update path
    images_right3 = glob.glob('extrinsic/image_1_*3.png')  # Update path
    images_right4 = glob.glob('extrinsic/image_1_*4.png')  # Update path
    images_left += images_left2 + images_left3 + images_left4
    images_right += images_right2 + images_right3 + images_right4

    gray_left = None
    # for left_img, right_img in tqdm.tqdm(zip(sorted(images_left), sorted(images_right)), desc="Camera Stereo Calibration for Extrinsic Parameters"):
    for left_img, right_img in zip(sorted(images_left), sorted(images_right)):
        print(f'left img: {left_img} \t\t right img: {right_img}')
        img_left = cv2.imread(left_img)
        img_right = cv2.imread(right_img)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)

        # If found, add object points and image points
        if ret_left and ret_right:
            objpoints.append(objp)
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2_left)
            imgpoints_right.append(corners2_right)

    # Stereo calibration
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx0_int, dist0_int, mtx1_int, dist1_int, gray_left.shape[::-1])

    print("Rotation matrix:\n", R)
    print("Translation vector:\n", T)
    return R, T

def main():
    chessboard_size = (9, 6)
    square_size = 70
    door_cam_internal_img = cv2.imread(f'intrinsic/door.jpg')  # Load the image
    window_cam_internal_img = cv2.imread(f'intrinsic/window2.jpg')  # Load the image
    if os.path.exists(f'intrinsic/config.json'):
        print(f'loading intrinsics from config...')
        with open('intrinsic/config.json', 'r') as json_file:
            camera_int_params = json.load(json_file)
    else:
        print(f"The file 'intrinsic/config.json' does not exist. Creating it...")
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
    if os.path.exists(f'extrinsic/config.json'):
        print(f'loading extrinsic from config...')
        with open('extrinsic/config.json', 'r') as json_file:
            camera_ext_params = json.load(json_file)
    else:
        print(f"The file 'extrinsic/config.json' does not exist. Creating it...")
        camera_ext_params = calibrate_extrinsics(camera_int_params, chessboard_size, square_size)
        data = {'R': camera_ext_params[0].tolist(),
                'T': camera_ext_params[1].tolist()}
        with open('extrinsic/config.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        with open('extrinsic/config.json', 'r') as json_file:
            camera_ext_params = json.load(json_file)

    print(f'camera_int_params:\n{camera_int_params}')
    print(f'camera_ext_params:\n{camera_ext_params}')



    manager = CameraManager(camera_int_params, camera_ext_params)
    try:
        manager.start()
        manager.process_frames()  # This will continuously process frames
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()

if __name__ == "__main__":
    main()
