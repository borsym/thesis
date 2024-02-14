import open3d as o3d
import cv2
import threading
import time
import numpy as np
import json
import os
import glob
import time
import copy

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


nerf_robot_keypoints = np.array([
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


def main():
    chessboard_size = (9, 6)
    square_size = 70  # one square size in mm

    if os.path.exists(f'intrinsic/intrinsics.json'):
        print(f'loading intrinsics from config...')
        with open('intrinsic/intrinsics.json', 'r') as json_file:
            camera_int_params = json.load(json_file)
    if os.path.exists(f'extrinsic/extrinsics.json'):
        print(f'loading extrinsic from config...')
        with open('extrinsic/extrinsics.json', 'r') as json_file:
            camera_ext_params = json.load(json_file)

    print(f'camera_int_params:\n{camera_int_params}')
    print(f'camera_ext_params:\n{camera_ext_params}')

    return camera_int_params, camera_ext_params


camera_int_params, camera_ext_params = main()
robot_mesh = o3d.io.read_triangle_mesh(
    "D:/thesis/realtime_update/meshes/save4.ply")
vertex_cloud_for_robot = np.array(robot_mesh.vertices)[::20000, :]


mtx0_int = np.asarray(camera_int_params['cam0_K'])
mtx1_int = np.asarray(camera_int_params['cam1_K'])
dist0_int = np.asarray(camera_int_params['cam0_dist'])
dist1_int = np.asarray(camera_int_params['cam1_dist'])

R = np.asarray(camera_ext_params['R'])
T = np.asarray(camera_ext_params['T'])

Proj_matrix0 = np.dot(
    mtx0_int, np.hstack((np.eye(3), np.zeros((3, 1)))))
Proj_matrix1 = np.dot(
    mtx1_int, np.hstack((R, T.reshape(-1, 1))))


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


robot_detector = RobotDetector()


def align_mesh_to_keyponts(keypoints):
    mesh_vertex_cloud = vertex_cloud_for_robot

    centroid_robot_keypoints = np.mean(keypoints, axis=0)
    centroid_nerf_keypoints = np.mean(nerf_robot_keypoints, axis=0)

    centered_keypoints = keypoints - centroid_robot_keypoints
    centered_mesh_vertex_cloud = mesh_vertex_cloud - centroid_nerf_keypoints
    centered_nerf_keypoints = nerf_robot_keypoints - centroid_nerf_keypoints

    # Perform Procrustes analysis
    _, mtx2, _ = procrustes(centered_keypoints, centered_nerf_keypoints)

    transformation_matrix = np.linalg.lstsq(
        centered_nerf_keypoints, mtx2, rcond=None)[0]
    transformed_mesh_vertex_cloud = centered_mesh_vertex_cloud @ transformation_matrix.T

    # transformed_mesh_vertex_cloud *= 1.2

    # Translate the transformed mesh vertices to match the robot keypoints centroid
    transformed_mesh_vertex_cloud += centroid_robot_keypoints
    # print(f'transformed_mesh_vertex_cloud: {transformed_mesh_vertex_cloud}')
    # print(
    #     f'transformed_mesh_vertex_cloud: {transformed_mesh_vertex_cloud.shape}')
    # # visualize in mathplotlib the transformed_mesh_vertex_cloud and the keypoints
    # save the value into a txt
    return transformed_mesh_vertex_cloud


visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

tmp_mesh = o3d.geometry.PointCloud()
visualizer.add_geometry(tmp_mesh)


def run():
    first_frame = True
    video0 = cv2.VideoCapture(
        'D:/thesis/realtime_update/recordings/Scenario1/Cam1/out.mp4')
    video1 = cv2.VideoCapture(
        'D:/thesis/realtime_update/recordings/Scenario1/Cam2/out.mp4')
    i = 0
    while True:
        i += 1
        print(f'Frame: {i}')
        start_time = time.time()
        # Read frames from each video
        ret0, frame0 = video0.read()
        ret1, frame1 = video1.read()

        # Check if either frame could not be read -> end of video
        if not ret0 or not ret1:
            break

        frame0 = cv2.undistort(frame0, mtx0_int,
                               dist0_int, None, mtx0_int)
        frame1 = cv2.undistort(frame1, mtx1_int,
                               dist1_int, None, mtx1_int)

        ##### robot detection #####
        result_frame0 = robot_detector.inference(frame0)
        result_frame1 = robot_detector.inference(frame1)

        points0 = robot_detector.get_keypoints(result_frame0)
        points1 = robot_detector.get_keypoints(result_frame1)

        robot_points_3d = robot_detector.triangulation(
            result_frame0, result_frame1, Proj_matrix0, Proj_matrix1, points0, points1)

        if len(robot_points_3d) > 0 and not first_frame:
            print("belepeett")
            # visualizer.remove_geometry(tmp_mesh, reset_bounding_box=False)
            transformed_vertices = align_mesh_to_keyponts(robot_points_3d)
            tmp_mesh.points = o3d.utility.Vector3dVector(transformed_vertices)
            visualizer.update_geometry(tmp_mesh)
        else:
            print("No points to update.")
            visualizer.add_geometry(tmp_mesh)
            first_frame = False
        visualizer.poll_events()
        visualizer.update_renderer()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(robot_points_3d[:, 0], robot_points_3d[:, 1],
        #            robot_points_3d[:, 2], c='r', marker='o')
        # ax.scatter(transformed_vertices[:, 0], transformed_vertices[:,
        #                                                             1], transformed_vertices[:, 2], c='b', marker='o')
        # plt.show()

        print(f'FPS: {1/(time.time()-start_time)}')

    video0.release()
    video1.release()


run()
