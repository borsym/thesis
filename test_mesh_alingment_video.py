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
from scipy.linalg import orthogonal_procrustes

# erre kene hasznalni mindig? TODO
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

first_robot_keypoints = np.array([
    [-1.5052,  0.3022,  5.6128],
    [-1.3801,  0.33318,  5.8066],
    [-1.1815,  0.3669,  5.1104],
    [-1.0278,  0.39118,  5.3233],
    [-1.502,  0.037726,  5.7502],
    [-1.4306,  0.10683,  5.7399],
    [-1.489,  0.10121,  5.6533],
    [-1.4394, -0.22139,  5.5396],
    [-1.4233, -0.36347,  5.5755],
    [-1.3666, -0.21828,  5.6215],
    [-1.2277,  0.20028,  5.4801],
    [-1.3271,  0.19053,  5.3658],
    [-1.0123,  0.18271,  5.1322],
    [-1.0651,  0.17497,  5.0526],
    [-1.0838, -0.14808,  5.0171],
    [-1.0229, -0.14653,  5.1043],
    [-0.99055, -0.12007,  4.969]])

second_robot_keypoints = np.array([
    [-1.5047,    0.30117,     5.6124],
    [-1.3917,    0.33586,     5.8115],
    [-1.1736,    0.36516,     5.1065],
    [-1.0317,    0.39315,     5.3238],
    [-1.5087,   0.038508,     5.7529],
    [-1.4395,    0.10826,     5.7437],
    [-1.4919,    0.10144,     5.6542],
    [-1.4421,   -0.22109,     5.5405],
    [-1.4287,   -0.36256,     5.5776],
    [-1.3747,   -0.21713,     5.6246],
    [-1.2334,    0.20253,     5.4818],
    [-1.3245,    0.19054,     5.3645],
    [-1.0119,    0.18405,     5.1308],
    [-1.0605,    0.17526,     5.0501],
    [-1.0791,   -0.14771,     5.0145],
    [-1.0243,   -0.14501,     5.1037],
    [-0.98723,   -0.11894,     4.9668]
])

third_robot_keypoints = np.array([
    [-1.5132,   0.30316,    5.6161],
    [-1.3919,   0.33585,    5.8115],
    [-1.1765,   0.36538,    5.1075],
    [-1.0263,   0.39105,    5.3211],
    [-1.5145,  0.039313,    5.7557],
    [-1.4426,   0.10865,    5.7452],
    [-1.4987,   0.10245,    5.6574],
    [-1.4484,  -0.22063,    5.5433],
    [-1.4338,  -0.36238,      5.58],
    [-1.3775,  -0.21718,    5.6258],
    [-1.2318,   0.20177,    5.4806],
    [-1.3283,   0.19099,    5.3659],
    [-1.0082,    0.1827,    5.1293],
    [-1.06,   0.17464,    5.0498],
    [-1.0789,  -0.14804,    5.0145],
    [-1.0208,  -0.14576,    5.1023],
    [-0.98475,  -0.11978,     4.966]
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
# robot_mesh = o3d.io.read_triangle_mesh(
#     "D:/thesis/realtime_update/meshes/save4.ply")
# vertex_cloud_for_robot = np.array(robot_mesh.vertices)[::20000, :]


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


def align_mesh_to_keypoints(keypoints, nerf_robot):

    # Calculate the centroid of both sets of keypoints
    centroid_robot_keypoints = np.mean(keypoints, axis=0)
    centroid_nerf_keypoints = np.mean(nerf_robot, axis=0)

    # Center the keypoints by subtracting their respective centroids
    centered_keypoints = keypoints - centroid_robot_keypoints
    centered_nerf_keypoints = nerf_robot - centroid_nerf_keypoints

    # Perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(
        centered_keypoints, centered_nerf_keypoints)

    # The rotation matrix is already computed by the procrustes analysis
    R = mtx1.T @ mtx2

    # The translation vector is the difference between centroids
    translation = centroid_robot_keypoints - centroid_nerf_keypoints @ R

    # Create a full 4x4 transformation matrix
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = R
    transform_4x4[:3, 3] = translation

    return transform_4x4


def orthogonal_procrustes_analysis(A, B):
    """
    Perform orthogonal Procrustes analysis to find the rotation (and translation)
    that best aligns two sets of points in a least-squares sense without scaling.

    Parameters:
    A -- the first set of points
    B -- the second set of points to align to A

    Returns:
    R -- the optimal rotation matrix
    t -- the optimal translation vector
    """
    # Calculate centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the points
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the matrix H
    H = A_centered.T @ B_centered

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = U @ Vt

    # Ensure a proper rotation (det(R) == 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Compute translation vector
    t = centroid_B - centroid_A @ R

    return R, t


def get_skeleton_connection_robot():
    with open("D:/thesis/realtime_update/robot-keypoints-connection.json", 'r') as file:
        annotations_data = json.load(file)
    skeleton_connections = annotations_data['skeleton']
    # shift the indices by 1 to match the 0-based indexing in open3d
    skeleton_connections = [[x - 1, y - 1]
                            for x, y in skeleton_connections]
    return skeleton_connections


#############
#############
#############
# TODO NEM FORDUL A MESH A VIDEO_FEED_FINALBEN SZOVAL MAS MEGOLDAS KELL AZ ICP NEM ELEG JO
#############
#############
#############

# Example of adjusting the scale for the first set of aligned keypoints
def calibrate_by_procrustes(points3d, camera, gt):
    """Calibrates the predictied 3d points by Procrustes algorithm.

    This function estimate an orthonormal matrix for aligning the predicted 3d
    points to the ground truth. This orhtonormal matrix is computed by
    Procrustes algorithm, which ensures the global optimality of the solution.
    """
    # Shift the center of points3d to the origin
    if camera is not None:
        singular_value = np.linalg.norm(camera, 2)
        camera = camera / singular_value
        points3d = points3d * singular_value
    scale = np.linalg.norm(gt) / np.linalg.norm(points3d)
    points3d = points3d * scale

    U, s, Vh = np.linalg.svd(points3d.T.dot(gt))
    rot = U.dot(Vh)
    if camera is not None:
        return points3d.dot(rot), rot.T.dot(camera)
    else:
        return points3d.dot(rot), None


def procrustes_mine(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def run():
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    robot_mesh = o3d.io.read_triangle_mesh(
        "D:/thesis/realtime_update/meshes/save4.ply", True)
    robot_mesh.compute_vertex_normals()
    cumulative_transformation_matrix = np.eye(4)

    # Perform initial alignment using Procrustes analysis
    _, transformed_nerf, transformation = procrustes_mine(
        first_robot_keypoints, nerf_robot_keypoints, scaling=True, reflection='best')

    initial_transformation_matrix = np.eye(4)
    initial_transformation_matrix[:3,
                                  :3] = transformation['rotation'] * transformation['scale']
    initial_transformation_matrix[:3, 3] = transformation['translation']

    prev_transformation = np.linalg.inv(initial_transformation_matrix)

    # Apply the initial transformation to the mesh
    robot_mesh.transform(initial_transformation_matrix)

    # Prepare visualizer geometries
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_robot_keypoints)
    pcd.paint_uniform_color([1, 0, 0])

    prev = o3d.geometry.PointCloud()
    prev.points = o3d.utility.Vector3dVector(transformed_nerf)
    prev.paint_uniform_color([0, 1, 0])

    visualizer.add_geometry(robot_mesh)
    visualizer.add_geometry(pcd)
    visualizer.add_geometry(prev)

    # Open video streams
    video0 = cv2.VideoCapture(
        'D:/thesis/realtime_update/recordings/Scenario1/Cam1/out.mp4')
    video1 = cv2.VideoCapture(
        'D:/thesis/realtime_update/recordings/Scenario1/Cam2/out.mp4')

    # Initialize a variable to store the previous frame keypoints
    prev_frame = None
    while True:
        # Read frames from each video
        ret0, frame0 = video0.read()
        ret1, frame1 = video1.read()

        # Check if either frame could not be read -> end of video
        if not ret0 or not ret1:
            break

        # Undistort frames
        frame0 = cv2.undistort(frame0, mtx0_int, dist0_int, None, mtx0_int)
        frame1 = cv2.undistort(frame1, mtx1_int, dist1_int, None, mtx1_int)

        # Robot detection and triangulation
        result_frame0 = robot_detector.inference(frame0)
        result_frame1 = robot_detector.inference(frame1)

        points0 = robot_detector.get_keypoints(result_frame0)
        points1 = robot_detector.get_keypoints(result_frame1)

        robot_points_3d = robot_detector.triangulation(
            result_frame0, result_frame1, Proj_matrix0, Proj_matrix1, points0, points1)

        # If robot is lost, continue to the next frame
        if len(robot_points_3d) == 0:
            print("lost robot")
            continue

        # Perform Procrustes analysis between the current and previous frames
        print("transformed_nerf", transformed_nerf)
        _, transformed_nerf, transformation = procrustes_mine(
            robot_points_3d, nerf_robot_keypoints, scaling=True, reflection='best')

        # Extract transformation matrix from the Procrustes result
        current_transformation_matrix = np.eye(4)
        current_transformation_matrix[:3,
                                      :3] = transformation['rotation'] * transformation['scale']
        current_transformation_matrix[:3, 3] = transformation['translation']

        robot_mesh.transform(prev_transformation)
        robot_mesh.transform(current_transformation_matrix)
        prev_transformation = np.linalg.inv(current_transformation_matrix)

        pcd.points = o3d.utility.Vector3dVector(robot_points_3d)
        prev.points = o3d.utility.Vector3dVector(transformed_nerf)

        visualizer.update_geometry(pcd)
        visualizer.update_geometry(robot_mesh)
        visualizer.update_geometry(prev)

        visualizer.poll_events()
        visualizer.update_renderer()

    # Release resources
    video0.release()
    video1.release()
    visualizer.destroy_window()


run()
