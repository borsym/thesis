from camera_calibration.camera_calibration import read_camera_parameters
import cv2
import time
import numpy as np
import time

from detectors.human_detector import HumanDetector
from detectors.robot_detector import RobotDetector
from visualizers.visualizer import Visualizer
from scipy.stats import zscore


class CameraParameters:
    def __init__(self, camera_int_params, camera_ext_params):
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


class VideoManager:
    """
    Initializes the video feed object.

    Args:
        camera_int_params (dict): Dictionary containing camera intrinsic parameters.
        camera_ext_params (dict): Dictionary containing camera extrinsic parameters.
    """

    def __init__(self, camera_int_params, camera_ext_params):
        # Initialization code remains mostly the same...
        # Initialize your models and visualizer here as before
        # Open video files instead of starting camera threads
        self.camera_params = CameraParameters(
            camera_int_params, camera_ext_params)
        self.video0 = cv2.VideoCapture(
            'D:/thesis/realtime_update/recordings/Scenario1/Cam1/out.mp4')
        self.video1 = cv2.VideoCapture(
            'D:/thesis/realtime_update/recordings/Scenario1/Cam2/out.mp4')
        self.visualizer = Visualizer()
        self.robot_detector = RobotDetector()
        self.human_detector = HumanDetector(
            camera_int_params, camera_ext_params)

    def read_frames(self):
        ret0, frame0 = self.video0.read()
        ret1, frame1 = self.video1.read()
        return ret0, ret1, frame0, frame1

    def undistort_frames(self, frame0, frame1):
        frame0 = cv2.undistort(frame0, self.camera_params.mtx0_int,
                               self.camera_params.dist0_int, None, self.camera_params.mtx0_int)
        frame1 = cv2.undistort(frame1, self.camera_params.mtx1_int,
                               self.camera_params.dist1_int, None, self.camera_params.mtx1_int)
        return frame0, frame1

    def run(self):
        while True:
            start_time = time.time()
            # Read frames from each video
            ret0, ret1, frame0, frame1 = self.read_frames()

            # Check if either frame could not be read -> end of video
            if not ret0 or not ret1:
                break

            #### undistroztion ####
            frame0, frame1 = self.undistort_frames(frame0, frame1)

            # ##### human detection #####
            result0, result1, frame0, frame1 = self.human_detector.inference(
                frame0, frame1)

            human_points_3d = self.human_detector.triangulation(
                result0, result1, self.camera_params.Proj_matrix0, self.camera_params.Proj_matrix1)

            if self.human_detector.previous_keypoints is not None:
                # Calculate the direction vector and start point
                direction_vector = np.mean(
                    self.human_detector.current_keypoints - self.human_detector.previous_keypoints, axis=0)
                start_point = (
                    self.human_detector.current_keypoints[5] + self.human_detector.current_keypoints[6]) / 2

                # Update the arrow with the calculated start point and direction vector
                self.visualizer.update_arrow(
                    start_point, direction_vector, "human")

            ##### robot detection #####
            result_frame0, result_frame1 = self.robot_detector.inference(
                frame0, frame1)

            points0, points1 = self.robot_detector.get_keypoints(
                result_frame0, result_frame1)

            robot_points_3d = self.robot_detector.triangulation(
                result_frame0, result_frame1, self.camera_params.Proj_matrix0, self.camera_params.Proj_matrix1, points0, points1)

            if self.robot_detector.previous_keypoints is not None and len(robot_points_3d) > 0:
                direction_vector = np.mean(
                    self.robot_detector.current_keypoints - self.robot_detector.previous_keypoints, axis=0)
                start_point = (
                    self.robot_detector.current_keypoints[5] + self.robot_detector.current_keypoints[6]) / 2

                # Update the arrow with the calculated start point and direction vector
                self.visualizer.update_arrow(
                    start_point, direction_vector, "robot")

            self.visualizer.run_human(human_points_3d)

            if len(robot_points_3d) == 0:
                continue

            _, _, transformation = self.visualizer.procrustes_mine(
                robot_points_3d, self.visualizer.nerf_robot_keypoints, scaling=True, reflection='best')

            initial_transformation_matrix = np.eye(4)
            initial_transformation_matrix[:3,
                                          :3] = transformation['rotation'] * transformation['scale']
            initial_transformation_matrix[:3,
                                          3] = transformation['translation']
            # remove previous transformation
            self.visualizer.robot_mesh.transform(
                self.visualizer.previous_transfomation)
            # add new transformation
            self.visualizer.robot_mesh.transform(
                initial_transformation_matrix)
            # save the previous transformation inverse
            self.visualizer.previous_transfomation = np.linalg.inv(
                initial_transformation_matrix)

            self.visualizer.visualizer.update_geometry(
                self.visualizer.robot_mesh)
            self.visualizer.run_robot(robot_points_3d)

            self.visualizer.update_open3d()

            elapsed_time = time.time() - start_time
            # Calculate frames per second
            fps = 1 / elapsed_time

            # Print FPS every 10 frames
            print(f"FPS: {fps:.2f}")
        self.video0.release()
        self.video1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    chessboard_size = (9, 6)
    square_size = 70  # one square size in mm
    ins, ex = read_camera_parameters(chessboard_size, square_size)
    VideoManager = VideoManager(ins, ex)
    VideoManager.run()
