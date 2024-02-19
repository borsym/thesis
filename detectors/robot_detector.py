import cv2
import numpy as np
import json
import time
from ultralytics import YOLO


class RobotDetector:
    """
    Class for detecting and tracking robots in a video feed.

    Attributes:
        model (YOLO): The object detection model used for robot detection.
        skeleton (list): The skeleton connections between robot keypoints.
        previous_keypoints (numpy.ndarray): The keypoints of the previous frame.
        current_keypoints (numpy.ndarray): The keypoints of the current frame.
    """

    def __init__(self) -> None:
        self.model = YOLO('D:/thesis/realtime_update/weights/best_new.pt')
        self.skeleton = self.get_skeleton()
        self.previous_keypoints = None
        self.current_keypoints = None

    def get_skeleton(self):
        """
        Retrieves the skeleton data from the 'robot-keypoints-connection.json' file.

        Returns:
            dict: The skeleton data.
        """
        with open("D:/thesis/realtime_update/robot-keypoints-connection.json", 'r') as file:
            annotations_data = json.load(file)
        return annotations_data['skeleton']

    def inference(self, frame0, frame1):
        return self.model(frame0), self.model(frame1)

    def get_keypoint(self, result):
        return result[0].keypoints.xy[0]

    def get_keypoints(self, result0, result1):
        """
        Get the xy-coordinates of the keypoints from the given result.

        Args:
            result (list): List of results containing keypoints.

        Returns:
            list: List of xy-coordinates of the keypoints.
        """
        return self.get_keypoint(result0), self.get_keypoint(result1)

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
        """
        Converts the frame coordinates to pose coordinates in meters.

        Args:
            frame (numpy.ndarray): The frame coordinates in millimeters.

        Returns:
            numpy.ndarray: The pose coordinates in meters.
        """
        pose_set = []
        for x, y, z in zip(frame[0], frame[1], frame[2]):
            # divide by 1000 if camera focal length is given in mm
            pose_set.append(np.asarray([x, y, z])/1000)
        return np.asarray(pose_set)

    def triangulation(self, results0, results1, P0, P1, points0, points1):
        """
        Performs triangulation to estimate the 3D coordinates of keypoints in a stereo vision system.

        Args:
            results0 (list): List of keypoints detected in the left image.
            results1 (list): List of keypoints detected in the right image.
            P0 (numpy.ndarray): Projection matrix of the left camera.
            P1 (numpy.ndarray): Projection matrix of the right camera.
            points0 (list): List of 2D coordinates of keypoints in the left image.
            points1 (list): List of 2D coordinates of keypoints in the right image.

        Returns:
            list: List of 3D coordinates of keypoints in the world coordinate system.
        """
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
