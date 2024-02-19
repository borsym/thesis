import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmengine import init_default_scope
from mmpose.evaluation.functional import nms


class HumanDetector:
    """
    Class for detecting and tracking human poses in a video feed.

    Args:
        camera_int_params (dict): Camera intrinsic parameters.
        camera_ext_params (dict): Camera extrinsic parameters.
    """

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
        """
        Selects bounding boxes of persons detected in the given image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The selected bounding boxes of persons.
        """
        det_result = inference_detector(self.detection_model, image)
        pred_instance = det_result.pred_instances.cpu().numpy()

        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        # select human bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                       pred_instance.scores > 0.3)]

        return bboxes[nms(bboxes, 0.3), :4]

    def mark_kp(self, image, x, y, color=(0, 0, 255)):
        """
        Marks a keypoint on the given image at the specified coordinates.

        Args:
            image (numpy.ndarray): The image on which to mark the keypoint.
            x (float): The x-coordinate of the keypoint.
            y (float): The y-coordinate of the keypoint.
            color (tuple, optional): The color of the marker. Defaults to (0, 0, 255).

        Returns:
            None
        """
        cv2.circle(image, (int(round(x)), int(round(y))),
                   10, color=color, thickness=cv2.FILLED)

    def mark_image(self, image, results, rgb):
        """
        Marks the keypoints and draws connections on the given image based on the pose estimation results.

        Args:
            image (numpy.ndarray): The input image to mark keypoints on.
            results (list): The pose estimation results.
            rgb (tuple): The RGB color tuple for drawing connections.

        Returns:
            numpy.ndarray: The image with marked keypoints and connections.
        """
        for pose in results[0].pred_instances.keypoints:
            for kp in pose:
                x, y = kp
                self.mark_kp(image, x, y)
            for start_point, end_point in self.connections:
                cv2.line(image, [int(pose[start_point][0]), int(pose[start_point][1])], [int(
                    pose[end_point][0]), int(pose[end_point][1])], color=rgb, thickness=3)
        return image

    def inference(self, image0, image1):
        """
        Perform inference on two input images.

        Args:
            image0 (numpy.ndarray): The first input image.
            image1 (numpy.ndarray): The second input image.

        Returns:
            tuple: A tuple containing the inference results for the first image, the inference results for the second image,
                   the marked first image, and the marked second image.
        """
        init_default_scope('mmdet')
        person_bboxes0 = self.select_person(image0)
        person_bboxes1 = self.select_person(image1)

        init_default_scope('mmpose')

        if person_bboxes0 == [] or person_bboxes1 == []:
            print("no person found")
            return None, None

        results0 = inference_topdown(self.model, image0, person_bboxes0)
        results1 = inference_topdown(self.model, image1, person_bboxes1)

        self.mark_image(image0, results0, (0, 255, 0))
        self.mark_image(image1, results1, (0, 255, 0))

        return results0, results1, image0, image1

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

    def save_to_npy(self):
        np.save("all_points_3d.npy", np.array(
            self.all_points_3d, dtype=object))

    def triangulation(self, results0, results1, P0, P1):
        """
        Perform triangulation to estimate 3D coordinates of keypoints.

        Args:
            results0 (list): List of pose estimation results from the left camera.
            results1 (list): List of pose estimation results from the right camera.
            P0 (numpy.ndarray): Projection matrix of the left camera.
            P1 (numpy.ndarray): Projection matrix of the right camera.

        Returns:
            numpy.ndarray: Array of 3D coordinates of keypoints.

        """
        left_pose = max(results0[0].pred_instances,
                        key=lambda item: np.mean(item.keypoint_scores))
        right_pose = max(results1[0].pred_instances,
                         key=lambda item: np.mean(item.keypoint_scores))

        points_4d_hom = cv2.triangulatePoints(P0, P1, np.asarray(left_pose.keypoints).squeeze().T,
                                              np.asarray(right_pose.keypoints).squeeze().T)

        points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
        points_3d = self.get_pose_xyz(points_3d)
        if self.current_keypoints is not None:
            self.previous_keypoints = self.current_keypoints
        self.current_keypoints = points_3d
        return points_3d
