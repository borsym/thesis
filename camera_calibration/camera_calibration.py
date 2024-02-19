import cv2
import numpy as np
import glob
import json
import os


def calibrate_intrinsics(image0, image1, chessboard_size=(9, 6), square_size=0.07):
    """
    Calibrates the intrinsic camera parameters using chessboard images.

    Args:
        image0 (numpy.ndarray): Image from camera 1.
        image1 (numpy.ndarray): Image from camera 2.
        chessboard_size (tuple, optional): Number of inner corners (width, height) of the chessboard. Defaults to (9, 6).
        square_size (float, optional): Size of each square on the chessboard in real world coordinates. Defaults to 0.07.

    Returns:
        tuple: A tuple containing the intrinsic camera parameters for camera 1 and camera 2.
               The tuple has the following format: (mtx0_int, dist0_int, mtx1_int, dist1_int).
               - mtx0_int (numpy.ndarray): Camera matrix for camera 1.
               - dist0_int (numpy.ndarray): Distortion coefficients for camera 1.
               - mtx1_int (numpy.ndarray): Camera matrix for camera 2.
               - dist1_int (numpy.ndarray): Distortion coefficients for camera 2.
    """
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
    """
    Calibrates the extrinsic parameters of a stereo camera system using a chessboard pattern.

    Args:
        camera_int_params (list): List of camera intrinsic parameters [mtx0_int, dist0_int, mtx1_int, dist1_int].
        chessboard_size (tuple, optional): Number of inner corners per a chessboard row and column. Defaults to (9, 6).
        square_size (int, optional): Size of a chessboard square in mm. Defaults to 70.

    Returns:
        tuple: Tuple containing the rotation matrix (R) and translation vector (T) of the stereo camera system.
    """
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


def load_camera_int_params(chessboard_size, square_size):
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
    return camera_int_params


def load_camera_ext_params(chessboard_size, square_size, camera_int_params):
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
    return camera_ext_params


def read_camera_parameters(chessboard_size, square_size):
    camera_int_params = load_camera_int_params(chessboard_size, square_size)
    camera_ext_params = load_camera_ext_params(
        chessboard_size, square_size, camera_int_params)

    print(f'camera_int_params:\n{camera_int_params}')
    print(f'camera_ext_params:\n{camera_ext_params}')

    return camera_int_params, camera_ext_params
