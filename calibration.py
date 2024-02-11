import cv2
import numpy as np
import glob


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
    print('Gathering Intrinsic Camera Parameters...')
    img = None
    # Replace '1' with the number of image pairs you have
    for i, img in enumerate([image0, image1]):
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        # print(i, ret)
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

    images_left = glob.glob('extrinsic/image_0_*.png')  # Update path
    images_right = glob.glob('extrinsic/image_1_*.png')  # Update path
    print(len(images_left), len(images_right))

    gray_left = None
    jj = 0
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
            gray_left, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)

        print(ret_left, ret_right)
        # If found, add object points and image points
        if ret_left and ret_right:
            print("itt", jj)
            objpoints.append(objp)
            corners2_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners2_left)
            imgpoints_right.append(corners2_right)

            # cv2.drawChessboardCorners(
            #     img_left, chessboard_size, corners2_left, ret_left)
            # cv2.imwrite(f'img-{jj}', img_left)
            # cv2.drawChessboardCorners(
            #     img_right, chessboard_size, corners2_right, ret_right)
            # cv2.imwrite(f'img-{jj}', img_right)
            print("v", jj)
            jj += 1
    # Stereo calibration
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx0_int, dist0_int, mtx1_int, dist1_int, gray_left.shape[::-1])

    print("Rotation matrix:\n", R)
    print("Translation vector:\n", T)
    return R, T


def main():
    chessboard_size = (9, 6)
    square_size = 70
    door_cam_internal_img = cv2.imread(
        f'intrinsic/door-test.jpg')  # Load the image
    window_cam_internal_img = cv2.imread(
        f'intrinsic/win5-test.jpg')  # Load the image
    camera_int_params = calibrate_intrinsics(
        door_cam_internal_img, window_cam_internal_img, chessboard_size, square_size)
    print("internal params")
    print(camera_int_params)
    print("external params")
    camera_ext_params = calibrate_extrinsics(
        camera_int_params, chessboard_size, square_size)
    print(camera_ext_params)


if __name__ == "__main__":
    main()
