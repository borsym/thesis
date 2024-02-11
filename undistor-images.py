import cv2
import os
import json
import numpy as np
import glob


def read_instrict():
    camera_int_params = None
    if os.path.exists(f'intrinsic/intrinsics.json'):
        print(f'loading intrinsics from config...')
        with open('intrinsic/intrinsics.json', 'r') as json_file:
            camera_int_params = json.load(json_file)
    return camera_int_params


def read_exstrict():
    camera_ext_params = None
    if os.path.exists(f'extrinsic/extrinsics.json'):
        print(f'loading extrinsic from config...')
        with open('extrinsic/extrinsics.json', 'r') as json_file:
            camera_ext_params = json.load(json_file)
    return camera_ext_params


def undistort_image(cam1_images, cam2_images, camera_int_params, camera_ext_params):
    mtx0_int = np.asarray(camera_int_params['cam0_K'])
    mtx1_int = np.asarray(camera_int_params['cam1_K'])
    dist0_int = np.asarray(camera_int_params['cam0_dist'])
    dist1_int = np.asarray(camera_int_params['cam1_dist'])

    R = np.asarray(camera_ext_params['R'])
    T = np.asarray(camera_ext_params['T'])

    save_undistored(cam1_images, mtx0_int, dist0_int,
                    "traning/undistored/cam1/undistorted_cam1")
    save_undistored(cam2_images, mtx1_int, dist1_int,
                    "traning/undistored/cam2/undistorted_cam2")


def save_undistored(cam_images, mtx_int, dist_int, path):
    for i in range(len(cam_images)):
        image = cv2.imread(cam_images[i])
        img_undistorted = cv2.undistort(
            image, mtx_int, dist_int, None, mtx_int)
        cv2.imwrite(
            f'{path}_{i}.png', img_undistorted)


def main():
    camera_int_params = read_instrict()
    camera_ext_params = read_exstrict()

    print(f'camera_int_params:\n{camera_int_params}')
    print(f'camera_ext_params:\n{camera_ext_params}')

    cam1_images = glob.glob('traning/cam1/*')
    cam2_images = glob.glob('traning/cam2/*')

    undistort_image(cam1_images, cam2_images,
                    camera_int_params, camera_ext_params)


if __name__ == "__main__":
    main()
