import cv2
import glob

def main():


    out0 = cv2.VideoWriter(f'/media/gables/Data/Camera Calibration/DFKI/Camera_Video/camera 0.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (2840, 2840))
    out1 = cv2.VideoWriter(f'/media/gables/Data/Camera Calibration/DFKI/Camera_Video/camera 1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (2840, 2840))

    for i in range(2,351):
        if i % 2 == 0:
            cam0_img = cv2.imread(f'/media/gables/Data/Camera Calibration/DFKI/Camera_Video/camera 0/image_0_{i}.png')
            cam1_img = cv2.imread(f'/media/gables/Data/Camera Calibration/DFKI/Camera_Video/camera 1/image_1_{i}.png')
            out0.write(cam0_img)
            out1.write(cam1_img)
            print(i)


if __name__ == "__main__":
    main()