from ultralytics import YOLO
import numpy as np
import cv2
import glob
import json

# Load the annotations from the JSON file
with open("D:/thesis/realtime_update/robot-keypoints-connection.json", 'r') as file:
    annotations_data = json.load(file)
skeleton = annotations_data['skeleton']

# Video parameters
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# output_video_path = 'keypoints_video.mp4'
output2_video_path = 'only_keypoints.mp4'

fps = 30
width = 1920
height = 1200
# video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
video_writer2 = cv2.VideoWriter(
    output2_video_path, fourcc, fps, (width, height))

model = YOLO('D:/thesis/realtime_update/convert_images/best.pt')

images = glob.glob('D:/thesis/realtime_update/recordings/Scenario1/Cam1/*.bmp')
images.sort()  # Make sure the images are sorted if necessary


for image_path in images:
    # Load the image
    original_image = cv2.imread(image_path)
    background_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Process the image through the model
    result = model(original_image)

    # Assuming result.keypoints.xy gives a list of keypoints for the current frame
    points = result[0].keypoints.xy[0]

    if len(points) > 0:
        # Draw keypoints
        for kp in points:
            x, y = kp
            # cv2.circle(original_image, (int(x), int(y)),
            #            radius=5, color=(255, 0, 0), thickness=-1)
            cv2.circle(background_image, (int(x), int(y)),
                       radius=5, color=(255, 0, 0), thickness=-1)

        # Draw skeleton links if available
        for link in skeleton:
            p1, p2 = int(link[0]), int(link[1])
            pt1, pt2 = points[p1 - 1], points[p2 - 1]
            # cv2.line(original_image, (int(pt1[0]), int(pt1[1])), (int(
            #     pt2[0]), int(pt2[1])), (255, 0, 255), 1)
            cv2.line(background_image, (int(pt1[0]), int(pt1[1])), (int(
                pt2[0]), int(pt2[1])), (255, 0, 255), 1)

    # Write the frame to the video
    # video_writer.write(original_image)
    video_writer2.write(background_image)

# Release the video writer
# video_writer.release()
video_writer2.release()
