import glob
import os
# ffmpeg -framerate 30 -i "image%04d.bmp" -c:v libx264 -pix_fmt yuv420p out.mp4
# get all the images from this folder: D:\thesis\realtime_update\recordings
images = glob.glob('D:/thesis/realtime_update/recordings/Scenario1/Cam2/*.bmp')

# sort the images by the last 4 digits of the filename

images.sort(key=lambda x: int(x[-8:-4]))

# rename the images
for i, image in enumerate(images):
    new_name = f'D:/thesis/realtime_update/recordings/Scenario1/Cam2/image{i:04}.bmp'
    print(f'{image} -> {new_name}')
    os.rename(image, new_name)
