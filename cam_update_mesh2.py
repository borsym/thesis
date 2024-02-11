from pypylon import pylon
import threading
from collections import defaultdict
import time

class CameraCapture:
    def __init__(self, camera_info, frame_dict, lock, name):
        self.camera_info = camera_info
        self.frame_dict = frame_dict
        self.lock = lock
        self.running = True
        self.cam_name = name

    def start(self):
        threading.Thread(target=self.capture_loop, args=()).start()

    def configure_for_trigger(self, camera):
        camera.TriggerSelector.SetValue("FrameStart")
        camera.TriggerMode.SetValue("On")
        camera.TriggerSource.SetValue("Line1")
        camera.PixelFormat.SetValue("RGB8")
        camera.StreamGrabber.MaxTransferSize = 4194304

        # camera.TriggerMode.SetValue("On")
        # camera.TriggerSource.SetValue("Line1")  # Adjust this based on your trigger line
        # Other configurations based on your trigger setup

    def wait_for_signal(self, camera):
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)
        while True:
            try:
                camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                break
            except pylon.TimeoutException:
                print(f"Camera trigger is not ready at {self.cam_name}")
        print(f"Camera trigger is ready at {self.cam_name}")


    def capture_loop(self):
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(self.camera_info))
        camera.Open()
        self.configure_for_trigger(camera)

        self.wait_for_signal(camera)
        # camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        converter = pylon.ImageFormatConverter()

        # Converting to OpenCV BGR format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while self.running and camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                img = image.GetArray()
                timestamp = time.time()

                with self.lock:
                    self.frame_dict[self.camera_info.GetSerialNumber()] = (img, timestamp)

            grabResult.Release()

        camera.Close()

    def stop(self):
        self.running = False

# Shared resource and lock
frame_dict = defaultdict(lambda: (None, 0))
frame_lock = threading.Lock()

# Getting connected cameras
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if len(devices) == 0:
    raise RuntimeError("No camera present.")
else:
    print(f'{len(devices)} camera devices found.')

# Initialize camera captures
camera1_info = devices[0]
camera2_info = devices[1]

camera1 = CameraCapture(camera1_info, frame_dict, frame_lock, "CAM_1")
camera2 = CameraCapture(camera2_info, frame_dict, frame_lock, "CAM_2")

# Start capturing
camera1.start()
camera2.start()

last_processed_time = defaultdict(lambda: 0)

i=0
try:
    while True:
        # Main loop
        with frame_lock:
            frame1, time1 = frame_dict[camera1_info.GetSerialNumber()]
            frame2, time2 = frame_dict[camera2_info.GetSerialNumber()]

        if frame1 is not None:
            print('alma')
        if frame2 is not None:
            print('korte')
        # if frame2 is not None:
        #     print(f'last: {last_processed_time[camera1_info.GetSerialNumber()]}')
        #     print(f'current: {time1}')
        #     if (frame2 is not None and time2 > last_processed_time[camera2_info.GetSerialNumber()]):
        #         print('frame2 test ok')
        #     if (frame1 is not None and time1 > last_processed_time[camera1_info.GetSerialNumber()]):
        #         print('frame1 test ok')
        # Check if new frames are available to process
        if frame1 is not None and \
           (frame2 is not None and time2 > last_processed_time[camera2_info.GetSerialNumber()]):
            last_processed_time[camera1_info.GetSerialNumber()] = time1
            last_processed_time[camera2_info.GetSerialNumber()] = time2
            print(f'{i}. image found in both cameras.')
            i += 1
            # Process the frames with your neural network model
            pass

        # Add your processing logic here
        # ...

except KeyboardInterrupt:
    camera1.stop()
    camera2.stop()