from pypylon import pylon
import threading
from collections import defaultdict
import time

class CameraArrayCapture:
    def __init__(self, num_cameras, frame_dict, lock):
        self.num_cameras = num_cameras
        self.frame_dict = frame_dict
        self.lock = lock
        self.running = True
        self.cameras = pylon.InstantCameraArray(num_cameras)

    def start(self):
        threading.Thread(target=self.capture_loop, args=()).start()

    def configure_cameras(self, use_trigger=True):
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if len(devices) < self.num_cameras:
            raise RuntimeError("Not enough cameras present.")

        for i in range(self.num_cameras):
            # self.cameras[i]
            # self.cameras[i].Open()
            # Configure trigger mode here if necessary
            # self.cameras[i].TriggerMode.SetValue("On")
            # self.cameras[i].TriggerSource.SetValue("Line1")
            camera = self.cameras[i]
            camera.Attach(tl_factory.CreateDevice(devices[i]))
            camera.Open()
            # camera.TriggerSelector.SetValue("FrameStart")
            camera.TriggerMode.SetValue("Off")
            # camera.TriggerSource.SetValue("Line1")
            # camera.PixelFormat.SetValue("RGB8")
            # camera.StreamGrabber.MaxTransferSize = 4194304

    def wait_for_signal(self):
        if not self.cameras:
            raise RuntimeError("Cameras not initialized.")
        while True:
            try:
                for camera in self.cameras:
                    camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                break
            except pylon.TimeoutException:
                print("Camera trigger is not ready")
        print("Camera trigger is ready")

    def capture_loop(self):
        self.configure_cameras(use_trigger=False)  # Set to False for non-triggered test

        for i in range(self.num_cameras):
            if not self.cameras[i].IsOpen():
                print(f"Camera {i} failed to open.")
                continue
            print(f"Camera {i} opened successfully.")

        # ... (rest of the capture loop)

        while self.running and self.cameras.IsGrabbing():
            for i in range(self.num_cameras):
                try:
                    grabResult = self.cameras[i].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                    if grabResult.GrabSucceeded():
                        # Process image
                        print(f"Camera {i} grabbed an image.")
                    else:
                        print(f"Camera {i} failed to grab an image.")
                    grabResult.Release()

                except Exception as e:
                    print(f"An error occurred with camera {i}: {e}")

        for i in range(self.num_cameras):
            self.cameras[i].Close()

    def stop(self):
        self.running = False

# Shared resource and lock
frame_dict = defaultdict(lambda: (None, 0))
frame_lock = threading.Lock()

# Initialize and start camera array capture
camera_capture = CameraArrayCapture(2, frame_dict, frame_lock)  # Adjust the number of cameras as needed
camera_capture.start()

last_processed_time = defaultdict(lambda: 0)

i=0
try:
    while True:
        # Main loop
        with frame_lock:
            frame1, time1 = frame_dict[0]
            frame2, time2 = frame_dict[1]
        if frame1 is not None:
            print('alma')
        if frame2 is not None:
            print('korte')
        # Check if new frames are available to process
        if (frame1 is not None and time1 > last_processed_time[0]) and \
           (frame2 is not None and time2 > last_processed_time[1]):
            last_processed_time[0] = time1
            last_processed_time[1] = time2

            # Process the frames with your neural network model
            pass

        # Add your processing logic here
        # ...

except KeyboardInterrupt:
    camera_capture.stop()

