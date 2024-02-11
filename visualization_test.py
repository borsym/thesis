import open3d as o3d
import cv2
import threading
from pypylon import pylon
import time

class CameraManager:
    def __init__(self):
        self.camera_threads = []
        self.vis = o3d.visualization.Visualizer()

    def start_cameras(self, num_cameras):
        for i in range(num_cameras):
            cv2.namedWindow(f'Camera {i}', cv2.WINDOW_NORMAL)
            camera_thread = CameraThread(i)
            camera_thread.start()
            self.camera_threads.append(camera_thread)

    def stop_cameras(self):
        for camera_thread in self.camera_threads:
            camera_thread.stop()
            camera_thread.join()

    # def update_cv2_windows(self):
    #     global latest_frames
    #     for index, frame in latest_frames.items():
    #         cv2.imshow(f'Camera {index}', frame)
    #         cv2.waitKey(1)

    def update_cv2_windows(self):
        global latest_frames
        for index, frame in latest_frames.items():
            window_name = f'Camera {index}'
            if frame is not None:
                # Get current window size
                window_size = cv2.getWindowImageRect(window_name)
                if window_size[2] > 0 and window_size[3] > 0:
                    # Resize frame to fit the window
                    resized_frame = cv2.resize(frame, (window_size[2], window_size[3]))
                    cv2.imshow(window_name, resized_frame)
                else:
                    cv2.imshow(window_name, frame)
                cv2.waitKey(1)

    def run_visualization(self):
        self.vis.create_window()
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.vis.add_geometry(mesh)

        while self.vis.poll_events():
            self.update_cv2_windows()  # Update OpenCV windows here
            self.vis.update_renderer()
            time.sleep(0.01)  # Add a small delay to reduce CPU usage

        self.vis.destroy_window()

    def run(self):
        self.start_cameras(2)  # Assuming two cameras
        self.run_visualization()  # This will also handle OpenCV updates
        self.stop_cameras()
        cv2.destroyAllWindows()

latest_frames = {}  # Global dictionary to store the latest frames
class CameraThread(threading.Thread):
    def __init__(self, camera_index):
        threading.Thread.__init__(self)
        self.camera_index = camera_index
        self.running = True

    def run(self):
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[self.camera_index]))
        # camera.Attach(pylon.tlFactory.CreateDevice(devices[self.cam_id]))
        camera.Open()

        camera.TriggerMode.SetValue("Off")
        # Assuming "Line1" is the trigger source, adjust as needed
        # self.camera.TriggerSource.SetValue("Line1")
        camera.PixelFormat.SetValue("RGB8")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        global latest_frames
        while self.running:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                latest_frames[self.camera_index] = grabResult.Array
            grabResult.Release()

        camera.Close()

    def stop(self):
        self.running = False
# class CameraThread(threading.Thread):
#     def __init__(self, camera_index, on_new_frame, cam_id):
#         threading.Thread.__init__(self)
#         self.camera_index = camera_index
#         self.on_new_frame = on_new_frame
#         self.running = True
#         self.cam_id = cam_id
#
#     def run(self):
#         tl_factory = pylon.TlFactory.GetInstance()
#         devices = tl_factory.EnumerateDevices()
#         camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[self.cam_id]))
#         # camera.Attach(pylon.tlFactory.CreateDevice(devices[self.cam_id]))
#         camera.Open()
#
#         camera.TriggerMode.SetValue("Off")
#         # Assuming "Line1" is the trigger source, adjust as needed
#         # self.camera.TriggerSource.SetValue("Line1")
#         camera.PixelFormat.SetValue("RGB8")
#         camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
#
#         print(f'camera {self.cam_id} run called')
#         while self.running and camera.IsGrabbing():
#             grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#             if grabResult.GrabSucceeded():
#                 # Notify about the new frame
#                 self.on_new_frame(grabResult.Array, self.camera_index)
#             grabResult.Release()
#
#         camera.Close()

    def stop(self):
        self.running = False


# class CameraManager:
#     def __init__(self):
#         self.camera_threads = []
#         self.vis = o3d.visualization.Visualizer()
#
#     def on_new_frame(self, frame, camera_index):
#         # Process and display the frame as needed
#         # For example, update the Open3D visualization with the new frame
#         pass
#
#     def start_cameras(self, num_cameras):
#         for i in range(num_cameras):
#             camera_thread = CameraThread(i, self.on_new_frame, i)
#             camera_thread.start()
#             self.camera_threads.append(camera_thread)
#
#     def stop_cameras(self):
#         for camera_thread in self.camera_threads:
#             camera_thread.stop()
#             camera_thread.join()
#
#     def run_visualization(self):
#         # Initialize and run Open3D visualization in the main thread
#         self.vis.create_window()
#         # Add and update Open3D geometry here
#         # ...
#         self.vis.run()
#         self.vis.destroy_window()
#
#     def run(self):
#         # Start camera threads
#         self.start_cameras(2)  # Assuming two cameras
#         # Run Open3D visualization
#         self.run_visualization()
#         # Stop camera threads
#         self.stop_cameras()


def main():
    camera_manager = CameraManager()
    camera_manager.run()

if __name__ == "__main__":
    main()