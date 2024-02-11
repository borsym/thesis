from pypylon import pylon
import time

def test_single_camera(camera_index):
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if len(devices) <= camera_index:
        print(f"No camera found at index {camera_index}")
        return

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[camera_index]))
    camera.Open()
    print(f"Testing camera at index {camera_index} with serial number: {devices[camera_index].GetSerialNumber()}")

    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        print(f"Camera at index {camera_index} grabbed an image successfully.")
    else:
        print(f"Camera at index {camera_index} failed to grab an image.")

    grabResult.Release()
    camera.Close()

# Test the second camera
test_single_camera(0)
test_single_camera(1)
