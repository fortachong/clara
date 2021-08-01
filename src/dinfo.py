import depthai as dai
import numpy as np

if __name__ == "__main__":
    for device in dai.Device.getAllAvailableDevices():
        print("-"*80)
        print(f"{device.getMxId()} {device.state}")
        # Intrinsics:
        with dai.Device() as device:
            calibData = device.readCalibration()
            m_rgb, width, height = np.array(calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB), dtype="object")
            print("RGB Camera Default intrinsics...")
            print(f"RGB Intrinsics:")
            print(f"({width}, {height})")
            print(m_rgb)
            print()
            # Left and Right 400
            m_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 640, 480), dtype='object')
            print("LEFT Camera (640, 480) intrinsics...")
            print(m_left)
            m_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 640, 480), dtype='object')
            print("RIGTH Camera (640, 480) intrinsics...")
            print(m_right)
            print()
            # Left and Right 720
            m_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720), dtype='object')
            print("LEFT Camera (1280, 720) intrinsics...")
            print(m_left)
            m_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720), dtype='object')
            print("RIGTH Camera (1280, 720) intrinsics...")
            print(m_right)
            print()
            # Left and Right 800
            m_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 800), dtype='object')
            print("LEFT Camera (1280, 800) intrinsics...")
            print(m_left)
            m_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 800), dtype='object')
            print("RIGTH Camera (1280, 800) intrinsics...")
            print(m_right)
            print()