from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time 
import argparse

# Connect Device
with dai.Device() as device:
    calibData = device.readCalibration()
    M_rgb, width, height = np.array(calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB))
    print("RGB Camera Default intrinsics...")
    print(M_rgb)
    print(width)
    print(height)

    M_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, 1280, 720))
    print("LEFT Camera resized intrinsics...")
    print(M_left)

    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    M_right = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), 1280, 720))

    H_left = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_left))
    print("LEFT Camera stereo rectification matrix...")
    print(H_left)

    lr_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
    print("Transformation matrix of where left Camera is W.R.T right Camera's optical center")
    print(lr_extrinsics)

    M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
    print("RIGHT Camera resized intrinsics...")
    print(M_right)

"""
C:\ai\opencv_competition_2021\clara\src\test_calibration.py:11: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  M_rgb, width, height = np.array(calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB))
RGB Camera Default intrinsics...
[[1483.971435546875, 0.0, 956.4616088867188], [0.0, 1482.0938720703125, 544.7604370117188], [0.0, 0.0, 1.0]]
1920
1080
LEFT Camera resized intrinsics...
[[853.88867188   0.         638.50714111]
 [  0.         854.19293213 364.89532471]
 [  0.           0.           1.        ]]
LEFT Camera stereo rectification matrix...
[[ 9.97610208e-01 -6.60148398e-03  4.52844732e+00]
 [ 3.57723374e-03  9.99799269e-01 -7.34644715e+00]
 [-4.14536098e-06 -2.41891507e-06  1.00352109e+00]]
Transformation matrix of where left Camera is W.R.T right Camera's optical center
[[ 9.99918282e-01 -2.79526622e-03  1.24759870e-02 -7.46226501e+00]
 [ 2.74379505e-03  9.99987662e-01  4.14082967e-03  1.72002781e-02]
 [-1.24874078e-02 -4.10625990e-03  9.99913573e-01  6.66968822e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

RIGHT Camera resized intrinsics...
[[854.11590576   0.         636.09185791]
 [  0.         854.77392578 357.98129272]
 [  0.           0.           1.        ]] 
jchong@DESKTOP-M28RMHT   opencv2021bk  C:\ai\opencv_competition_2021\clara\src   main ↑1 +7 ~2 -1 !      [21:44]
❯
"""

