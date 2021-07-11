#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Datasetter: It is like a casette recorder for data training
# Tool for capturing images of the theremin

import cv2
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import argparse
import queue as Queue
import team

# Parameters
PARAMS = {
    'CAPTURE_DEVICE': 0,
    'KEY_QUIT': 'q',
    'HUD_COLOR': (153,219,112),
    'VIDEO_RESOLUTION': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    'PALM_DETECTION_MODEL_PATH': "models/palm_detection.blob",
    'PALM_THRESHOLD': 0.5,
    'PALM_NMS_THRESHOLD': 0.3,
    'PALM_DETECTION_INPUT_LENGTH': 128,
    'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
    'LM_THRESHOLD': 0.5,
    'LM_INPUT_LENGTH': 224,
    'FPS': 2,
    'ROI_DP_LOWER_TH': 100,
    'ROI_DP_UPPER_TH': 10000,
    'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
    'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6),
    'PREVIEW_WIDTH': 640,
    'PREVIEW_HEIGHT': 400,
    'HAND_BUFFER_PIXELS': 20,
    'HAND_SIZE': 400,
    'DATASET_PATH': 'data/positions',
    'DEPTH_ROI_FILENAME': 'roi.pkl',
    'DEPTH_CAPTURE_FILENAME': 'depth.pkl',
    'DEPTH_RESOLUTION': '400',
    'BODY_ROI_FILENAME': 'roi_position.pkl',
    'IMG_THEREMIN_FOLDER': 'theremin'
} 

class DatasetteThereminCapture:
    def __init__(
            self, 
            queue,
            preview_width,
            preview_height
        ):
        # Message processing queue
        self.queue = queue    
        # Preview size
        self.preview_width = preview_width
        self.preview_height = preview_height
        # Saved Images
        self.images = None
        self.image_buffer = {}

    # Check current pipeline
    def check_pipeline(self):
        if self.pipeline is not None:
            node_map = self.pipeline.getNodeMap()
            for idx, node in node_map.items():
                print(f"{idx}: {node.getName()}")

    # Pipeline for depth
    def create_pipeline_depth(self):
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Creating Pipeline RGB Camera ...")
        
        # Pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        cam_rgb = pipeline.createColorCamera()
        still_encoder = pipeline.createVideoEncoder()

        control_in = pipeline.createXLinkIn()
        config = pipeline.createXLinkIn()
        still_out = pipeline.createXLinkOut()
        xout_rgb = pipeline.createXLinkOut()

        control_in.setStreamName('control')
        config.setStreamName('config')
        still_out.setStreamName('still')
        xout_rgb.setStreamName("rgb")

        # width, height
        cam_rgb.setPreviewSize(self.preview_width, self.preview_height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        still_encoder.setDefaultProfilePreset(cam_rgb.getStillSize(), 1, dai.VideoEncoderProperties.Profile.MJPEG)


        # cam -> out
        cam_rgb.preview.link(xout_rgb.input)
        cam_rgb.still.link(still_encoder.input)
        control_in.out.link(cam_rgb.inputControl)
        config.out.link(cam_rgb.inputConfig)
        still_encoder.bitstream.link(still_out.input)


        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S") 
        print(f"[{ts}]: Pipeline Created...")    
        return pipeline

    # Show instructions
    def show_instructions(self, instr, frame, orig, font=cv2.FONT_HERSHEY_SIMPLEX, size=1, color=(240,180,100), thickness=2):
        cv2.putText(
            frame, 
            f"{instr}", 
            orig, 
            font, 
            size, 
            color, 
            thickness
        )
    
    def show_img(self, instr, frame):
        dframe = frame.copy()
        self.show_instructions(instr, dframe, orig=(50,40), color=(0,0,255), size=0.6)
        cv2.imshow("camera", dframe)

    # Capture Image
    def capture_img(self):
        self.pipeline = self.create_pipeline_depth()
        self.check_pipeline()
        with dai.Device(self.pipeline) as device:
            ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            print(f"[{ts}]: Pipeline Started...")

            # 1. Queue for rgb camera
            q_rgb = device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
            # 2. Queue for trigger
            q_control = device.getInputQueue(name='control')
            # 3. Queue for still image
            q_still = device.getOutputQueue('still')

            # current_fps
            self.current_fps = FPS(mean_nb_frames=20)
            frame_number = 0
            
            while True:
                # print(device.getChipTemperature().average)
                frame_number += 1
                self.current_fps.update()
                # Get frame
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()

                # Show frame
                instr = f"q: quit | c: capture | s: save images ({len(self.image_buffer.keys())})"
                self.show_img(instr, frame)             

                # Commands
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # quit
                    break

                if key == ord('c'):
                    ctrl = dai.CameraControl()
                    ctrl.setCaptureStill(True)
                    q_control.send(ctrl)

                    still_frames = q_still.tryGetAll()
                    for idx, still_frame in enumerate(still_frames):
                        # Decode JPEG
                        k = f"{frame_number}_{idx}"
                        print(k)
                        frame = cv2.imdecode(still_frame.getData(), cv2.IMREAD_UNCHANGED)
                        self.image_buffer[k] = frame

                    
                if key == ord('s'):
                    self.images = self.image_buffer
                    self.image_buffer = {}                
                
                elif key == 32:
                    # Pause on space bar
                    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prwidth', default=PARAMS['PREVIEW_WIDTH'], type=int, help="Preview Width")
    parser.add_argument('--prheight', default=PARAMS['PREVIEW_HEIGHT'], type=int, help="Preview Height")
    parser.add_argument('--prefix', default='capture', type=str, help="Name prefix for captured images")
    args = parser.parse_args()

    print(team.banner)
    # Message Queue
    messages = Queue.Queue()

    # Datasette recorder
    datasette = DatasetteThereminCapture(
        queue=messages,
        preview_width=args.prwidth,
        preview_height=args.prheight
    )
    
    # Capture Theremin Images
    datasette.capture_img() 
    # Save collection of images
    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    if datasette.images is not None:
        # Save to file
        path = PARAMS['DATASET_PATH'] + "/" + PARAMS['IMG_THEREMIN_FOLDER'] + "/"
        for name, frame in datasette.images.items():
            filename = f"{path}{args.prefix}_{name}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[{ts}]: Saved image to file: {filename}")
