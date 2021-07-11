#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Datasetter: It is like a casette recorder for data training
# Tool for calibration of the position of the person and the theremin antena
# The outputs are two rois one for antena and one for body (fixed in this version
# , can be improved by using a body pose estimation)

import cv2
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import argparse
import queue as Queue
import pickle
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
    'BODY_ROI_FILENAME': 'roi_position.pkl'
} 

class DatasetteROICapture:
    def __init__(
            self, 
            queue,
            preview_width=640,
            preview_height=400,
            fps=30, 
            depth_stream_name='depth', 
            cam_resolution='400'
        ):
        # Message processing queue
        self.queue = queue
        # Camera options = 400, 720, 800 for depth
        cam_res = {
            '400': (
                    dai.MonoCameraProperties.SensorResolution.THE_400_P,
                    640,
                    400
                ),
            '720': (
                    dai.MonoCameraProperties.SensorResolution.THE_720_P,
                    1280,
                    720
                ),
            '800': (
                    dai.MonoCameraProperties.SensorResolution.THE_800_P,
                    1280,
                    800
                )
        }
        self.depth_mono_resolution_left = cam_res[cam_resolution][0]
        self.depth_mono_resolution_right = cam_res[cam_resolution][0]
        self.depth_res_w = cam_res[cam_resolution][1]
        self.depth_res_h = cam_res[cam_resolution][2]
        self.depth_fps = fps
        self.depth_stream_name = depth_stream_name
        # ROI for depth
        self.depth_topx = 1
        self.depth_bottomx = 0
        self.depth_topy = 1
        self.depth_bottomy = 0
        self.depth_roi = None

        self.depth_frame = None
        self.show_depth = True
        self.depth_data = None

        # Preview size
        self.preview_width = preview_width
        self.preview_height = preview_height
        # Pipeline
        self.pipeline = None
        # Initial roi
        self.body_roi_data = None
        self.body_roi = {
            'topl': [0.4, 0.4],
            'bottomr': [0.6, 0.6],
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.step_size = 0.05

    # Check current pipeline
    def check_pipeline(self):
        if self.pipeline is not None:
            node_map = self.pipeline.getNodeMap()
            for idx, node in node_map.items():
                print(f"{idx}: {node.getName()}")

    # Pipeline for depth
    def create_pipeline_depth(self):
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Creating Pipeline for Depth ...")
        # Pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Setting up Depth...")    
        # Mono Left Camera
        mono_l = pipeline.createMonoCamera()
        # Mono Right Camera
        mono_r = pipeline.createMonoCamera()
        # Mono Camera Settings
        mono_l.setResolution(self.depth_mono_resolution_left)
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_r.setResolution(self.depth_mono_resolution_right)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        # Depth and Output
        stereo = pipeline.createStereoDepth()
        xout_depth = pipeline.createXLinkOut()
        # Spatial calculator
        spatial_calculator = pipeline.createSpatialLocationCalculator()
        xout_spatial = pipeline.createXLinkOut()
        xin_spatial_config = pipeline.createXLinkIn()

        # Stream Names
        xout_depth.setStreamName(self.depth_stream_name)
        xout_spatial.setStreamName("spatial")
        xin_spatial_config.setStreamName("config")
        # Stereo Depth parameters 
        output_depth = True
        output_rectified = False
        lr_check = False
        subpixel = False
        extended = False
        stereo.setOutputDepth(output_depth)
        stereo.setOutputRectified(output_rectified)
        stereo.setConfidenceThreshold(255)
        stereo.setLeftRightCheck(lr_check)
        stereo.setSubpixel(subpixel)
        stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout

        # Median Filter
        median = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5

        # incomptatible options
        if lr_check or extended or subpixel:
            median = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

        stereo.setMedianFilter(median) 
        # Mono L / R -> Stereo L / R
        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)
        # Stereo Depth -> Out
        stereo.depth.link(xout_depth.input)
        # Configuration:
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(
            dai.Point2f(self.body_roi['topl'][0],self.body_roi['topl'][1]),
            dai.Point2f(self.body_roi['bottomr'][0],self.body_roi['bottomr'][1])
        )
        # Config calculator with initial ROI
        spatial_calculator.setWaitForConfigInput(False)
        spatial_calculator.initialConfig.addROI(config)
        # Depth -> Calculator
        spatial_calculator.passthroughDepth.link(xout_depth.input)
        stereo.depth.link(spatial_calculator.inputDepth)
        # Calculator -> Spatial data out, config -> calculator
        spatial_calculator.out.link(xout_spatial.input)
        xin_spatial_config.out.link(spatial_calculator.inputConfig)
        
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

    # Show spatial data
    def show_spatial_data(self, frame, spatial_data, color=(240,180,100)):
        for data in spatial_data:
            roi = data.config.roi
            roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            #depth_min = data.depthMin
            #depth_max = data.depthMax

            font_type = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(frame, f"X: {int(data.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), font_type, 0.5, color)
            cv2.putText(frame, f"Y: {int(data.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), font_type, 0.5, color)
            cv2.putText(frame, f"Z: {int(data.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), font_type, 0.5, color)

    # Show display with depth
    def show_depth_map(self, spatial_data, instr):
        if self.depth_frame is not None:
            dframe = self.depth_frame.copy()
            depth_frame_color = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_color = cv2.equalizeHist(depth_frame_color)
            depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_OCEAN)

            self.show_spatial_data(depth_frame_color, spatial_data)
            self.show_instructions(instr, depth_frame_color, orig=(50,40), color=(0,0,255), size=0.6)
            cv2.imshow(self.depth_stream_name, depth_frame_color)


    # Capture roi for body position (torso)
    def capture_body(self):
        self.pipeline = self.create_pipeline_depth()
        self.check_pipeline()
        with dai.Device(self.pipeline) as device:
            ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            print(f"[{ts}]: Pipeline Started...")

            # 1. Queue for depth
            q_d = device.getOutputQueue(name=self.depth_stream_name, maxSize=4, blocking=False)
            # 2. Queue for spatial calculator
            q_s = device.getOutputQueue(name="spatial", maxSize=4, blocking=False)
            # 3. Queue for spatial config
            config = dai.SpatialLocationCalculatorConfigData()
            q_config = device.getInputQueue("config")
            
            # current_fps
            new_config = False
            self.current_fps = FPS(mean_nb_frames=20)
            frame_number = 0
            
            while True:
                # print(device.getChipTemperature().average)
                frame_number += 1
                self.current_fps.update()
                # Get frame
                in_depth = q_d.get()
                self.depth_frame = in_depth.getFrame()

                sp_data = q_s.get().getSpatialLocations()
                for data in sp_data:
                    self.body_roi['x'] = data.spatialCoordinates.x
                    self.body_roi['y'] = data.spatialCoordinates.y
                    self.body_roi['z'] = data.spatialCoordinates.z
                
                # Show depth
                instr = "q: quit | wasd: move roi | r: save"
                self.show_depth_map(sp_data, instr)
                
                # Commands
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # quit
                    break

                if key == ord('w'):
                    if self.body_roi['topl'][1] - self.step_size >= 0:
                        self.body_roi['topl'][1] -= self.step_size
                        self.body_roi['bottomr'][1] -= self.step_size
                        new_config = True
                elif key == ord('a'):
                    if self.body_roi['topl'][0] - self.step_size >= 0:
                        self.body_roi['topl'][0] -= self.step_size
                        self.body_roi['bottomr'][0] -= self.step_size
                        new_config = True                    
                elif key == ord('s'):
                    if self.body_roi['bottomr'][1] + self.step_size <= 1:
                        self.body_roi['topl'][1] += self.step_size
                        self.body_roi['bottomr'][1] += self.step_size
                        new_config = True
                elif key == ord('d'):
                    if self.body_roi['bottomr'][0] + self.step_size <= 1:
                        self.body_roi['topl'][0] += self.step_size
                        self.body_roi['bottomr'][0] += self.step_size
                        new_config = True

                if key == ord('r'):
                    # Save capture
                    self.body_roi_data = self.body_roi

                if new_config:
                    config.roi = dai.Rect(
                        dai.Point2f(self.body_roi['topl'][0],self.body_roi['topl'][1]),
                        dai.Point2f(self.body_roi['bottomr'][0],self.body_roi['bottomr'][1])
                    )
                    cfg = dai.SpatialLocationCalculatorConfig()
                    cfg.addROI(config)
                    q_config.send(cfg)
                    new_config = False

                elif key == 32:
                    # Pause on space bar
                    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdblob', default=PARAMS['PALM_DETECTION_MODEL_PATH'], type=str, 
                        help="Palm detection blob path")
    parser.add_argument('--pdth', default=PARAMS['PALM_THRESHOLD'], type=float, help="Palm Detector Threshold")
    parser.add_argument('--pdnms', default=PARAMS['PALM_NMS_THRESHOLD'], type=float, help="Palm Detector NMS Threshold")
    parser.add_argument('--lmblob', default=PARAMS['LM_DETECTION_MODEL_PATH'], type=str, 
                        help="Hand Landmark detection blob path")
    parser.add_argument('--lmth', default=PARAMS['LM_THRESHOLD'], type=float, help="Landmark Detector Threshold")
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Capture FPS")
    parser.add_argument('--prwidth', default=PARAMS['PREVIEW_WIDTH'], type=int, help="Preview Width")
    parser.add_argument('--prheight', default=PARAMS['PREVIEW_HEIGHT'], type=int, help="Preview Height")
    parser.add_argument('--pixbuff', default=PARAMS['HAND_BUFFER_PIXELS'], type=int, help="Extra buffer for hand gestures in pixels")
    parser.add_argument('--hsize', default=PARAMS['HAND_SIZE'], type=int, help="Frame size for showing hand")
    parser.add_argument('--mode', default=0, type=int, help="Capture Mode: 0 -> Capture Body ROI, 1 -> Antenna ROI")
    args = parser.parse_args()

    print(team.banner)
    # Message Queue
    messages = Queue.Queue()

    # Datasette recorder
    datasette = DatasetteROICapture(
        queue=messages,
        preview_width=args.prwidth,
        preview_height=args.prheight,
        fps=args.fps
    )

    if args.mode == 0:
        # Capture Body roi
        datasette.capture_body() 
        # Save latest ROI into file
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        if datasette.body_roi_data is not None:
            # Save to file
            roi = datasette.body_roi_data
            filename = PARAMS['DATASET_PATH'] + "/" + PARAMS['BODY_ROI_FILENAME']
            pickle.dump(roi, open(filename, "wb"))
            print(f"[{ts}]: ROI saved to file: {filename}")
        else:
            print(f"[{ts}]: No ROI defined")
    else:
        pass