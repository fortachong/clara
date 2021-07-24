#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Datasetter: It is like a casette recorder for data training
# A tool for capturing left hand gestures according to the method for
# Theremine playing.
# Here we test an idea about capturing depth only inside the region 
# Two modes:
# mode = 0: Defined hands regions
# Commands: q: quit | r: start rh ROI | l: start lh ROI | s: save ROI
# mode = 1: Capture hands frames for analysis and exploration
# Commands: q: quit | r: start capture | s: save

import os
import cv2
from cv2 import aruco
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS
import queue as Queue
import argparse
import pickle
import config
import team

# Landmarks:
LM_WRIST = 0
LM_THUMB_CMC = 1
LM_THUMB_MCP = 2
LM_THUMB_IP = 3
LM_THUMB_TIP = 4
LM_INDEX_FINGER_MCP = 5
LM_INDEX_FINGER_PIP = 6
LM_INDEX_FINGER_DIP = 7
LM_INDEX_FINGER_TIP = 8
LM_MIDDLE_FINGER_MCP = 9
LM_MIDDLE_FINGER_PIP = 10
LM_MIDDLE_FINGER_DIP = 11
LM_MIDDLE_FINGER_TIP = 12
LM_RING_FINGER_MCP = 13
LM_RING_FINGER_PIP = 14
LM_RING_FINGER_DIP = 15
LM_RING_FINGER_TIP = 16
LM_PINKY_MCP = 17
LM_PINKY_PIP = 18
LM_PINKY_DIP = 19
LM_PINKY_TIP = 20

# # Parameters
# PARAMS = {
#     'CAPTURE_DEVICE': 0,
#     'KEY_QUIT': 'q',
#     'HUD_COLOR': (153,219,112),
#     'LANDMARKS_COLOR': (0,255,0),
#     'LANDMARKS': [
#                     LM_WRIST, 
#                     LM_THUMB_TIP, 
#                     LM_INDEX_FINGER_TIP, 
#                     LM_MIDDLE_FINGER_TIP,
#                     LM_RING_FINGER_TIP,
#                     LM_PINKY_TIP
#                 ],

#     'VIDEO_RESOLUTION': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
#     'PALM_DETECTION_MODEL_PATH': "models/palm_detection.blob",
#     'PALM_THRESHOLD': 0.5,
#     'PALM_NMS_THRESHOLD': 0.3,
#     'PALM_DETECTION_INPUT_LENGTH': 128,
#     'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
#     'LM_THRESHOLD': 0.5,
#     'LM_INPUT_LENGTH': 224,
#     'FPS': 2,
#     'ROI_DP_LOWER_TH': 100,
#     'ROI_DP_UPPER_TH': 10000,
#     'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
#     'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6),
#     'PREVIEW_WIDTH': 640,
#     'PREVIEW_HEIGHT': 400,
#     'HAND_BUFFER_PIXELS': 20,
#     'HAND_SIZE': 400,
#     'DATASET_PATH': 'data/positions',
#     'DEPTH_ROI_FILENAME': 'roi.pkl',
#     'DEPTH_CAPTURE_FILENAME': 'depth_capture.pkl',
#     'DEPTH_RESOLUTION': '400',
#     'BODY_ROI_FILENAME': 'roi_position.pkl',
#     'ANTENNA_ROI_FILENAME': 'antenna_position.pkl',
#     'ROI_TOLERANCE_BODY_Z': 50,
#     'ROI_TOLERANCE_ANTENNA_Z': 5,
#     'ROI_TOLERANCE_ANTENNA_X': 15,
#     'ROI_HAND_ID': 2
# } 

PARAMS = config.PARAMS

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST)
    return resized.transpose(2,0,1)

class DatasetteDepthCapture:
    def __init__(
            self, 
            queue,
            preview_width=640,
            preview_height=400,
            hand_buffer_pixels=20,
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
        # ROI for depth right hand
        self.rh_depth_topx = 1
        self.rh_depth_bottomx = 0
        self.rh_depth_topy = 1
        self.rh_depth_bottomy = 0
        self.lh_depth_topx = 1
        self.lh_depth_bottomx = 0
        self.lh_depth_topy = 1
        self.lh_depth_bottomy = 0
        self.depth_roi = {}

        # For capturing depth frames
        self.depth_frame = None
        self.show_depth = True
        self.depth_data = None

        # For capturing hands
        self.hand_buffer_pixels = hand_buffer_pixels
        
        # Preview size
        self.preview_width = preview_width
        self.preview_height = preview_height
        # Pipeline
        self.pipeline = None
        # ROIs
        self.position_rois = None

    # Pipeline for hand landmarks
    def create_pipeline_lm(self):
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Creating Pipeline for Landmark Detection ...")
        # Pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        # Color Camera
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Right Mono Camera ...")       
        # Mono Camera
        mono_r = pipeline.createMonoCamera()
        # Mono Camera Settings
        mono_r.setResolution(self.depth_mono_resolution_right)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        mono_r.out.link(cam_out.input)         
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Pipeline Created ...")
        return pipeline

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
        # Stream Names
        xout_depth.setStreamName(self.depth_stream_name)
        # Stereo Depth parameters 
        output_depth = True
        output_rectified = False
        lr_check = False
        subpixel = False
        extended = False
        stereo.setOutputDepth(output_depth)
        stereo.setOutputRectified(output_rectified)
        stereo.setConfidenceThreshold(200)
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
        
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S") 
        print(f"[{ts}]: Pipeline Created...")    
        return pipeline

    # Show display with depth
    def show_depth_map(self, instr):
        if self.depth_frame is not None:
            dframe = self.depth_frame.copy()
            depth_frame_color = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_color = cv2.equalizeHist(depth_frame_color)
            depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_OCEAN)
            w, h = depth_frame_color.shape[1], depth_frame_color.shape[0]
            
            # Right hand roi (red)
            if 'right_hand' in self.depth_roi:
                topx = int(self.depth_roi['right_hand']['topx'] * w)
                bottomx = int(self.depth_roi['right_hand']['bottomx'] * w)
                topy = int(self.depth_roi['right_hand']['topy'] * h)
                bottomy = int(self.depth_roi['right_hand']['bottomy'] * h)
                cv2.rectangle(depth_frame_color, (topx, topy), (bottomx, bottomy), (0,0,255),2)
            # Left hand roi (green)
            if 'left_hand' in self.depth_roi:
                topx = int(self.depth_roi['left_hand']['topx'] * w)
                bottomx = int(self.depth_roi['left_hand']['bottomx'] * w)
                topy = int(self.depth_roi['left_hand']['topy'] * h)
                bottomy = int(self.depth_roi['left_hand']['bottomy'] * h)
                cv2.rectangle(depth_frame_color, (topx, topy), (bottomx, bottomy), (0,255,0),2)

            depth_frame_color = cv2.flip(depth_frame_color, 1)
            self.show_instructions(instr, depth_frame_color, orig=(50,40), color=(0,0,255), size=0.6)
            cv2.imshow(self.depth_stream_name, depth_frame_color)

    # Set ROI 
    def set_depth_ROI(self, roi):
        self.depth_roi = roi
    
    # Set Body and Antenna ROI
    def set_ROIs(self, rois):
        self.position_rois = rois

    # Check current pipeline
    def check_pipeline(self):
        if self.pipeline is not None:
            node_map = self.pipeline.getNodeMap()
            for idx, node in node_map.items():
                print(f"{idx}: {node.getName()}")

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

    # Show axis for 

    # Capture hand positions to establish a ROI for depth
    def capture_hand(self):
        self.pipeline = self.create_pipeline_lm()
        self.check_pipeline()

        with dai.Device(self.pipeline) as device:
            ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            print(f"[{ts}]: Pipeline Started...")

            # Queues
            # 1. Out: Video output
            q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            
            # Aruco markers
            # Parameters for aruco marker detection
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            parameters =  aruco.DetectorParameters_create()

            # current_fps
            self.current_fps = FPS(mean_nb_frames=20)
            frame_number = 0
            start_roi = False
            right_capture = True
            
            # Tmp points (only for drawing)
            rh_tmp_depth_topx = lh_tmp_depth_topx = 0
            rh_tmp_depth_bottomx = lh_tmp_depth_bottomx = 0
            rh_tmp_depth_topy = lh_tmp_depth_topy = 0
            rh_tmp_depth_bottomy = lh_tmp_depth_bottomy = 0
            
            while True:
                frame_number += 1
                self.current_fps.update()
                        
                # In video queue
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()
                            
                # Datasette is a cool name
                gray = video_frame.copy()
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                _ = aruco.drawDetectedMarkers(gray, corners, ids)

                if ids is not None:
                    for marker, id_ in zip(corners, ids):
                        if id_ == PARAMS['ROI_HAND_ID']:
                            if start_roi:
                                dtx_candidate_a, dty_candidate_a = marker[0,0,0], marker[0,0,1]
                                dbx_candidate_a, dby_candidate_a = marker[0,2,0], marker[0,2,1]
                                
                                dtx_candidate = dtx_candidate_a / self.preview_width
                                dbx_candidate = dbx_candidate_a / self.preview_width

                                dty_candidate = dty_candidate_a / self.preview_height
                                dby_candidate = dby_candidate_a / self.preview_height
                                
                                if right_capture is True:
                                    if self.rh_depth_topx > dtx_candidate:
                                        self.rh_depth_topx = dtx_candidate
                                        rh_tmp_depth_topx = dtx_candidate_a
                                    if self.rh_depth_bottomx < dbx_candidate:
                                        self.rh_depth_bottomx = dbx_candidate
                                        rh_tmp_depth_bottomx = dbx_candidate_a
                                    if self.rh_depth_topy > dty_candidate:
                                        self.rh_depth_topy = dty_candidate
                                        rh_tmp_depth_topy = dty_candidate_a
                                    if self.rh_depth_bottomy < dby_candidate:
                                        self.rh_depth_bottomy = dby_candidate
                                        rh_tmp_depth_bottomy = dby_candidate_a
                                   
                                    cv2.rectangle(
                                        gray,
                                        (int(rh_tmp_depth_topx), int(rh_tmp_depth_topy)),
                                        (int(rh_tmp_depth_bottomx), int(rh_tmp_depth_bottomy)),
                                        (0, 0, 255),
                                        2
                                    )
                                else:
                                    if self.lh_depth_topx > dtx_candidate:
                                        self.lh_depth_topx = dtx_candidate
                                        lh_tmp_depth_topx = dtx_candidate_a
                                    if self.lh_depth_bottomx < dbx_candidate:
                                        self.lh_depth_bottomx = dbx_candidate
                                        lh_tmp_depth_bottomx = dbx_candidate_a
                                    if self.lh_depth_topy > dty_candidate:
                                        self.lh_depth_topy = dty_candidate
                                        lh_tmp_depth_topy = dty_candidate_a
                                    if self.lh_depth_bottomy < dby_candidate:
                                        self.lh_depth_bottomy = dby_candidate
                                        lh_tmp_depth_bottomy = dby_candidate_a
                                   
                                    cv2.rectangle(
                                        gray,
                                        (int(lh_tmp_depth_topx), int(lh_tmp_depth_topy)),
                                        (int(lh_tmp_depth_bottomx), int(lh_tmp_depth_bottomy)),
                                        (255, 0, 0),
                                        2
                                    )                                                         

                self.current_fps.display(gray, orig=(50,50), color=(0,0,255), size=0.6)
                instr = "q: quit | r: start rh | l: start lh | s: save | t: reset"
                self.show_instructions(instr, gray, orig=(50,70), color=(0,0,255), size=0.6)
                
                # Show roi
                cv2.line(gray, 
                    (self.position_rois['antenna']['absolute']['topx'], 0),
                    (self.position_rois['antenna']['absolute']['topx'], self.preview_height),
                    (0,255,0),
                    2
                )
                cv2.imshow("Hand Position Configurations", gray)

                # Commands
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # Save depth limits
                    break

                if key == ord('r'):
                    # Start saving the roi of the right hand
                    start_roi = True
                    right_capture = True

                if key == ord('l'):
                    # Start saving the roi of the left hand
                    start_roi = True
                    right_capture = False                

                if key == ord('s'):
                    # Save roi (relative coords)
                    if right_capture:
                        self.depth_roi['right_hand'] = {
                            'topx': self.rh_depth_topx,
                            'bottomx': self.rh_depth_bottomx,
                            'topy': self.rh_depth_topy,
                            'bottomy': self.rh_depth_bottomy
                        }
                    else:
                        self.depth_roi['left_hand'] = {
                            'topx': self.lh_depth_topx,
                            'bottomx': self.lh_depth_bottomx,
                            'topy': self.lh_depth_topy,
                            'bottomy': self.lh_depth_bottomy
                        }

                    start_roi = False
                    # Reset roi 
                    self.rh_depth_topx = self.lh_depth_topx = 1
                    self.rh_depth_bottomx = self.lh_depth_bottomx = 0
                    self.rh_depth_topy = self.lh_depth_topy = 1
                    self.rh_depth_bottomy = self.lh_depth_bottomy = 0
                    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
                    print(f"[{ts}]: Latest ROI:")
                    print(self.depth_roi)

                if key == ord('t'):
                    start_roi = False
                    # Reset roi 
                    self.rh_depth_topx = self.lh_depth_topx = 1
                    self.rh_depth_bottomx = self.lh_depth_bottomx = 0
                    self.rh_depth_topy = self.lh_depth_topy = 1
                    self.rh_depth_bottomy = self.lh_depth_bottomy = 0
                    
                elif key == 32:
                    # Pause on space bar
                    cv2.waitKey(0)

    # Capture Depth using ROI specified
    def capture_depth(self):
        self.pipeline = self.create_pipeline_depth()
        self.check_pipeline()
        with dai.Device(self.pipeline) as device:
            ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            print(f"[{ts}]: Pipeline Started...")

            # Queue for depth
            q_d = device.getOutputQueue(name=self.depth_stream_name, maxSize=4, blocking=False)
            # current_fps
            self.current_fps = FPS(mean_nb_frames=20)
            depth_data = []
            frame_number = 0
            start_capture = False

            while True:
                # print(device.getChipTemperature().average)
                frame_number += 1
                self.current_fps.update()
                # Get frame
                in_depth = q_d.get()
                self.depth_frame = in_depth.getFrame()
                w, h = self.depth_frame.shape[1], self.depth_frame.shape[0]
                
                if self.depth_roi is not None and start_capture is True:
                    # Left Hand
                    topx = int(self.depth_roi['left_hand']['topx'] * w)
                    bottomx = int(self.depth_roi['left_hand']['bottomx'] * w)
                    topy = int(self.depth_roi['left_hand']['topy'] * h)
                    bottomy = int(self.depth_roi['left_hand']['bottomy'] * h)
                    dd = self.depth_frame[topy:bottomy+1, topx:bottomx+1]
                    datapoint = {
                        'hand': 'left',
                        'depth_map': dd,
                        'timestamp': datetime.now(),
                        'frame': frame_number
                    }                    
                    depth_data.append(datapoint)

                    # Right Hand
                    topx = int(self.depth_roi['right_hand']['topx'] * w)
                    bottomx = int(self.depth_roi['right_hand']['bottomx'] * w)
                    topy = int(self.depth_roi['right_hand']['topy'] * h)
                    bottomy = int(self.depth_roi['right_hand']['bottomy'] * h)
                    dd = self.depth_frame[topy:bottomy+1, topx:bottomx+1]
                    datapoint = {
                        'hand': 'right',
                        'depth_map': dd,
                        'timestamp': datetime.now(),
                        'frame': frame_number
                    }                    
                    depth_data.append(datapoint)
                    
                    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
                    print(f"[{ts}] Saving frame: {frame_number}")
    

                # Show depth
                instr = "q: quit | r: start capture | d: delete | s: save"
                self.show_depth_map(instr)
                
                # Commands
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # quit
                    break

                if key == ord('r'):
                    # Start depth data 
                    start_capture = True

                if key == ord('d'):
                    # Start depth data 
                    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
                    print(f"[{ts}] Reset...")
                    depth_data = []
                    start_capture = False                    
                    
                if key == ord('s'):
                    # Save capture
                    start_capture = False
                    self.depth_data = depth_data
                    depth_data = []

                elif key == 32:
                    # Pause on space bar
                    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Capture FPS")
    parser.add_argument('--prwidth', default=PARAMS['PREVIEW_WIDTH'], type=int, help="Preview Width")
    parser.add_argument('--prheight', default=PARAMS['PREVIEW_HEIGHT'], type=int, help="Preview Height")
    parser.add_argument('--pixbuff', default=PARAMS['HAND_BUFFER_PIXELS'], type=int, help="Extra buffer for hand gestures in pixels")
    parser.add_argument('--hsize', default=PARAMS['HAND_SIZE'], type=int, help="Frame size for showing hand")
    parser.add_argument('--mode', default=0, type=int, help="Capture Mode: 0 -> Capture ROI, 1 -> Depth")
    parser.add_argument('--prefix', default='capture', type=str, help="Depth dataset prefix name")
    parser.add_argument('--antenna', default=PARAMS['ANTENNA_ROI_FILENAME'], type=str, help="ROI of the Theremin antenna")
    parser.add_argument('--body', default=PARAMS['BODY_ROI_FILENAME'], type=str, help="ROI of body position")
    args = parser.parse_args()

    print(team.banner)

    # Read positon Rois
    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    print(f"[{ts}] Reading ROIs...")
    filename_1 = f"{PARAMS['DATASET_PATH']}/{args.body}"
    filename_2 = f"{PARAMS['DATASET_PATH']}/{args.antenna}"
    rois = None
    with open(filename_1, "rb") as fl1:
        with open(filename_2, "rb") as fl2:
            rois = {}
            rois['body'] = pickle.load(fl1)
            rois['antenna'] = pickle.load(fl2)
            for k, v in rois['body'].items(): print(f"{k}: {v}")
            for k, v in rois['antenna'].items(): print(f"{k}: {v}")        

    if rois is None:
        print("ROIs not defined: Please run the previous step for configuration")
        exit()

    # Message Queue
    messages = Queue.Queue()

    # Datasette recorder
    datasette = DatasetteDepthCapture(
        queue=messages,
        fps=args.fps,
        preview_width=args.prwidth,
        preview_height=args.prheight,
        hand_buffer_pixels=args.pixbuff
    )

    # The ROIs
    datasette.set_ROIs(rois)

    # mode 0: Capture the hands regions (left and right)
    if args.mode == 0:
        # Capture Hands ROI
        datasette.capture_hand() 
        # Save latest ROI into file
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        if datasette.depth_roi is not None and len(datasette.depth_roi) > 0:
            # Save to file
            roi = datasette.depth_roi
            filename = PARAMS['DATASET_PATH'] + "/" + PARAMS['DEPTH_ROI_FILENAME']
            pickle.dump(roi, open(filename, "wb"))
            print(f"[{ts}]: ROI saved to file: {filename}")
        else:
            print(f"[{ts}]: No ROI defined")
    else:
        # Read ROI from file (allows to capture a dataset for future analysis)
        filename = PARAMS['DATASET_PATH'] + "/" + PARAMS['DEPTH_ROI_FILENAME']
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        if os.path.isfile(filename):
            print(f"[{ts}]: Reading ROI from file: {filename}")
            with open(filename, "rb") as file:
                roi = pickle.load(file)
                print("---> Left Hand ROI:")
                for k, v in roi['left_hand'].items(): print(f"{k}: {v}")
                print("---> Right Hand ROI:")
                for k, v in roi['right_hand'].items(): print(f"{k}: {v}")
                datasette.set_depth_ROI(roi)
        else:
            print(f"[{ts}]: No ROI defined: {filename}")

        # Start Capture            
        datasette.capture_depth() 
        # Verify if there was data captured and save to file
        if datasette.depth_data is not None and len(datasette.depth_data) > 0:
            # Save to file
            depth_data = datasette.depth_data
            filename = f"{PARAMS['DATASET_PATH']}/{args.prefix}_{PARAMS['DEPTH_CAPTURE_FILENAME']}"
            pickle.dump(depth_data, open(filename, "wb"))
            print(f"[{ts}]: Depth Data saved to file: {filename}, {len(depth_data)} datapoints")
        else:
            print(f"[{ts}]: No Depth Data was captured")
   
    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    print(f"[{ts}]: Closing...")
    cv2.destroyAllWindows()