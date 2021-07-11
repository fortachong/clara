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
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS
import queue as Queue
import argparse
import pickle
import team

# Credits:
# Hand Tracking Model from: geax
# https://github.com/geaxgx/depthai_hand_tracker

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

# Parameters
PARAMS = {
    'CAPTURE_DEVICE': 0,
    'KEY_QUIT': 'q',
    'HUD_COLOR': (153,219,112),
    'LANDMARKS_COLOR': (0,255,0),
    'LANDMARKS': [
                    LM_WRIST, 
                    LM_THUMB_TIP, 
                    LM_INDEX_FINGER_TIP, 
                    LM_MIDDLE_FINGER_TIP,
                    LM_RING_FINGER_TIP,
                    LM_PINKY_TIP
                ],

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
    'DEPTH_RESOLUTION': '400'
} 

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST)
    return resized.transpose(2,0,1)

class DatasetteDepthCapture:
    def __init__(
            self, 
            queue,
            pd_path='',
            pd_score_thresh=0.5,
            pd_nms_thresh=0.3,
            lm_path='',
            lm_score_threshold=0.5,
            preview_width=640,
            preview_height=400,
            hand_buffer_pixels=20,
            hand_size=400,
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

        # Palm detector path
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        # Landmark detector
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.fps = fps
        # For capturing hands
        self.hand_buffer_pixels = hand_buffer_pixels
        self.hand_size = hand_size
        
        # Preview size
        self.preview_width = preview_width
        self.preview_height = preview_height
        # Some flags
        self.use_lm = True
        self.show_landmarks = True
        self.show_handedness = False
        self.show_pd_box = True
        self.show_pd_kps = True
        self.show_rot_rect = False
        self.show_scores = True
        self.show_landmarks = True
        # Palm detector input size
        self.pd_input_length = PARAMS['PALM_DETECTION_INPUT_LENGTH']
        # Landmark detetector input size
        self.lm_input_length = PARAMS['LM_INPUT_LENGTH']
        # Pipeline
        self.pipeline = None

    # Post process inference from Palm Detector
    def pd_postprocess(self, inference):
        scores = np.array(
            inference.getLayerFp16("classificators"), 
            dtype=np.float16) # 896
        bboxes = np.array(
            inference.getLayerFp16("regressors"), 
            dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(
            self.pd_score_thresh, 
            scores, 
            bboxes, 
            self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        mpu.detections_to_rect(self.regions)
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    # Render Palm Detection
    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.frame_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 1)
            if self.show_pd_kps:
                for i,kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.frame_size)
                    y = int(kp[1] * self.frame_size)
                    cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                    cv2.putText(frame, str(i), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Palm score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.frame_size+10), int((r.pd_box[1]+r.pd_box[3])*self.frame_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 2)

    # Process Landmarks
    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]    
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add"))
        # lm_raw = np.array(inference.getLayerFp16("Squeeze"))
        
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3*i:3*(i+1)]/self.lm_input_length)
        region.landmarks = lm

    # Transform xy coordinates to normalized xy and y-rescaled
    def lm_transform(self, region):
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
        mat = cv2.getAffineTransform(src, dst) 

        dst_normal = np.array([ (x/self.frame_size, y/self.frame_size) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
        mat_normal = cv2.getAffineTransform(src, dst_normal)

        lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)

        lm_xy_normal = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
        lm_xy_normal = np.squeeze(cv2.transform(lm_xy_normal, mat_normal)).astype(np.float32)
        
        region.lm_xy = lm_xy    
        region.lm_xy_normalized = lm_xy_normal
        # y rescaled
        lm_xy_y_rescaled = lm_xy_normal.copy()
        lm_xy_y_rescaled[:, 1] = (lm_xy_y_rescaled[:, 1] - self.pad_h / self.frame_size) * (self.frame_size/self.preview_height)
        region.lm_xy_y_rescaled = lm_xy_y_rescaled

    # Render Landmarks
    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            
            if self.show_landmarks:
                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
                # print(region.rect_points[1:])
                mat = cv2.getAffineTransform(src, dst)
                lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
                lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
                list_connections = [[0, 1, 2, 3, 4], 
                                    [0, 5, 6, 7, 8], 
                                    [5, 9, 10, 11, 12],
                                    [9, 13, 14 , 15, 16],
                                    [13, 17],
                                    [0, 17, 18, 19, 20]]
                lines = [np.array([lm_xy[point] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 0, 0), 2, cv2.LINE_AA)
                for x,y in lm_xy:
                    # print(x,y)
                    cv2.circle(frame, (x, y), 2, (0,128,255), -1)

    # Pipeline for hand landmarks
    def create_pipeline_lm(self):
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Creating Pipeline for Landmark Detection ...")

        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(
            num_layers=4, 
            min_scale=0.1484375,
            max_scale=0.75,
            input_size_height=128,
            input_size_width=128,
            anchor_offset_x=0.5,
            anchor_offset_y=0.5,
            strides=[8,16,16,16],
            aspect_ratios=[1.0],
            reduce_boxes_in_lowest_layer=False,
            interpolated_scale_aspect_ratio=1.0,
            fixed_anchor_size=True
        )
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: {self.nb_anchors} anchors have been created")

        # Pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        # Color Camera
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Color Camera ...")       
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        # Palm Detector
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Mediapipe Palm Detector NN ...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        pd_nn.setNumInferenceThreads(2)
        pd_in = pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        # Hand Landmark Detector
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Mediapipe Hand Landmark NN...")       
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        lm_nn.setNumInferenceThreads(2)
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)           
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

            self.show_instructions(instr, depth_frame_color, orig=(50,40), color=(0,0,255), size=0.6)
            cv2.imshow(self.depth_stream_name, depth_frame_color)

    # Set ROI 
    def set_ROI(self, roi):
        self.depth_roi = roi

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
            # 2. In: Palm Detector Input
            q_pd_in = device.getInputQueue(name="pd_in")
            # 3. Out: Palm Detector Output
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=False)
            # 4. Landmarks Out
            q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=False)
            # 5. Landmarks In
            q_lm_in = device.getInputQueue(name="lm_in")
            
            # current_fps
            self.current_fps = FPS(mean_nb_frames=20)
            frame_number = 0
            start_roi = False
            right_capture = True
            
            while True:
                frame_number += 1
                self.current_fps.update()
                        
                # In video queue
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()

                # Dimensions of the Video Frame
                h, w = video_frame.shape[:2]
                self.h = h
                self.w = w
                # Padding top and bottom
                self.frame_size = max(self.h, self.w)
                self.pad_h = (self.frame_size - self.h)//2
                self.pad_w = (self.frame_size - self.w)//2

                video_frame = cv2.copyMakeBorder(
                    video_frame,
                    self.pad_h,
                    self.pad_h,
                    self.pad_w,
                    self.pad_w,
                    cv2.BORDER_CONSTANT
                )

                # Frame for NN
                frame_nn = dai.ImgFrame()
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
                q_pd_in.send(frame_nn)
                
                # Datasette is a cool name
                datasette_frame = video_frame.copy()
                
                # Palm inference
                inference = q_pd_out.get()
                self.pd_postprocess(inference)
                self.pd_render(datasette_frame)
                if self.use_lm:
                    # Prepare data for landmarks
                    for region in self.regions:
                        img_hand = mpu.warp_rect_img(
                            region.rect_points, 
                            video_frame, 
                            self.lm_input_length, 
                            self.lm_input_length
                        )
                        nn_data = dai.NNData()
                        nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                        q_lm_in.send(nn_data)

                    # Retrieve Landmarks
                    for region in self.regions:
                        inference = q_lm_out.get()
                        self.lm_postprocess(region, inference)
                        self.lm_render(datasette_frame, region)

                        if start_roi:
                            self.lm_transform(region)
                            # Save min and max regions
                            dtx_candidate = np.min(region.lm_xy_y_rescaled[:, 0])
                            dbx_candidate = np.max(region.lm_xy_y_rescaled[:, 0])
                            dty_candidate = np.min(region.lm_xy_y_rescaled[:, 1])
                            dby_candidate = np.max(region.lm_xy_y_rescaled[:, 1])
                            
                            # Get rigth hand square enclosure (only right hand)
                            if region.handedness >= 0.8 and right_capture is True:
                                if self.rh_depth_topx > dtx_candidate:
                                    self.rh_depth_topx = dtx_candidate
                                if self.rh_depth_bottomx < dbx_candidate:
                                    self.rh_depth_bottomx = dbx_candidate
                                if self.rh_depth_topy > dty_candidate:
                                    self.rh_depth_topy = dty_candidate
                                if self.rh_depth_bottomy < dby_candidate:
                                    self.rh_depth_bottomy = dby_candidate

                            # Get left hand square enclosure (only left hand)
                            if region.handedness <= 0.2 and right_capture is not True:
                                if self.lh_depth_topx > dtx_candidate:
                                    self.lh_depth_topx = dtx_candidate
                                if self.lh_depth_bottomx < dbx_candidate:
                                    self.lh_depth_bottomx = dbx_candidate
                                if self.lh_depth_topy > dty_candidate:
                                    self.lh_depth_topy = dty_candidate
                                if self.lh_depth_bottomy < dby_candidate:
                                    self.lh_depth_bottomy = dby_candidate

                self.current_fps.display(datasette_frame, orig=(50,50), color=(0,0,255), size=0.6)
                instr = "q: quit | r: start rh ROI | l: start lh ROI | s: save ROI"
                self.show_instructions(instr, datasette_frame, orig=(50,70), color=(0,0,255), size=0.6)
                
                # Show roi
                cv2.imshow("Hand Position Configurations", datasette_frame)

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
                
                if self.depth_roi is not None and start_capture is True:
                    w, h = self.depth_frame.shape[1], self.depth_frame.shape[0]
                    topx = int(self.depth_roi['topx'] * w)
                    bottomx = int(self.depth_roi['bottomx'] * w)
                    topy = int(self.depth_roi['topy'] * h)
                    bottomy = int(self.depth_roi['bottomy'] * h)
                    dd = self.depth_frame[topy:bottomy+1, topx:bottomx+1]
                    datapoint = {
                        'depth_map': dd,
                        'timestamp': datetime.now(),
                        'frame': frame_number
                    }
                    depth_data.append(datapoint)

                # Show depth
                instr = "q: quit | r: start capture | s: save"
                self.show_depth_map(instr)
                
                # Commands
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # quit
                    break

                if key == ord('r'):
                    # Start depth data 
                    start_capture = True
                    
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
    parser.add_argument('--mode', default=0, type=int, help="Capture Mode: 0 -> Capture ROI, 1 -> Depth")
    parser.add_argument('--prefix', default='capture', type=str, help="Depth dataset prefix name")
    args = parser.parse_args()

    print(team.banner)
    # Message Queue
    messages = Queue.Queue()

    # Datasette recorder
    datasette = DatasetteDepthCapture(
        queue=messages,
        pd_path=args.pdblob,
        pd_score_thresh=args.pdth,
        pd_nms_thresh=args.pdnms,
        lm_path=args.lmblob,
        lm_score_threshold=args.lmth,
        fps=args.fps,
        preview_width=args.prwidth,
        preview_height=args.prheight,
        hand_buffer_pixels=args.pixbuff,
        hand_size=args.hsize
    )

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
                print(roi)
                datasette.set_ROI(roi)
        else:
            print(f"[{ts}]: No ROI defined: {filename}")                            
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