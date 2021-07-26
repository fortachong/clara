#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Ether: A Depth based theremin
# At the moment relies on supercollider for synthesis
# But future version will have its own internal synth
# Also could provide controller features through midi/osc
# in the future

import os
import cv2
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import equal_tempered as eqtmp
from pythonosc import udp_client
import threading
import queue as Queue
import argparse
import pickle
import utils
import config
import team

PARAMS = config.PARAMS


def get_z(frame, depth_threshold_max, depth_threshold_min):
    z = None
    if frame is not None:
        dframe = frame.copy()
        filter_cond = (dframe > depth_threshold_max) | (dframe < depth_threshold_min)
        z = dframe[~filter_cond]
    return z


# Ether: main class implementing the OAKD pipeline for the depth based theremin
class Ether:
    def __init__(
            self, 
            antenna_roi=None,
            depth_threshold_max=700,
            depth_threshold_min=400,
            cam_resolution='400'
        ):
        # Camera options = 400, 720, 800
        cam_res = PARAMS['DEPTH_CAMERA_RESOLUTIONS']
        cam_resolution = str(cam_resolution)
        self.depth_mono_resolution_left = cam_res[cam_resolution][0]
        self.depth_mono_resolution_right = cam_res[cam_resolution][0]
        self.depth_res_w = cam_res[cam_resolution][1]
        self.depth_res_h = cam_res[cam_resolution][2]
        self.depth_stream_name = 'depth'
        # ROI for depth
        self.depth_topx = 1
        self.depth_bottomx = 0
        self.depth_topy = 1
        self.depth_bottomy = 0
        self.depth_roi = None

        self.depth_frame = None
        self.show_depth = True
        self.depth_data = None

        # Depth thresholds (experimentally obtained)
        self.depth_threshold_max = depth_threshold_max
        self.depth_threshold_min = depth_threshold_min

        # Pipeline
        self.pipeline = None

        # Antenna roi
        self.antenna_x = 0
        self.antenna_z = 0
        self.antenna_roi = antenna_roi
        # Transform x and y to the same cordinates used
        if self.antenna_roi is not None:
            x1 = self.antenna_roi['absolute']['bottomx']
            self.antenna_z = self.antenna_roi['z']
            self.antenna_x = ((x1 - PARAMS['INTRINSICS_RIGHT_CX'])*self.antenna_z)/PARAMS['INTRINSICS_RIGHT_FX']

    # Set ROI 
    def set_ROI(self, roi):
        self.depth_roi = roi

    # Check current pipeline
    def check_pipeline(self):
        if self.pipeline is not None:
            node_map = self.pipeline.getNodeMap()
            for idx, node in node_map.items():
                print(f"{idx}: {node.getName()}")

    # Show display with depth
    def show_depth_map(
                    self, 
                    instr, 
                    topx1, 
                    topy1, 
                    bottomx1, 
                    bottomy1,
                    topx2, 
                    topy2, 
                    bottomx2, 
                    bottomy2
                ):
        if self.depth_frame is not None:
            dframe = self.depth_frame.copy()
            depth_frame_color = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_color = cv2.equalizeHist(depth_frame_color)
            depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_OCEAN)
            
            # region 1
            cv2.rectangle(depth_frame_color, (topx1, topy1), (bottomx1, bottomy1), (0,0,255),2)
            # region 2
            cv2.rectangle(depth_frame_color, (topx2, topy2), (bottomx2, bottomy2), (0,0,255),2)

            # antenna position
            if self.antenna_roi is not None:
                cv2.line(depth_frame_color, 
                    (self.antenna_roi['absolute']['topx'], 0),
                    (self.antenna_roi['absolute']['topx'], self.preview_height),
                    (0,255,0),
                    2
                )

            self.current_fps.display(depth_frame_color, orig=(50,20), color=(0,0,255), size=0.6)
            self.show_instructions(instr, depth_frame_color, orig=(50,40), color=(0,0,255), size=0.6)
            cv2.imshow(self.depth_stream_name, depth_frame_color)

    # Show display with depth
    def show_depth_map_segmentation(self, topx, topy, bottomx, bottomy):
        if self.depth_frame is not None:
            dframe = self.depth_frame.copy()
            dframe[dframe > self.depth_threshold_max] = 2**16 - 1
            dframe[dframe < self.depth_threshold_min] = 2**16 - 1
            
            depth_frame_color = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_color = cv2.equalizeHist(depth_frame_color)
            depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_OCEAN)    

            # Crop rectangle (right hand)
            if self.depth_roi is not None:
                crop_dm = depth_frame_color[topy:bottomy+1, topx:bottomx+1, :].copy()
                # crop_dm = cv2.resize(crop_dm, (crop_dm.shape[1]*2, crop_dm.shape[0]*2), interpolation=cv2.INTER_AREA)
                cv2.imshow('thresholded', crop_dm)

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
        median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

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

    # Capture Depth using ROI specified, stream all data through a queue
    def capture_depth(self, queue, stop_flag):
        # Create pipeline
        self.pipeline = self.create_pipeline_depth()
        self.check_pipeline()
        with dai.Device(self.pipeline) as device:
            ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            print(f"[{ts}]: Pipeline Started...")

            # Queue for depth
            q_d = device.getOutputQueue(name=self.depth_stream_name, maxSize=4, blocking=False)
            # current_fps
            self.current_fps = FPS(mean_nb_frames=20)
            
            # define inline function to get coordinates in mm and px
            x_coordinate_mm = lambda x, z: ((x - PARAMS['INTRINSICS_RIGHT_CX'])*z)/PARAMS['INTRINSICS_RIGHT_FX']
            y_coordinate_mm = lambda y, z: ((y - PARAMS['INTRINSICS_RIGHT_CY'])*z)/PARAMS['INTRINSICS_RIGHT_FY']
            x_coordinate_px = lambda x: int(x * self.depth_res_w)
            y_coordinate_px = lambda y: int(y * self.depth_res_h)
            distance_to_antenna = lambda x, z: np.sqrt((x - self.antenna_x)**2 + (z - self.antenna_z)**2)

            # If Roi is not defined, stop everything


            if self.depth_roi is not None:
                # ROI (right hand)
                topx_rh = x_coordinate_px(self.depth_roi['right_hand']['topx'])
                bottomx_rh = x_coordinate_px(self.depth_roi['right_hand']['bottomx']) 
                topy_rh = y_coordinate_px(self.depth_roi['right_hand']['topy'])
                bottomy_rh = y_coordinate_px(self.depth_roi['right_hand']['bottomy'])           

                # Get limits (right hand)
                min_x_min_z_rh = x_coordinate_mm(topx_rh, self.depth_threshold_min)
                max_x_min_z_rh = x_coordinate_mm(bottomx_rh, self.depth_threshold_min)
                min_x_max_z_rh = x_coordinate_mm(topx_rh, self.depth_threshold_max)
                max_x_max_z_rh = x_coordinate_mm(bottomx_rh, self.depth_threshold_max) 
                
                # ROI (left hand)
                topx_lh = x_coordinate_px(self.depth_roi['left_hand']['topx'])
                bottomx_lh = x_coordinate_px(self.depth_roi['left_hand']['bottomx']) 
                topy_lh = y_coordinate_px(self.depth_roi['left_hand']['topy'])
                bottomy_lh = y_coordinate_px(self.depth_roi['left_hand']['bottomy'])           

                # Get yz limits (left hand)
                min_y_min_z_lh = y_coordinate_mm(topy_lh, self.depth_threshold_min)
                max_y_min_z_lh = y_coordinate_mm(bottomy_lh, self.depth_threshold_min)
                min_y_max_z_lh = y_coordinate_mm(topy_lh, self.depth_threshold_max)
                max_y_max_z_lh = y_coordinate_mm(bottomy_lh, self.depth_threshold_max) 

                # Distances to antena (right hand)
                d_min_x_min_z_rh = distance_to_antenna(min_x_min_z_rh, self.depth_threshold_min)
                d_max_x_min_z_rh = distance_to_antenna(max_x_min_z_rh, self.depth_threshold_min)
                d_min_x_max_z_rh = distance_to_antenna(min_x_max_z_rh, self.depth_threshold_max)
                d_max_x_max_z_rh = distance_to_antenna(max_x_max_z_rh, self.depth_threshold_max)

                dmin_rh = min(d_min_x_min_z_rh, d_max_x_min_z_rh, d_min_x_max_z_rh, d_max_x_max_z_rh)
                dmax_rh = max(d_min_x_min_z_rh, d_max_x_min_z_rh, d_min_x_max_z_rh, d_max_x_max_z_rh)
                ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
                print(f"[{ts}]: Distances: {d_min_x_min_z_rh}, {d_max_x_min_z_rh}, {d_min_x_max_z_rh}, {d_max_x_max_z_rh}")
                print(f"Min: {dmin_rh}")
                print(f"Max: {dmax_rh}")

                # Display Loop
                while True:
                    self.current_fps.update()
                    # Get frame
                    in_depth = q_d.get()
                    self.depth_frame = in_depth.getFrame()

                    # Send data to queue (only right hand at the moment)
                    message = {
                        'DATA': 1,
                        'depth': self.depth_frame,
                        'roi': {
                            'topx': topx_rh,
                            'topy': topy_rh,
                            'bottomx': bottomx_rh,
                            'bottomy': bottomy_rh
                        },
                        'antenna_x': self.antenna_x,
                        'antenna_z': self.antenna_z,
                        'depth_threshold_min': self.depth_threshold_min,
                        'depth_threshold_max': self.depth_threshold_max,
                        'dmin': dmin_rh,
                        'dmax': dmax_rh
                    }
                    queue.put(message)

                    # Show depth
                    instr = "q: quit"
                    self.show_depth_map(
                                    instr, 
                                    topx_rh, 
                                    topy_rh, 
                                    bottomx_rh, 
                                    bottomy_rh,
                                    topx_lh, 
                                    topy_lh, 
                                    bottomx_lh, 
                                    bottomy_lh
                                )

                    # Show threshold image
                    self.show_depth_map_segmentation(topx_rh, topy_rh, bottomx_rh, bottomy_rh)
                    
                    # Commands
                    key = cv2.waitKey(1) 
                    if key == ord('q') or key == 27:
                        # quit
                        break
