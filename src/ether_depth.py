#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Capture depth in a 3d roi defined previously through calibration

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
    'BODY_ROI_FILENAME': 'roi_position.pkl',
    'ANTENNA_ROI_FILENAME': 'antenna_position.pkl',
    'BODY_BUFFER': 50,
    'ANTENNA_BUFFER': 5,
    'DEPTH_RESOLUTION': '400',
    'INTRINSICS_RIGHT_CX': 318.04592896,
    'INTRINSICS_RIGHT_CY': 198.99064636,
    'INTRINSICS_RIGHT_FX': 427.05795288,
    'INTRINSICS_RIGHT_FY': 427.38696289,
    'SC_SERVER': '127.0.0.1',
    'SC_PORT': 57121
} 

def xyz_numpy(frame, idxs, topx, topy, cx, cy, fx, fy):
    u = idxs[:,1]
    v = idxs[:,0]
    z = frame[v,u]
    x = ((u + topx - cx)*z)/fx
    y = ((v + topy - cy)*z)/fy
    return x, y, z

def xyz(frame, idxs, topx, topy, cx, cy, fx, fy):
    xyz_c = []
    for v, u in idxs:
        z = frame[v, u]
        x = ((u + topx - cx)*z)/fx
        y = ((v + topy - cy)*z)/fy
        xyz_c.append([x,y,z])
    return xyz_c

def get_z(frame, depth_threshold_max, depth_threshold_min):
    z = None
    if frame is not None:
        dframe = frame.copy()
        filter_cond = (dframe > depth_threshold_max) | (dframe < depth_threshold_min)
        z = dframe[~filter_cond]
    return z

def transform_xyz(frame, topx, topy, depth_threshold_max, depth_threshold_min):
    point_cloud = None
    if frame is not None:
        dframe = frame.copy()
        filter_cond = (dframe > depth_threshold_max) | (dframe < depth_threshold_min)
        dm_frame_filtered_idxs = np.argwhere(~filter_cond)
        point_cloud = xyz_numpy(
            dframe, 
            dm_frame_filtered_idxs,
            topx,
            topy,
            PARAMS['INTRINSICS_RIGHT_CX'],
            PARAMS['INTRINSICS_RIGHT_CY'],
            PARAMS['INTRINSICS_RIGHT_FX'],
            PARAMS['INTRINSICS_RIGHT_FY']
        )    
    return point_cloud

class DepthTheremin:
    def __init__(
            self, 
            queue,
            fps=30,
            preview_width=640,
            preview_height=400,
            antenna_roi=None,
            depth_stream_name='depth', 
            depth_threshold_max=700,
            depth_threshold_min=400,
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

        # Depth thresholds (experimentally obtained)
        self.depth_threshold_max = depth_threshold_max
        self.depth_threshold_min = depth_threshold_min

        # Preview size
        self.preview_width = preview_width
        self.preview_height = preview_height

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

    def show_monitor(self, device, frame):
        pass


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
            frame_number = 0
            
            if self.depth_roi is not None:
                # Fixed parameters right hand
                topx_rh = int(self.depth_roi['right_hand']['topx'] * self.depth_res_w)
                bottomx_rh = int(self.depth_roi['right_hand']['bottomx'] * self.depth_res_w)
                topy_rh = int(self.depth_roi['right_hand']['topy'] * self.depth_res_h)
                bottomy_rh = int(self.depth_roi['right_hand']['bottomy'] * self.depth_res_h)
                # Distances x,z
                c1_rh = (((topx_rh - PARAMS['INTRINSICS_RIGHT_CX']) * self.depth_threshold_min)/PARAMS['INTRINSICS_RIGHT_FX'], self.depth_threshold_min)
                c2_rh = (((bottomx_rh - PARAMS['INTRINSICS_RIGHT_CX']) * self.depth_threshold_min)/PARAMS['INTRINSICS_RIGHT_FX'], self.depth_threshold_min)
                c3_rh = (((topx_rh - PARAMS['INTRINSICS_RIGHT_CX']) * self.depth_threshold_max)/PARAMS['INTRINSICS_RIGHT_FX'], self.depth_threshold_max)
                c4_rh = (((bottomx_rh - PARAMS['INTRINSICS_RIGHT_CX']) * self.depth_threshold_max)/PARAMS['INTRINSICS_RIGHT_FX'], self.depth_threshold_max)
                d1_rh = np.sqrt((c1_rh[0] - self.antenna_x)**2 + (c1_rh[1] - self.antenna_z)**2)
                d2_rh = np.sqrt((c2_rh[0] - self.antenna_x)**2 + (c2_rh[1] - self.antenna_z)**2)
                d3_rh = np.sqrt((c3_rh[0] - self.antenna_x)**2 + (c3_rh[1] - self.antenna_z)**2)
                d4_rh = np.sqrt((c4_rh[0] - self.antenna_x)**2 + (c4_rh[1] - self.antenna_z)**2)

                dmin_rh = min(d1_rh, d2_rh, d3_rh, d4_rh)
                dmax_rh = max(d1_rh, d2_rh, d3_rh, d4_rh)
                ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
                print(f"[{ts}]: Distances: {d1_rh}, {d2_rh}, {d3_rh}, {d4_rh}")
                print(f"Min: {dmin_rh}")
                print(f"Max: {dmax_rh}")

                print(c1_rh)
                print(c2_rh)
                print(c3_rh)
                print(c4_rh)



                # Fixed parameters left hand
                topx_lh = int(self.depth_roi['left_hand']['topx'] * self.depth_res_w)
                bottomx_lh = int(self.depth_roi['left_hand']['bottomx'] * self.depth_res_w)
                topy_lh = int(self.depth_roi['left_hand']['topy'] * self.depth_res_h)
                bottomy_lh = int(self.depth_roi['left_hand']['bottomy'] * self.depth_res_h)

                
                
                # Display Loop
                while True:
                    # print(device.getChipTemperature().average)
                    frame_number += 1
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
                    self.queue.put(message)

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

# Create Synth in supercollider 
class EtherSynth:
    def __init__(
            self,
            sc_server,
            sc_port
        ):
        self.sc_server = sc_server
        self.sc_port = sc_port
        print("> Initializing SC Client at {}:{} <".format(self.sc_server, self.sc_port))
        
    def set_tone(self, frequency):
        print("------> theremin freq: {} Hz <------".format(frequency))
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/f", frequency)

        
    def set_volume(self, volume):
        print("------> theremin vol: {} <------".format(volume))
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/a", volume)


# Process messages from inference (depth map)
# and send proper parameters to synthesizer
class SynthMessageProcessor(threading.Thread):
    def __init__(
            self, 
            queue, 
            synth, 
            scale,
            adjust_dmin=20,
            adjust_dmax=500
        ):
        threading.Thread.__init__(self)
        self.synth = synth
        self.queue = queue
        self.active = True
        self.dmin = adjust_dmin
        self.dmax = adjust_dmax

        # Scale
        self.scale = scale

        # Vol
        self.volume = 0

    # Process a Hand Landmark Message
    def process(self, message):
        if 'DATA' in message:
            self.synth.set_volume(1)
            # Only z
            #zs = get_z(message['depth'], message['depth_threshold_max'], message['depth_threshold_min'])
            #distance = np.mean(zs)
            #print(f"----> Centroid Z: {centroid_z}")

            points = transform_xyz(
                message['depth'], 
                message['roi']['topx'], 
                message['roi']['topy'], 
                message['depth_threshold_max'],
                message['depth_threshold_min']
            )
            if points is not None:
                # Calculates Centroid (x,y), ignore y
                # and distance to Antenna center (kind of)
                points_x = points[0]
                points_z = points[2]
                centroid_x = np.mean(points_x)
                centroid_z = np.mean(points_z)
                distance = np.sqrt((centroid_x-message['antenna_x'])**2 + (centroid_z-message['antenna_z'])**2)
                print(distance)
                range_ = self.dmax - self.dmin
                f0 = np.clip(distance, self.dmin, self.dmax) - self.dmin
                f0 = 1 - f0 / range_
                print(f0)
                #print("----> (x, z) Info:")
                #print(f"----> Centroid (X, Z): ({centroid_x}, {centroid_z})")
                #print(f"----> Distance to ({self.antenna_x}, {self.antenna_z}): {distance}")

                # process the thresholds
                #rang = message['depth_threshold_max'] - message['depth_threshold_min']
                #f0 = np.clip(distance, message['depth_threshold_min'], message['depth_threshold_max']) - message['depth_threshold_min']
                #f0 = 1 - f0 / rang
                #print(f0)
                freq = self.scale.from_0_1_to_f(f0)
                print(freq)
                # send to synth
                self.synth.set_tone(freq)

    # Run thread
    def run(self):
        while self.active:
            message = self.queue.get()
            if 'STOP' in message:
                self.active = False
            else:
                # Process
                self.process(message)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scserver', default="{}".format(PARAMS['SC_SERVER']), type=str, 
                        help="IP Address of Supercollider Server")
    parser.add_argument("--scport", default=PARAMS['SC_PORT'], type=int,
                        help="Port of Supercollider Server")
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Capture FPS")
    parser.add_argument('--antenna', default=PARAMS['ANTENNA_ROI_FILENAME'], type=str, help="ROI of the Theremin antenna")
    parser.add_argument('--body', default=PARAMS['BODY_ROI_FILENAME'], type=str, help="ROI of body position")
    parser.add_argument('--prwidth', default=PARAMS['PREVIEW_WIDTH'], type=int, help="Preview Width")
    parser.add_argument('--prheight', default=PARAMS['PREVIEW_HEIGHT'], type=int, help="Preview Height")
    args = parser.parse_args()

    print(team.banner)

    # Read positon Rois. If rois missing -> go to configuration
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
        print("ROIs not defined: Please run configuration")
        exit()


    # Message Queue
    messages = Queue.Queue()

    # Depth based Theremin 
    the = DepthTheremin(
        queue=messages,
        fps=args.fps,
        preview_width=args.prwidth,
        preview_height=args.prheight,
        depth_threshold_min=rois['antenna']['z'] + PARAMS['ANTENNA_BUFFER'],
        depth_threshold_max=rois['body']['z'] - PARAMS['BODY_BUFFER'],
        antenna_roi=rois['antenna']
    )

    # Read ROI from file
    filename = PARAMS['DATASET_PATH'] + "/" + PARAMS['DEPTH_ROI_FILENAME']
    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    if os.path.isfile(filename):
        print(f"[{ts}]: Reading ROI from file: {filename}")
        with open(filename, "rb") as file:
            roi = pickle.load(file)
            print(roi)
            the.set_ROI(roi)
    else:
        print(f"[{ts}]: No ROI defined: {filename}")

    if the.depth_roi is not None:
        scale = eqtmp.EqualTempered(octaves=7, start_freq=220, resolution=1000)
        # Create Synthesizer
        synth = EtherSynth(args.scserver, args.scport)
        # Process Thread
        smp = SynthMessageProcessor(messages, synth, scale)
        smp.start()
        # Depth
        the.capture_depth()
        # quit
        message = {'STOP': 1}
        messages.put(message)
        cv2.destroyAllWindows()
