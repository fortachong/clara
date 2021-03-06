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
import time
import utils
import config
import team

PARAMS = config.PARAMS

# xyz coordinates from depth map
def xyz(frame, idxs, topx, topy, cx, cy, fx, fy):
    xyz_c = []
    for v, u in idxs:
        z = frame[v, u]
        x = ((u + topx - cx)*z)/fx
        y = ((v + topy - cy)*z)/fy
        xyz_c.append([x,y,z])
    return xyz_c


# Numpy version
def xyz_numpy(frame, idxs, topx, topy, cx, cy, fx, fy):
    u = idxs[:,1]
    v = idxs[:,0]
    z = frame[v,u]
    x = ((u + topx - cx)*z)/fx
    y = ((v + topy - cy)*z)/fy
    return x, y, z    


def get_z(frame, depth_threshold_max, depth_threshold_min):
    z = None
    if frame is not None:
        dframe = frame.copy()
        filter_cond = (dframe > depth_threshold_max) | (dframe < depth_threshold_min)
        z = dframe[~filter_cond]
    return z


def transform_xyz(
        depth_frame, 
        topx, 
        bottomx, 
        topy, 
        bottomy, 
        depth_threshold_min, 
        depth_threshold_max
    ):
    point_cloud = None
    if depth_frame is not None:
        dframe = depth_frame.copy()
        # Limit the region
        dframe = dframe[topy:bottomy+1, topx:bottomx+1]
        # filter z
        filter_cond_z = (dframe > depth_threshold_max) | (dframe < depth_threshold_min)
        # ids of the filtered dframe
        dm_frame_filtered_idxs = np.argwhere(~filter_cond_z)
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


# Main class implementing the OAKD pipeline
class DepthTheremin:
    def __init__(
            self, 
            queue,
            fps=30,
            antenna_roi=None,
            depth_stream_name='depth', 
            depth_threshold_max=700,
            depth_threshold_min=400,
            cam_resolution='400',
            use_projection=0
        ):
        # Message processing queue
        self.queue = queue
        # Camera options = 400, 720, 800 for depth
        cam_res = PARAMS['DEPTH_CAMERA_RESOLUTIONS']
        cam_resolution = str(cam_resolution)
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
        self.use_projection = use_projection
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
                    (self.antenna_roi['absolute']['topx'], self.depth_res_h),
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
        mono_l.setFps(self.depth_fps)
        mono_r.setResolution(self.depth_mono_resolution_right)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_r.setFps(self.depth_fps)
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
            
            # define inline function to get coordinates in mm and px
            x_coordinate_mm = lambda x, z: ((x - PARAMS['INTRINSICS_RIGHT_CX'])*z)/PARAMS['INTRINSICS_RIGHT_FX']
            x_coordinate_px = lambda x: int(x * self.depth_res_w)
            y_coordinate_mm = lambda y, z: ((y - PARAMS['INTRINSICS_RIGHT_CY'])*z)/PARAMS['INTRINSICS_RIGHT_FY']            
            y_coordinate_px = lambda y: int(y * self.depth_res_h)
            distance_to_antenna = lambda x, z: np.sqrt((x - self.antenna_x)**2 + (z - self.antenna_z)**2)

            if self.depth_roi is not None:
                # Fixed parameters right hand
                topx_rh = x_coordinate_px(self.depth_roi['right_hand']['topx'])
                bottomx_rh = x_coordinate_px(self.depth_roi['right_hand']['bottomx']) 
                topy_rh = y_coordinate_px(self.depth_roi['right_hand']['topy'])
                bottomy_rh = y_coordinate_px(self.depth_roi['right_hand']['bottomy'])           
                # Get xz limits (right hand)
                min_x_min_z_rh = x_coordinate_mm(topx_rh, self.depth_threshold_min)
                max_x_min_z_rh = x_coordinate_mm(bottomx_rh, self.depth_threshold_min)
                min_x_max_z_rh = x_coordinate_mm(topx_rh, self.depth_threshold_max)
                max_x_max_z_rh = x_coordinate_mm(bottomx_rh, self.depth_threshold_max) 
                # Distances to antena (right hand)
                d_min_x_min_z_rh = distance_to_antenna(min_x_min_z_rh, self.depth_threshold_min)
                d_max_x_min_z_rh = distance_to_antenna(max_x_min_z_rh, self.depth_threshold_min)
                d_min_x_max_z_rh = distance_to_antenna(min_x_max_z_rh, self.depth_threshold_max)
                d_max_x_max_z_rh = distance_to_antenna(max_x_max_z_rh, self.depth_threshold_max)
                # Fixed parameters left hand
                topx_lh = x_coordinate_px(self.depth_roi['left_hand']['topx'])
                bottomx_lh = x_coordinate_px(self.depth_roi['left_hand']['bottomx']) 
                topy_lh = y_coordinate_px(self.depth_roi['left_hand']['topy'])
                bottomy_lh = y_coordinate_px(self.depth_roi['left_hand']['bottomy'])
                # Get yz limits (left hand) y is inverted
                min_y_min_z_lh = y_coordinate_mm(bottomy_lh, self.depth_threshold_min)
                max_y_min_z_lh = y_coordinate_mm(topy_lh, self.depth_threshold_min)
                min_y_max_z_lh = y_coordinate_mm(bottomy_lh, self.depth_threshold_max)
                max_y_max_z_lh = y_coordinate_mm(topy_lh, self.depth_threshold_max) 
                # Get min and max y coordinate
                # y_min = min(min_y_min_z_lh, max_y_min_z_lh, min_y_max_z_lh, max_y_max_z_lh)
                # y_max = max(min_y_min_z_lh, max_y_min_z_lh, min_y_max_z_lh, max_y_max_z_lh)
                # it seems like y is inverted
                y_min = max_y_min_z_lh
                y_max = min_y_min_z_lh
                # Min and max distances to antenna
                dmin_rh = min(d_min_x_min_z_rh, d_max_x_min_z_rh, d_min_x_max_z_rh, d_max_x_max_z_rh)
                dmax_rh = max(d_min_x_min_z_rh, d_max_x_min_z_rh, d_min_x_max_z_rh, d_max_x_max_z_rh)
                
                # Get diagonal vector and distance (we could try with antenna reference)
                diag_x = max_x_max_z_rh-self.antenna_x
                diag_z = self.depth_threshold_max-self.antenna_z
                diag_distance = np.sqrt(diag_x**2 + diag_z**2)
                diag_x_u = diag_x/diag_distance
                diag_z_u = diag_z/diag_distance
                vector_u = np.array([diag_x_u, diag_z_u])
                # function to calculate the vector projection
                #def vector_projection(x, z):
                #    vector_p = np.array([x-self.antenna_x, z-self.antenna_z])
                #    p_proj_u = vector_u*(np.dot(vector_u, vector_p))
                #    proj_distance = np.linalg.norm(p_proj_u)
                #    p_proj_u_x = p_proj_u[0] + self.antenna_x
                #    p_proj_u_z = p_proj_u[1] + self.antenna_z
                #    return p_proj_u_x, p_proj_u_z, proj_distance

                # Send message with the settings starting values
                message = {
                    'CONFIG': 1,
                    'roi': {
                        'left': {
                            'topx': topx_lh,
                            'topy': topy_lh,
                            'bottomx': bottomx_lh,
                            'bottomy': bottomy_lh,
                            'y_min': y_min,
                            'y_max': y_max
                        },
                        'right': {
                            'topx': topx_rh,
                            'topy': topy_rh,
                            'bottomx': bottomx_rh,
                            'bottomy': bottomy_rh,
                            'dmin': dmin_rh,
                            'dmax': dmax_rh,
                            'vector_u': vector_u
                        }
                    },
                    'antenna_x': self.antenna_x,
                    'antenna_z': self.antenna_z,
                    'depth_threshold_min': self.depth_threshold_min,
                    'depth_threshold_max': self.depth_threshold_max
                }
                self.queue.put(message)

                # Stream and Display Loop
                while True:
                    self.current_fps.update()
                    # Get frame
                    in_depth = q_d.tryGet()
                    if in_depth is not None:
                        self.depth_frame = in_depth.getFrame()

                        # Send data to queue (only right hand at the moment)
                        message = {
                            'DATA': 1,
                            'depth': self.depth_frame
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
        # print("------> theremin freq: {} Hz <------".format(frequency))
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/f", frequency)

        
    def set_volume(self, volume):
        # print("------> theremin vol: {} <------".format(volume))
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
            left_hand_upper_th=0.8,
            left_hand_lower_th=0.2,
            adjust_dmin=20,
            adjust_dmax=500,
            dist_function=None,
            use_projection=0,
            smooth=0
        ):
        threading.Thread.__init__(self)
        self.synth = synth
        self.queue = queue
        self.active = True
        self.dmin = adjust_dmin
        self.dmax = adjust_dmax
        self.dist_function = dist_function
        self.config = None
        self.use_projection = use_projection
        self.left_hand_upper_threshold = left_hand_upper_th
        self.left_hand_lower_threshold = left_hand_lower_th

        # Scale
        self.scale = scale

        # Vol
        self.volume = 0

        # Exp (to be done)
        self.smooth = smooth
        self.f0_ = 0
        self.f0__ = 0
        self.moving_avg_ = lambda x: x


    # Process a Message
    def process(self, message):
        # Initial Configuration
        if 'CONFIG' in message:
            self.config = message
            self.dmin = self.config['roi']['right']['dmin']
            self.dmax = self.config['roi']['right']['dmax']
            self.range_f = self.dmax - self.dmin
            self.ymin = self.config['roi']['left']['y_min']
            self.ymax = self.config['roi']['left']['y_max']
            self.range_y = self.ymax - self.ymin

            # Define the projection function
            def vector_projection(x, z):
                vector_p = np.array([x-self.config['antenna_x'], z-self.config['antenna_z']])
                p_proj_u = self.config['vector_u']*(np.dot(self.config['vector_u'], vector_p))
                proj_distance = np.linalg.norm(p_proj_u)
                p_proj_u_x = p_proj_u[0] + self.config['antenna_x']
                p_proj_u_z = p_proj_u[1] + self.config['antenna_z']
                return p_proj_u_x, p_proj_u_z, proj_distance

            self.func_project_ = vector_projection
            if self.smooth:
                self.moving_avg_ = lambda x: (self.f0_ + x) /2

            print(self.config)
        # Data message
        if 'DATA' in message:
            if self.config is not None:

                # Right Hand Point cloud (xyz):
                points_rh = transform_xyz(
                    message['depth'],
                    self.config['roi']['right']['topx'],
                    self.config['roi']['right']['bottomx'],
                    self.config['roi']['right']['topy'],
                    self.config['roi']['right']['bottomy'],
                    self.config['depth_threshold_min'],
                    self.config['depth_threshold_max']
                )

                if points_rh is not None:
                    # Calculates Centroid (x,y), ignore y
                    # and calculate distance to Antenna center (kind of)
                    points_rh_x = points_rh[0]
                    points_rh_z = points_rh[2]
                    if points_rh_x.size > 0 and points_rh_z.size > 0:
                        centroid_x, centroid_z, distance = self.dist_function(
                            points_rh_x, 
                            points_rh_z, 
                            self.config['antenna_x'], 
                            self.config['antenna_z']
                        )

                        # use projection along the diagonal
                        if self.use_projection:
                            if distance is not None:
                                _, _, distance = self.func_project_(centroid_x, centroid_z)

                        if distance is not None:
                            f0 = np.clip(distance, self.dmin, self.dmax) - self.dmin
                            f0 = 1 - f0 / self.range_f
                            # f0 = (self.f0_ + self.f0__ + f0)/3
                            # f0 = (self.f0_ + f0)/2
                            f0 = self.moving_avg_(f0)
                            freq = self.scale.from_0_1_to_f(f0)
                            # self.f0__ = self.f0_
                            self.f0_ = f0
                            # send to synth
                            self.synth.set_tone(freq)

                # Left Hand Point cloud (xyz):
                points_lh = transform_xyz(
                    message['depth'],
                    self.config['roi']['left']['topx'],
                    self.config['roi']['left']['bottomx'],
                    self.config['roi']['left']['topy'],
                    self.config['roi']['left']['bottomy'],
                    self.config['depth_threshold_min'],
                    self.config['depth_threshold_max']
                )

                if points_lh is not None:
                    points_lh_y = points_lh[1]
                    points_lh_z = points_lh[2]

                    if points_lh_y.size > 0 and points_lh_z.size > 0:
                        centroid_y, _ = utils.y_filter_out_(points_lh_y, points_lh_z)
                        if centroid_y is not None:
                            v0 = np.clip(centroid_y, self.ymin, self.ymax) - self.ymin
                            v0 = 1 - v0 / self.range_y
                            if v0 > self.left_hand_upper_threshold: v0 = 1
                            if v0 < self.left_hand_lower_threshold: v0 = 0
                            # send vol to synth
                            self.synth.set_volume(v0)

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
    parser.add_argument('--res', default=PARAMS['DEPTH_RESOLUTION'], type=int, help="Depth Resolution used")
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Capture FPS")
    parser.add_argument('--antenna', default=PARAMS['ANTENNA_ROI_FILENAME'], type=str, help="ROI of the Theremin antenna")
    parser.add_argument('--body', default=PARAMS['BODY_ROI_FILENAME'], type=str, help="ROI of body position")
    parser.add_argument('--distance', default=0, type=int, help="Distance mode: 0 -> normal, 1 -> filter outliers, 2 -> only fingers")    
    parser.add_argument('--proj', default=0, type=int, help="1 -> Use projection over diagonal")
    parser.add_argument('--octaves', default=3, type=int, help="Octaves")
    parser.add_argument('--smooth', default=0, type=int, help="1 -> Moving average frequency with previous sample")
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
        cam_resolution=args.res,
        depth_threshold_min=rois['antenna']['z'] + PARAMS['ANTENNA_BUFFER'],
        depth_threshold_max=rois['body']['z'] - PARAMS['BODY_BUFFER'],
        antenna_roi=rois['antenna'],
        use_projection=args.proj
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
        # distance function
        if args.distance == 1:
            dist_func = lambda px, pz, antx, antz: utils.distance_filter_out_(px, pz, antx, antz)
        elif args.distance == 2:
            dist_func = lambda px, pz, antx, antz: utils.distance_filter_fingers_(px, pz, antx, antz)
        elif args.distance == 3:
            dist_func = lambda px, pz, antx, antz: utils.mean_distance_filter_fingers_centroid(px, pz, antx, antz)
        else:
            dist_func = lambda px, pz, antx, antz: utils.distance_(px, pz, antx, antz)

        scale = eqtmp.EqualTempered(octaves=args.octaves, start_freq=220, resolution=1000)
        # Create Synthesizer
        synth = EtherSynth(args.scserver, args.scport)
        # Process Thread
        smp = SynthMessageProcessor(messages, synth, scale, dist_function=dist_func, smooth=args.smooth)
        smp.start()
        # Depth
        the.capture_depth()
        # quit
        message = {'STOP': 1}
        messages.put(message)
        cv2.destroyAllWindows()
