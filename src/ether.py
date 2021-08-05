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
from multiprocessing import Process, Queue, Value
import time
import cv2
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import equal_tempered as eqtmp
from pythonosc import udp_client
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


# Communication with Supercollider
class EtherSupercollider:
    def __init__(
            self,
            sc_server,
            sc_port,
            vol_upper_th=0.8,
            vol_lower_th=0.2
        ):
        self.sc_server = sc_server
        self.sc_port = sc_port
        self.vol_upper_threshold = vol_upper_th
        self.vol_lower_threshold = vol_lower_th
        print("> Initializing SC Client at {}:{} <".format(self.sc_server, self.sc_port))
        
    def set_tone(self, frequency):
        print("------> theremin freq: {} Hz <------".format(frequency))
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/f", frequency)

        
    def set_volume(self, volume):
        if volume > self.vol_upper_threshold: volume = 1
        if volume < self.vol_lower_threshold: volume = 0
        print("------> theremin vol: {} <------".format(volume))
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/a", volume)

# Process messages (config and depth) in order to produce the sound
class EtherSynth:
    def __init__(
            self, 
            supercollider, 
            scale,
            dist_function=None,
            use_projection=0
        ):
        self.supercollider = supercollider
        self.scale = scale
        self.dist_function = dist_function
        self.config_ = None
        self.use_projection = use_projection

        # Try Exp decay
        self.f0_ = 0
        self.f0__ = 0

    # Process a configuration message
    def config(self, message):
        self.config_ = message
        # diagonal distance min and max
        self.dmin = self.config_['roi']['right']['dmin']
        self.dmax = self.config_['roi']['right']['dmax']
        self.range_f = self.dmax - self.dmin
        # y min and max
        self.ymin = self.config_['roi']['left']['y_min']
        self.ymax = self.config_['roi']['left']['y_max']
        self.range_y = self.ymax - self.ymin

        # Define the projection function:
        # it project a vector (x, z) over the diagonal
        def vector_projection(x, z):
            vector_p = np.array([x - self.config_['antenna_x'], z - self.config_['antenna_z']])
            p_proj_u = self.config_['vector_u']*(np.dot(self.config_['vector_u'], vector_p))
            proj_distance = np.linalg.norm(p_proj_u)
            p_proj_u_x = p_proj_u[0] + self.config_['antenna_x']
            p_proj_u_z = p_proj_u[1] + self.config_['antenna_z']
            return p_proj_u_x, p_proj_u_z, proj_distance

        self.func_project_ = vector_projection
        print(self.config_)

    # Process a depth message
    def process(self, message):
        if self.config is not None:
            # Right Hand Point cloud (xyz):
            points_rh = utils.transform_xyz(
                message['depth'],
                self.config_['roi']['right']['topx'],
                self.config_['roi']['right']['bottomx'],
                self.config_['roi']['right']['topy'],
                self.config_['roi']['right']['bottomy'],
                self.config_['depth_threshold_min'],
                self.config_['depth_threshold_max'],
                PARAMS['INTRINSICS_RIGHT_CX'],
                PARAMS['INTRINSICS_RIGHT_CY'],
                PARAMS['INTRINSICS_RIGHT_FX'],
                PARAMS['INTRINSICS_RIGHT_FY']
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
                        freq = self.scale.from_0_1_to_f(f0)
                        # self.f0__ = self.f0_
                        # self.f0_ = f0
                        # send to synth
                        self.supercollider.set_tone(freq)

            # Left Hand Point cloud (xyz):
            points_lh = utils.transform_xyz(
                message['depth'],
                self.config['roi']['left']['topx'],
                self.config['roi']['left']['bottomx'],
                self.config['roi']['left']['topy'],
                self.config['roi']['left']['bottomy'],
                self.config['depth_threshold_min'],
                self.config['depth_threshold_max'],
                PARAMS['INTRINSICS_RIGHT_CX'],
                PARAMS['INTRINSICS_RIGHT_CY'],
                PARAMS['INTRINSICS_RIGHT_FX'],
                PARAMS['INTRINSICS_RIGHT_FY']
            )

            if points_lh is not None:
                points_lh_y = points_lh[1]
                points_lh_z = points_lh[2]

                if points_lh_y.size > 0 and points_lh_z.size > 0:
                    centroid_y, _ = utils.y_filter_out_(points_lh_y, points_lh_z)
                    if centroid_y is not None:
                        v0 = np.clip(centroid_y, self.ymin, self.ymax) - self.ymin
                        v0 = 1 - v0 / self.range_y
                        # send vol to synth
                        self.supercollider.set_volume(v0)

    # Process a Message
    def process_message(self, message):
        # Initial Configuration
        if 'CONFIG' in message:
            self.config(message)

        # Data message
        if 'DATA' in message:
            self.process(self, message)



# Ether: main class implementing the OAKD pipeline for the depth based theremin
class Ether:
    def __init__(
            self, 
            antenna_roi=None,
            depth_threshold_max=700,
            depth_threshold_min=400,
            cam_resolution='400',
            fps=30
        ):
        # Camera options = 400, 720, 800
        cam_res = PARAMS['DEPTH_CAMERA_RESOLUTIONS']
        cam_resolution = str(cam_resolution)
        self.depth_mono_resolution_left = cam_res[cam_resolution][0]
        self.depth_mono_resolution_right = cam_res[cam_resolution][0]
        self.depth_res_w = cam_res[cam_resolution][1]
        self.depth_res_h = cam_res[cam_resolution][2]
        self.depth_stream_name = 'depth'
        self.depth_fps = fps
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

    # Set Distance function
    def set_dist_function(self, dist_function):
        self.dist_function = dist_function

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
        # Mono Camera Settings (Resolution and Fps)
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

    # Capture Depth using ROI specified, stream all data through a queue
    def stream_depth(self, synth_queue, gui_queue, stop_flag):
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
                range_y = y_max - y_min
                # Min and max distances to antenna
                dmin_rh = min(d_min_x_min_z_rh, d_max_x_min_z_rh, d_min_x_max_z_rh, d_max_x_max_z_rh)
                dmax_rh = max(d_min_x_min_z_rh, d_max_x_min_z_rh, d_min_x_max_z_rh, d_max_x_max_z_rh)
                range_rh = dmax_rh - dmin_rh
                
                # Get diagonal vector and distance (we could try with antenna reference)
                diag_x = max_x_max_z_rh-self.antenna_x
                diag_z = self.depth_threshold_max-self.antenna_z
                diag_distance = np.sqrt(diag_x**2 + diag_z**2)
                diag_x_u = diag_x/diag_distance
                diag_z_u = diag_z/diag_distance
                vector_u = np.array([diag_x_u, diag_z_u])
                
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
                synth_queue.put(message)
                gui_queue.put(message)

                # Stream and Display Loop
                while True:
                    self.current_fps.update()
                    # Get frame
                    in_depth = q_d.get()
                    self.depth_frame = in_depth.getFrame()

                    # Send data to queue (only right hand at the moment)
                    message = {
                         'DATA': 1,
                         'depth': self.depth_frame
                    }
                    synth_queue.put(message)
                    gui_queue.put(message)

                    # Show depth 
                    # instr = "q: quit"
                    # self.show_depth_map(
                    #                 instr, 
                    #                 topx_rh, 
                    #                 topy_rh, 
                    #                 bottomx_rh, 
                    #                 bottomy_rh,
                    #                 topx_lh, 
                    #                 topy_lh, 
                    #                 bottomx_lh, 
                    #                 bottomy_lh
                    #             )

                    # Show threshold image
                    # self.show_depth_map_segmentation(topx_rh, topy_rh, bottomx_rh, bottomy_rh)

                    if stop_flag.value:
                        break

                    # Commands
                    # key = cv2.waitKey(1) 
                    # if key == ord('q') or key == 27:
                        # quit
                    #    break

                    # Verify fps    
                    print(self.current_fps.get())

                # cv2.destroyAllWindows()


# Function that implements the synth processor
def process_synth(
        queue, start_flag, stop_flag, 
        sc_server, 
        sc_port, 
        octaves, 
        dist_function, 
        use_projection
    ):
    print("Synth processor started...")
    # Create instances for synthesis
    # Scale
    scale = eqtmp.EqualTempered(octaves=octaves, start_freq=220, resolution=1000)
    # Supercollider
    sc = EtherSupercollider(sc_server, sc_port)
    # Synth
    synth = EtherSynth(
        supercollider=sc,
        scale=scale,
        dist_function=dist_function,
        use_projection=use_projection
    )
    
    # Process the messages in the queue
    try:
        while True:
            if start_flag.value:
                message = queue.get()
                if message is not None:
                    # synth.process_message(message)
                    # print(type(message))
                    pass

                # synth.process_message(message)
            if stop_flag.value:
                break                
    except KeyboardInterrupt:
        stop_flag.value = True
    except Exception as error:
        stop_flag.value = True


# Function that implements the gui processor
def process_gui(queue, start_flag, stop_flag):
    print("GUI processor started...")
    # Process the messages in the queue
    try:
        while True:
            if start_flag.value:
                message = queue.get()
                if message is not None:
                    # synth.process_message(message)
                    print(type(message))
            if stop_flag.value:
                break                
    except KeyboardInterrupt:
        stop_flag.value = True



class EtherGui:
    def __init__(self, octaves):
        self.octaves = octaves
        self.context_ = {
            'previous_lh': None,
            'previous_rh': None,
            'init_xz': False,
            'fig_xz': None,
            'ax_xz': None,
            'plot_xz': None,
            'centroid_plot_xz': None,
            'centroid_plot_xz_f': None,
            'centroid_plot_xz_fingers': None,
            'init_yz': False,
            'fig_yz': None,
            'ax_yz': None,
            'plot_yz': None,
            'centroid_plot_yz': None
        }
    
    def init_context(self):
        #ht = self.octaves*12
        #nvectors = np.linspace(0, 1, ht) * diag_distance
        #notes_ = np.tile(nvectors, 2).reshape((2, -1))
        #us = np.broadcast_to(vector_u, (ht, 2))
        #notes = us.T*notes_ + np.broadcast_to(np.array([antenna_x, antenna_z]), (ht, 2)).T
        pass

    def loop(self, stop_flag):
        pass


if __name__ == "__main__": 
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--scserver', default="{}".format(PARAMS['SC_SERVER']), type=str, 
                        help="IP Address of Supercollider Server")
    parser.add_argument("--scport", default=PARAMS['SC_PORT'], type=int,
                        help="Port of Supercollider Server")
    parser.add_argument('--res', default=PARAMS['DEPTH_RESOLUTION'], type=int, help="Depth Resolution used")
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Capture FPS")
    parser.add_argument('--antenna', default=PARAMS['ANTENNA_ROI_FILENAME'], type=str, help="ROI of the Theremin antenna")
    parser.add_argument('--body', default=PARAMS['BODY_ROI_FILENAME'], type=str, help="ROI of body position")
    parser.add_argument('--proj', default=0, type=int, help='Use projection')
    parser.add_argument('--distance', default=0, type=int, help="Distance mode: 0 -> normal, 1 -> filter outliers, 2 -> only fingers") 
    parser.add_argument('--octaves', default=3, type=int, help="Octaves") 
    args = parser.parse_args()

    print(team.banner)

    # Variables to sync all processes
    START = Value('b', False)
    STOP = Value('b', False)

    # Read files and configure
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
        ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
        print(f"[{ts}]: Body and Antenna ROIs not defined: Please run configuration")
        exit()

    # Read ROI from file
    filename = PARAMS['DATASET_PATH'] + "/" + PARAMS['DEPTH_ROI_FILENAME']
    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
    roi = None
    if os.path.isfile(filename):
        print(f"[{ts}]: Reading ROI from file: {filename}")
        with open(filename, "rb") as file:
            roi = pickle.load(file)
            print(roi)
            rois['left_hand']= roi['left_hand']
            rois['right_hand']= roi['right_hand']
    else:
        print(f"[{ts}]: No Hands ROIs defined: {filename}")
        exit()

    # distance function
    if args.distance == 1:
        dist_func = utils.distance_filter_out_
    elif args.distance == 2:
        dist_func = utils.distance_filter_fingers_
    else:
        dist_func = utils.distance_

    # Depth based Theremin 
    ether = Ether(
        cam_resolution=args.res,
        fps=args.fps,
        depth_threshold_min=rois['antenna']['z'] + PARAMS['ANTENNA_BUFFER'],
        depth_threshold_max=rois['body']['z'] - PARAMS['BODY_BUFFER'],
        antenna_roi=rois['antenna']
    )
    # Set the ROI
    ether.set_ROI(roi)
    # Choose a distance function
    ether.set_dist_function(dist_func)

    # Synth queue
    synth_q = Queue()
    # Gui queue
    gui_q = Queue()

    # Processes
    processes = []
    # synth processor
    synth_ = Process(
        target=process_synth,
        args=(
            synth_q, START, STOP,
            args.scserver, args.scport, args.octaves,
            dist_func, args.proj
        )
    )
    synth_.start()
    processes.append(synth_)


    START.value = True 
    try:
        ether.stream_depth(synth_q, gui_q, STOP)
    except KeyboardInterrupt:
        STOP.value = True
        cv2.destroyAllWindows()


    print("Depth Stream stopped...")
    # join processes
    for process in processes:
        process.join()

    # Close all queues
    synth_q.close()
    gui_q.close()    
    