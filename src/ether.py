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


# Create Synth in supercollider 
class EtherSynth:
    def __init__(
            self,
            scale,
            sc_server,
            sc_port
        ):
        self.scale = scale
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


# Ether: main class implementing the OAKD pipeline for the depth based theremin
class Ether:
    def __init__(
            self, 
            antenna_roi=None,
            depth_threshold_max=700,
            depth_threshold_min=400,
            cam_resolution='400',
            use_projection=0,
            fps=30,
            synth=None,
            dist_function=None
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

    # Set ROI 
    def set_ROI(self, roi):
        self.depth_roi = roi

    # Set synth
    def set_synth(self, synth):
        self.synth = synth

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

    # Capture Depth using ROI specified, stream all data through a queue
    def stream_depth(self, queue, stop_flag):
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

            if self.depth_roi is not None and self.synth is not None:
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
                # function to calculate the vector projection
                #def vector_projection(x, z):
                #    vector_p = np.array([x-self.antenna_x, z-self.antenna_z])
                #    p_proj_u = vector_u*(np.dot(vector_u, vector_p))
                #    proj_distance = np.linalg.norm(p_proj_u)
                #    p_proj_u_x = p_proj_u[0] + self.antenna_x
                #    p_proj_u_z = p_proj_u[1] + self.antenna_z
                #    return p_proj_u_x, p_proj_u_z, proj_distance

                # Send message with the settings starting values
                # message = {
                #     'CONFIG': 1,
                #     'roi': {
                #         'left': {
                #             'topx': topx_lh,
                #             'topy': topy_lh,
                #             'bottomx': bottomx_lh,
                #             'bottomy': bottomy_lh,
                #             'y_min': y_min,
                #             'y_max': y_max
                #         },
                #         'right': {
                #             'topx': topx_rh,
                #             'topy': topy_rh,
                #             'bottomx': bottomx_rh,
                #             'bottomy': bottomy_rh,
                #             'dmin': dmin_rh,
                #             'dmax': dmax_rh
                #         }
                #     },
                #     'antenna_x': self.antenna_x,
                #     'antenna_z': self.antenna_z,
                #     'depth_threshold_min': self.depth_threshold_min,
                #     'depth_threshold_max': self.depth_threshold_max
                # }
                # self.queue.put(message)

                # Stream and Display Loop
                while True:
                    self.current_fps.update()
                    # Get frame
                    in_depth = q_d.get()
                    self.depth_frame = in_depth.getFrame()

                    # Synth part
                    # Right Hand Point cloud (xyz):
                    points_rh = utils.transform_xyz(
                        self.depth_frame,
                        topx_rh,
                        bottomx_rh,
                        topy_rh,
                        bottomy_rh,
                        self.depth_threshold_min,
                        self.depth_threshold_max,
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
                                self.antenna_x, 
                                self.antenna_z
                            )

                            # use projection along the diagonal
                            if self.use_projection:
                                if distance is not None:
                                    _, _, distance = self.func_project_(centroid_x, centroid_z)       

                            if distance is not None:
                                f0 = np.clip(distance, dmin_rh, dmax_rh) - dmin_rh
                                f0 = 1 - f0 / range_rh
                                freq = self.synth.scale.from_0_1_to_f(f0)
                                # send to synth (udp)
                                self.synth.set_tone(freq)



                         

                    # Left Hand Point cloud (xyz):
                    points_lh = utils.transform_xyz(
                        self.depth_frame,
                        topx_lh,
                        bottomx_lh,
                        topy_lh,
                        bottomy_lh,
                        self.depth_threshold_min,
                        self.depth_threshold_max,
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
                                v0 = np.clip(centroid_y, y_min, y_max) - y_min
                                v0 = 1 - v0 / range_y
                                # send vol to synth
                                self.synth.set_volume(v0)

                    # Send data to queue (only right hand at the moment)
                    # message = {
                    #     'DATA': 1,
                    #     'depth': self.depth_frame
                    # }
                    # self.queue.put(message)

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

                cv2.destroyAllWindows()

def gui(queue, start_flag, stop_flag):
    print("GUI processor")
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
        dist_func = lambda px, pz, antx, antz: utils.distance_filter_out_(px, pz, antx, antz)
    elif args.distance == 2:
        dist_func = lambda px, pz, antx, antz: utils.distance_filter_fingers_(px, pz, antx, antz)
    else:
        dist_func = lambda px, pz, antx, antz: utils.distance_(px, pz, antx, antz)

    # Synthesizer
    scale = eqtmp.EqualTempered(octaves=3, start_freq=220, resolution=100)
    # Create Synthesizer
    synth = EtherSynth(scale, args.scserver, args.scport)

    # Depth based Theremin 
    ether = Ether(
        cam_resolution=args.res,
        fps=args.fps,
        depth_threshold_min=rois['antenna']['z'] + PARAMS['ANTENNA_BUFFER'],
        depth_threshold_max=rois['body']['z'] - PARAMS['BODY_BUFFER'],
        antenna_roi=rois['antenna'],
        use_projection=args.proj
    )
    # Set the ROI
    ether.set_ROI(roi)
    # Choose a distance function
    ether.set_dist_function(dist_func)
    # Add the synth part that communicates with supercollider
    ether.set_synth(synth)

    # Gui queue
    gui_q = Queue()



    procs = []
    pr = Process(target=gui, args=(gui_q, START, STOP))
    pr.start()
    procs.append(pr)

    # Stream depth to SC and gui (to be done)
    ether.stream_depth(gui_q, None)


    # Join
    for proc in procs:
        proc.join()  




    # Create synth processor


    # # Synth
    # synth_q = Queue()
    # # Gui
    # gui_q = Queue()

    # # Processes
    # procs = []
    # pr2 = Process(target=send_control_to_synth, args=(synth_q, START, STOP))
    # pr2.start()
    # #pr3
    # procs.append(pr2)

    # # Start Streaming depth values:
    # START.value = True
    # try:
    #     ether.stream_depth(synth_q, STOP)
    # except KeyboardInterrupt:
    #     STOP.value = True
 
    # # complete the processes
    # for proc in procs:
    #     proc.join()   
    