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

from numpy.core.shape_base import block
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import threading
import queue as Queue
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib
import utils
import config
import team

# Parameters
PARAMS = config.PARAMS

# Get xyz coordinates from depth frame
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


# Class implementing the OAKD pipeline
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
            adjust_dmin=20,
            adjust_dmax=500,
            show_plot=0,
            distance=0
        ):
        # Message processing queue (delete)
        self.queue = queue
        # Camera options = 400, 720, 800 for depth
        cam_res = PARAMS['DEPTH_CAMERA_RESOLUTIONS']
        cam_resolution = str(cam_resolution)
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

        # Cloud of coordinates (list of x,y,z coordinates)
        self.point_cloud = None

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

        # Distances max and min
        self.dmin = adjust_dmin
        self.dmax = adjust_dmax

        # Show visualization
        self.show_plot = show_plot

        # Distance type
        self.distance = distance

    # Transform a region defined by topx, bottomx, topy, bottomy as a point cloud
    def transform_xyz(self, topx, bottomx, topy, bottomy):
        point_cloud = None
        if self.depth_frame is not None:
            dframe = self.depth_frame.copy()
            # Limit the region
            dframe = dframe[topy:bottomy+1, topx:bottomx+1]
            # filter z
            filter_cond_z = (dframe > self.depth_threshold_max) | (dframe < self.depth_threshold_min)
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

    # Show display with depth
    def show_depth_map(
                    self,
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
            if self.depth_roi is not None: 
                cv2.rectangle(depth_frame_color, (topx1, topy1), (bottomx1, bottomy1), (0,255,0),2)
                # region 2
                cv2.rectangle(depth_frame_color, (topx2, topy2), (bottomx2, bottomy2), (0,255,0),2)

                # antenna position
                if self.antenna_roi is not None:
                    cv2.line(depth_frame_color, 
                        (self.antenna_roi['absolute']['topx'], 0),
                        (self.antenna_roi['absolute']['topx'], self.depth_res_h),
                        (0,255,0),
                        2
                    )                     

            # Flip
            depth_frame_color = cv2.flip(depth_frame_color, 1)
            cv2.imshow(self.depth_stream_name, depth_frame_color)

    # Show only the segmented region
    def show_depth_map_segmentation(
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
            dm = np.zeros((self.depth_frame.shape[0], self.depth_frame.shape[1], 3), np.uint8)
            dm[:,:] = (85,83,249)
            dm_frame = dm.copy()
            dm_frame[:,:] = (114,100,76)
            dm[self.depth_frame > self.depth_threshold_max] = (114,100,76)
            dm[self.depth_frame < self.depth_threshold_min] = (114,100,76)
            dm_frame[topy1:bottomy1+1, topx1:bottomx1+1] = dm[topy1:bottomy1+1, topx1:bottomx1+1]
            dm_frame[topy2:bottomy2+1, topx2:bottomx2+1] = dm[topy2:bottomy2+1, topx2:bottomx2+1]
            
            #dframe[dframe > self.depth_threshold_max] = 2**16 - 1
            #dframe[dframe < self.depth_threshold_min] = 2**16 - 1          
            #depth_frame_color = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            #depth_frame_color = cv2.equalizeHist(depth_frame_color)
            #depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_OCEAN)    

            # Crop rectangle (right hand)
            if self.depth_roi is not None:
                #crop_dm = depth_frame_color[topy:bottomy+1, topx:bottomx+1, :].copy()
                #cv2.imshow('thresholded', crop_dm)

                # region 1
                cv2.rectangle(dm_frame, (topx1, topy1), (bottomx1, bottomy1), (0,255,0),2)
                # region 2
                cv2.rectangle(dm_frame, (topx2, topy2), (bottomx2, bottomy2), (0,255,0),2)

                # antenna position
                if self.antenna_roi is not None:
                    cv2.line(dm_frame, 
                        (self.antenna_roi['absolute']['topx'], 0),
                        (self.antenna_roi['absolute']['topx'], self.depth_res_h),
                        (0,255,0),
                        2
                    )                     

                # crop_dm = crop_dm[topy:bottomy+1, topx:bottomx+1, :].copy()
                dm_frame = cv2.flip(dm_frame, 1)
                self.current_fps.display(dm_frame, orig=(50,20), color=(0,255,0), size=0.6)
                self.show_instructions(instr, dm_frame, orig=(50,40), color=(0,255,0), size=0.6) 
                cv2.imshow('thresholded', dm_frame)


    # Set ROI (left and right hand)
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

    # Matplotlib plot Initialization for Right hand xz plot
    def init_plot_xz(
            self, 
            width, height,
            x, z,
            centroid_x,
            centroid_z,
            min_z, max_z, 
            antenna_x, antenna_z, 
            min_x_min_z, max_x_min_z, 
            min_x_max_z, max_x_max_z
        ):
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(width*px, height*px))
        # set limits
        ax.set_xlim((antenna_x - 150, max_x_min_z + 150))
        ax.set_ylim((min_z - 100, max_z + 100))
        # antenna location
        ax.plot(antenna_x, antenna_z, marker='X', color='b', ms=8)
        ax.text(antenna_x + 5, antenna_z - 8, 'ANTENNA', color='b', fontsize='large')
        # draw limiting region
        x0, x1 = ax.get_xlim()
        ax.axline((x0, min_z), (x1, min_z), ls='dashed', color='r', linewidth=1.2)
        ax.axline((x0, max_z), (x1, max_z), ls='dashed', color='r', linewidth=1.2)
        ax.axline((antenna_x, min_z), (min_x_max_z, max_z), ls='dashed', color='r', linewidth=1.2)
        ax.axline((max_x_min_z, min_z), (max_x_max_z, max_z), ls='dashed', color='r', linewidth=1.2)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        # Random initial plot (will be update every frame)
        plot = ax.scatter(x, z, marker='+', color='k', s=2.2)
        centroid_plot = ax.scatter(centroid_x, centroid_z, marker='X', color='r', s=20)
        # draw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, ax, plot, centroid_plot

    # Plot xz (Right Hand)
    def plot_xz(
            self, 
            x,
            z,
            centroid_x,
            centroid_z,
            fig,
            ax,
            plot,
            centroid_plot
        ):
        plot.set_offsets(np.stack([x,z], axis=1))
        centroid_plot.set_offsets(np.array([centroid_x,centroid_z]))
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, ax, plot, centroid_plot

    # Matplotlib plot Initialization for Left hand yz plot
    def init_plot_yz(
            self, 
            width, height,
            y, z,
            centroid_y,
            centroid_z,
            min_z, max_z, 
            min_y_min_z, max_y_min_z, 
            min_y_max_z, max_y_max_z
        ):
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(width*px, height*px))
        # set limits
        ymin = min(min_y_min_z, max_y_min_z, min_y_max_z, max_y_max_z)
        ymax = max(min_y_min_z, max_y_min_z, min_y_max_z, max_y_max_z)
        ax.set_ylim((ymin - 150, ymax + 150))
        ax.set_xlim((min_z - 100, max_z + 100))
        # draw limiting region
        y0, y1 = ax.get_ylim()
        ax.axline((min_z, y0), (min_z, y1), ls='dashed', color='r', linewidth=1.2)
        ax.axline((max_z, y0), (max_z, y1), ls='dashed', color='r', linewidth=1.2)
        ax.axline((min_z, min_y_min_z), (max_z, min_y_max_z), ls='dashed', color='r', linewidth=1.2)
        ax.axline((min_z, max_y_min_z), (max_z, max_y_max_z), ls='dashed', color='r', linewidth=1.2)
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        # Random initial plot (will be update every frame)
        plot = ax.scatter(z, y, marker='+', color='k', s=2.2)
        centroid_plot = ax.scatter(centroid_z, centroid_y, marker='X', color='r', s=20)
        # draw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, ax, plot, centroid_plot

    # Plot yz (Left Hand)
    def plot_yz(
            self, 
            y,
            z,
            centroid_y,
            centroid_z,
            fig,
            ax,
            plot,
            centroid_plot
        ):
        plot.set_offsets(np.stack([z, y], axis=1))
        centroid_plot.set_offsets(np.array([centroid_z, centroid_y]))
        fig.canvas.draw()
        fig.canvas.flush_events()
        return fig, ax, plot, centroid_plot

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

            # distance function
            if self.distance == 1:
                dist_func = lambda px, pz, antx, antz: utils.distance_filter_out_(px, pz, antx, antz)
            elif self.distance == 2:
                dist_func = lambda px, pz, antx, antz: utils.distance_filter_fingers_(px, pz, antx, antz)
            else:
                dist_func = lambda px, pz, antx, antz: utils.distance_(px, pz, antx, antz)

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
                # Fixed parameters left hand
                topx_lh = x_coordinate_px(self.depth_roi['left_hand']['topx'])
                bottomx_lh = x_coordinate_px(self.depth_roi['left_hand']['bottomx']) 
                topy_lh = y_coordinate_px(self.depth_roi['left_hand']['topy'])
                bottomy_lh = y_coordinate_px(self.depth_roi['left_hand']['bottomy'])
                # Get yz limits (left hand)
                min_y_min_z_lh = y_coordinate_mm(topy_lh, self.depth_threshold_min)
                max_y_min_z_lh = y_coordinate_mm(bottomy_lh, self.depth_threshold_min)
                min_y_max_z_lh = y_coordinate_mm(topy_lh, self.depth_threshold_max)
                max_y_max_z_lh = y_coordinate_mm(bottomy_lh, self.depth_threshold_max) 

                # Matplotlib plot
                init_xz = False
                fig_xz = ax_xz = plot_xz = centroid_plot_xz = None
                init_yz = False
                fig_yz = ax_yz = plot_yz = centroid_plot_yz = None
                # Display Loop
                while True:
                    self.current_fps.update()
                    # Get frame
                    in_depth = q_d.get()
                    self.depth_frame = in_depth.getFrame()
                    
                    # Right hand Point Cloud
                    pc_rh = self.transform_xyz(topx_rh, bottomx_rh, topy_rh, bottomy_rh)
                    if pc_rh is not None:
                        # Calculates Centroid (x,y), ignore y
                        # and distance to 0,0 (where ever it is)
                        points_x = pc_rh[0]
                        points_z = pc_rh[2]
                        centroid_x, centroid_z, distance = dist_func(points_x, points_z, self.antenna_x, self.antenna_z)

                        if distance is not None:

                            #centroid_x = np.mean(points_x)
                            #centroid_z = np.mean(points_z)
                            #distance = np.sqrt((centroid_x-self.antenna_x)**2 + (centroid_z-self.antenna_z)**2)
                            #print("----> (x, z) Info:")
                            #print(f"----> Centroid (X, Z): ({centroid_x}, {centroid_z})")
                            #print(f"----> Distance to ({self.antenna_x}, {self.antenna_z}): {distance}")

                            # Show visualization xz
                            if self.show_plot:
                                if init_xz:
                                    if fig_xz is not None:
                                        fig_xz, ax_xz, plot_xz, centroid_plot_xz = self.plot_xz(
                                            points_x,
                                            points_z,
                                            centroid_x,
                                            centroid_z,
                                            fig_xz,
                                            ax_xz,
                                            plot_xz,
                                            centroid_plot_xz
                                        )
                                else:
                                    fig_xz, ax_xz, plot_xz, centroid_plot_xz = self.init_plot_xz(
                                            self.depth_res_w,
                                            self.depth_res_h,
                                            points_x, points_z,
                                            centroid_x,
                                            centroid_z,
                                            self.depth_threshold_min, self.depth_threshold_max, 
                                            self.antenna_x, self.antenna_z, 
                                            min_x_min_z_rh, max_x_min_z_rh, 
                                            min_x_max_z_rh, max_x_max_z_rh
                                    )
                                    init_xz = True
        
                                if fig_xz is not None:
                                    plot_img_xz = np.frombuffer(fig_xz.canvas.tostring_rgb(), dtype=np.uint8)
                                    plot_img_xz  = plot_img_xz.reshape(fig_xz.canvas.get_width_height()[::-1] + (3,))
                                    plot_img_xz = cv2.cvtColor(plot_img_xz, cv2.COLOR_RGB2BGR)
                                    plot_img_xz = cv2.putText(plot_img_xz, f"Distance = {distance}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                                    cv2.imshow("Right Hand xz plot", plot_img_xz)

                    # Left hand Point Cloud
                    pc_lh = self.transform_xyz(topx_lh, bottomx_lh, topy_lh, bottomy_lh)
                    if pc_lh is not None:
                        points_y = pc_lh[1]
                        points_z = pc_lh[2]
                        centroid_y = np.mean(points_y)
                        centroid_z = np.mean(points_z)

                        # Show visualization yz
                        if self.show_plot:
                            if init_yz:
                                if fig_yz is not None:
                                    fig_yz, ax_yz, plot_yz, centroid_plot_yz = self.plot_yz(
                                        points_y,
                                        points_z,
                                        centroid_y,
                                        centroid_z,
                                        fig_yz,
                                        ax_yz,
                                        plot_yz,
                                        centroid_plot_yz
                                    )
                            else:
                                fig_yz, ax_yz, plot_yz, centroid_plot_yz = self.init_plot_yz(
                                        self.depth_res_w,
                                        self.depth_res_h,
                                        points_y, points_z,
                                        centroid_y,
                                        centroid_z,
                                        self.depth_threshold_min, self.depth_threshold_max,
                                        min_y_min_z_lh, max_y_min_z_lh, 
                                        min_y_max_z_lh, max_y_max_z_lh
                                )
                                init_yz = True

                            if fig_yz is not None:
                                plot_img_yz = np.frombuffer(fig_yz.canvas.tostring_rgb(), dtype=np.uint8)
                                plot_img_yz  = plot_img_yz.reshape(fig_yz.canvas.get_width_height()[::-1] + (3,))
                                plot_img_yz = cv2.cvtColor(plot_img_yz, cv2.COLOR_RGB2BGR)
                                cv2.imshow("Left Hand yz plot", plot_img_yz)
                        
                    # Show depth
                    instr = "q: quit"
                    self.show_depth_map(
                                        topx_rh, 
                                        topy_rh, 
                                        bottomx_rh, 
                                        bottomy_rh,
                                        topx_lh, 
                                        topy_lh, 
                                        bottomx_lh, 
                                        bottomy_lh
                                    )

                    # Show threshold image (right)
                    self.show_depth_map_segmentation(
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
                    
                    # Commands
                    key = cv2.waitKey(1) 
                    if key == ord('q') or key == 27:
                        # quit
                        break

                    elif key == 32:
                        # Pause on space bar
                        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Capture FPS")
    parser.add_argument('--res', default=PARAMS['DEPTH_RESOLUTION'], type=int, help="Depth Resolution used")
    parser.add_argument('--antenna', default=PARAMS['ANTENNA_ROI_FILENAME'], type=str, help="ROI of the Theremin antenna")
    parser.add_argument('--body', default=PARAMS['BODY_ROI_FILENAME'], type=str, help="ROI of body position")
    parser.add_argument('--plot', default=0, type=int, help="Show visualization in real time")
    parser.add_argument('--distance', default=0, type=int, help="Distance mode: 0 -> normal, 1 -> filter outliers, 2 -> only fingers")
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
        print("ROIs not defined: Please run configuration")
        exit()

    # Message Queue
    messages = Queue.Queue()
    
    # Depth Theremin (an attempt)
    the = DepthTheremin(
        queue=messages,
        fps=args.fps,
        cam_resolution=args.res,
        depth_threshold_min=rois['antenna']['z'] + PARAMS['ANTENNA_BUFFER'],
        depth_threshold_max=rois['body']['z'] - PARAMS['BODY_BUFFER'],
        antenna_roi=rois['antenna'],
        show_plot=args.plot,
        distance=args.distance
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
        the.capture_depth()
