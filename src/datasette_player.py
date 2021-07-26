#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Allows to play a dataset capture only for right hand
# Controls: play, restart, pause, stop
# Slow control 0.5 -> 1

import os
import time
import cv2
import pickle
import argparse
from datetime import datetime
from matplotlib.colors import rgb_to_hsv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import utils
import config
import team

PARAMS = config.PARAMS


# Show only the segmented region
def show_depth_map_segmentation(
                            depth_frame,
                            antenna_x_abs,
                            height,
                            topx1, 
                            topy1, 
                            bottomx1, 
                            bottomy1,
                            topx2, 
                            topy2, 
                            bottomx2, 
                            bottomy2
                        ):
    dm = np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), np.uint8)
    dm[:,:] = (85,83,249)
    dm_frame = dm.copy()
    dm_frame[:,:] = (114,100,76)
    dm[depth_frame > depth_threshold_max] = (114,100,76)
    dm[depth_frame < depth_threshold_min] = (114,100,76)
    dm_frame[topy1:bottomy1+1, topx1:bottomx1+1] = dm[topy1:bottomy1+1, topx1:bottomx1+1]
    dm_frame[topy2:bottomy2+1, topx2:bottomx2+1] = dm[topy2:bottomy2+1, topx2:bottomx2+1]
    # region 1
    cv2.rectangle(dm_frame, (topx1, topy1), (bottomx1, bottomy1), (0,255,0),2)
    # region 2
    cv2.rectangle(dm_frame, (topx2, topy2), (bottomx2, bottomy2), (0,255,0),2)

    # antenna position
    cv2.line(dm_frame, 
        (antenna_x_abs, 0),
        (antenna_x_abs, height),
        (0,255,0),
        2
    )

    # crop_dm = crop_dm[topy:bottomy+1, topx:bottomx+1, :].copy()
    dm_frame = cv2.flip(dm_frame, 1)
    # self.current_fps.display(dm_frame, orig=(50,20), color=(0,255,0), size=0.6)
    # cv2.imshow(frame_name, dm_frame)
    return dm_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', default=PARAMS['FPS'], type=int, help="Player rate")
    parser.add_argument('--res', default=PARAMS['DEPTH_RESOLUTION'], type=int, help="Depth Resolution used")
    parser.add_argument('--antenna', default=PARAMS['ANTENNA_ROI_FILENAME'], type=str, help="ROI of the Theremin antenna")
    parser.add_argument('--body', default=PARAMS['BODY_ROI_FILENAME'], type=str, help="ROI of body position")
    parser.add_argument('--proj', default=0, type=int, help='Use projections')
    parser.add_argument('--file', type=str, help="Capture File", required=True)
    args = parser.parse_args()

    print(team.banner)

    cam_res = PARAMS['DEPTH_CAMERA_RESOLUTIONS']
    cam_resolution = str(args.res)
    depth_res_w = cam_res[cam_resolution][1]
    depth_res_h = cam_res[cam_resolution][2]

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
        print("Body and Antenna ROIs not defined: Please run configuration")
        exit()

    # Read ROI from file
    filename = PARAMS['DATASET_PATH'] + "/" + PARAMS['DEPTH_ROI_FILENAME']
    ts = datetime.now().strftime("%Y-%d-%m %H:%M:%S")
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

    # Read capture file
    with open(args.file, "rb") as cap:
        dataset = pickle.load(cap)
        # frames, from, to
        if dataset:
            # antenna location x, z
            antenna_x_abs = rois['antenna']['absolute']['bottomx']
            antenna_z = rois['antenna']['z']
            antenna_x = ((antenna_x_abs - PARAMS['INTRINSICS_RIGHT_CX'])*antenna_z)/PARAMS['INTRINSICS_RIGHT_FX']             

            # depth limits
            depth_threshold_min = rois['antenna']['z'] + PARAMS['ANTENNA_BUFFER']
            depth_threshold_max = rois['body']['z'] - PARAMS['BODY_BUFFER']

            # define inline function to get coordinates in mm and px
            x_coordinate_mm = lambda x, z: ((x - PARAMS['INTRINSICS_RIGHT_CX'])*z)/PARAMS['INTRINSICS_RIGHT_FX']
            x_coordinate_px = lambda x: int(x * depth_res_w)
            y_coordinate_mm = lambda y, z: ((y - PARAMS['INTRINSICS_RIGHT_CY'])*z)/PARAMS['INTRINSICS_RIGHT_FY']            
            y_coordinate_px = lambda y: int(y * depth_res_h)

            # Fixed parameters right hand
            topx_rh = x_coordinate_px(rois['right_hand']['topx'])
            bottomx_rh = x_coordinate_px(rois['right_hand']['bottomx']) 
            topy_rh = y_coordinate_px(rois['right_hand']['topy'])
            bottomy_rh = y_coordinate_px(rois['right_hand']['bottomy'])           

            # Get xz limits (right hand)
            min_x_min_z_rh = x_coordinate_mm(topx_rh, depth_threshold_min)
            max_x_min_z_rh = x_coordinate_mm(bottomx_rh, depth_threshold_min)
            min_x_max_z_rh = x_coordinate_mm(topx_rh, depth_threshold_max)
            max_x_max_z_rh = x_coordinate_mm(bottomx_rh, depth_threshold_max) 
            # Fixed parameters left hand
            topx_lh = x_coordinate_px(rois['left_hand']['topx'])
            bottomx_lh = x_coordinate_px(rois['left_hand']['bottomx']) 
            topy_lh = y_coordinate_px(rois['left_hand']['topy'])
            bottomy_lh = y_coordinate_px(rois['left_hand']['bottomy'])
            # Get yz limits (left hand)
            min_y_min_z_lh = y_coordinate_mm(topy_lh, depth_threshold_min)
            max_y_min_z_lh = y_coordinate_mm(bottomy_lh, depth_threshold_min)
            min_y_max_z_lh = y_coordinate_mm(topy_lh, depth_threshold_max)
            max_y_max_z_lh = y_coordinate_mm(bottomy_lh, depth_threshold_max) 
            # Get diagonal vector and distance (we could try with antenna reference)
            diag_x = max_x_max_z_rh-antenna_x
            diag_z = depth_threshold_max-antenna_z
            diag_distance = np.sqrt(diag_x**2 + diag_z**2)
            diag_x_u = diag_x/diag_distance
            diag_z_u = diag_z/diag_distance
            vector_u = np.array([diag_x_u, diag_z_u])
            # function to calculate the vector projection
            def vector_projection(x, z):
                vector_p = np.array([x-antenna_x, z-antenna_z])
                p_proj_u = vector_u*(np.dot(vector_u, vector_p))
                proj_distance = np.linalg.norm(p_proj_u)
                p_proj_u_x = p_proj_u[0] + antenna_x
                p_proj_u_z = p_proj_u[1] + antenna_z
                return p_proj_u_x, p_proj_u_z, proj_distance
                        
            # Timestamps
            timestamps = []
            frame_data = {}
            frames = {}
            # process dataset 
            for datapoint in dataset:
                data = {
                    'left': None,
                    'right': None
                }
                milis = int(datapoint['timestamp'].timestamp() * 1000)
                timestamps.append(milis)
                frames[milis] = datapoint['frame']
                if datapoint['hand'] == 'left':
                    data['left'] = datapoint['depth_map']
                else:
                    data['right'] = datapoint['depth_map']
                frame_data[milis] = data

            # Global variables
            STATE = 1
            CURRENT_TS_IDX = 0
            CONTEXT = {
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

            # Show One Frame
            def show_frame():
                global STATE, CURRENT_TS_IDX
                final_frame = np.zeros((depth_res_h, depth_res_w*3, 3), np.uint8)
                if CURRENT_TS_IDX >= len(timestamps):
                    CURRENT_TS_IDX = 0
                ts = timestamps[CURRENT_TS_IDX]
                lh = None
                rh = None
                dm = np.zeros((depth_res_h, depth_res_w), np.uint16)
                lh = frame_data[ts]['left']
                if frame_data[ts]['left'] is None:
                    lh = CONTEXT['previous_lh']
                else:
                    CONTEXT['previous_lh'] = lh
                rh = frame_data[ts]['right']
                if frame_data[ts]['right'] is None:
                    rh = CONTEXT['previous_rh']
                else:
                    CONTEXT['previous_rh'] = rh
                
                if lh is not None:
                    dm[topy_lh:bottomy_lh+1, topx_lh:bottomx_lh+1] = lh

                if rh is not None:
                    dm[topy_rh:bottomy_rh+1, topx_rh:bottomx_rh+1] = rh
                
                # Center frame
                frame_ = show_depth_map_segmentation(
                        dm,
                        antenna_x_abs,
                        depth_res_h,
                        topx_rh, 
                        topy_rh,
                        bottomx_rh, 
                        bottomy_rh,
                        topx_lh, 
                        topy_lh, 
                        bottomx_lh, 
                        bottomy_lh
                    )

                img_xz = None
                img_yz = None

                # Plot xz
                pc_rh = utils.transform_xyz(
                    dm, topx_rh, bottomx_rh, topy_rh, bottomy_rh,
                    depth_threshold_min, depth_threshold_max,
                    PARAMS['INTRINSICS_RIGHT_CX'],
                    PARAMS['INTRINSICS_RIGHT_CY'],
                    PARAMS['INTRINSICS_RIGHT_FX'],
                    PARAMS['INTRINSICS_RIGHT_FY']                  
                )
                if pc_rh is not None:
                    # Calculates Centroid (x,y), ignore y
                    # and distance to 0,0 (where ever it is)
                    points_x = pc_rh[0]
                    points_z = pc_rh[2]
                    centroid_x, centroid_z, distance = utils.distance_(points_x, points_z, antenna_x, antenna_z)
                    # filter out outliers
                    centroid_f_x, centroid_f_z, distance_f = utils.distance_filter_out_(points_x, points_z, antenna_x, antenna_z)
                    # only fingers
                    fing_centroid_x, fing_centroid_z, distance_fing = utils.distance_filter_fingers_(points_x, points_z, antenna_x, antenna_z)

                    if args.proj:
                        if distance_fing is not None:
                            fing_centroid_x, fing_centroid_z, distance_fing = vector_projection(fing_centroid_x, fing_centroid_z)
                            
                    # Show visualization xz
                    if CONTEXT['init_xz']:
                        if CONTEXT['fig_xz'] is not None:
                            CONTEXT['fig_xz'], CONTEXT['ax_xz'], CONTEXT['plot_xz'], CONTEXT['centroid_plot_xz'], CONTEXT['centroid_plot_xz_f'], CONTEXT['centroid_plot_xz_fingers'] = utils.plot_xz(
                                points_x,
                                points_z,
                                centroid_x,
                                centroid_z,
                                centroid_f_x,
                                centroid_f_z,
                                fing_centroid_x,
                                fing_centroid_z,
                                CONTEXT['fig_xz'], 
                                CONTEXT['ax_xz'], CONTEXT['plot_xz'], 
                                CONTEXT['centroid_plot_xz'], 
                                CONTEXT['centroid_plot_xz_f'], 
                                CONTEXT['centroid_plot_xz_fingers']
                            )
                    else:
                        CONTEXT['fig_xz'], CONTEXT['ax_xz'], CONTEXT['plot_xz'], CONTEXT['centroid_plot_xz'], CONTEXT['centroid_plot_xz_f'], CONTEXT['centroid_plot_xz_fingers'] = utils.init_plot_xz(
                                depth_res_w,
                                depth_res_h,
                                points_x, points_z,
                                centroid_x,
                                centroid_z,
                                centroid_f_x,
                                centroid_f_z,
                                fing_centroid_x,
                                fing_centroid_z,
                                depth_threshold_min,
                                depth_threshold_max,
                                antenna_x, antenna_z,
                                min_x_min_z_rh, max_x_min_z_rh, 
                                min_x_max_z_rh, max_x_max_z_rh
                        )
                        CONTEXT['init_xz'] = True

                    if CONTEXT['fig_xz'] is not None:
                        img_xz = np.frombuffer(CONTEXT['fig_xz'].canvas.tostring_rgb(), dtype=np.uint8)
                        img_xz  = img_xz.reshape(CONTEXT['fig_xz'].canvas.get_width_height()[::-1] + (3,))
                        img_xz = cv2.cvtColor(img_xz, cv2.COLOR_RGB2BGR)
                        img_xz = cv2.putText(img_xz, f"Distance 1 = {distance}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        img_xz = cv2.putText(img_xz, f"Distance 2 = {distance_f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        img_xz = cv2.putText(img_xz, f"Distance 3 = {distance_fing}", (320,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                
                # Plot yz
                pc_lh = utils.transform_xyz(
                    dm, topx_lh, bottomx_lh, topy_lh, bottomy_lh,
                    depth_threshold_min, depth_threshold_max,
                    PARAMS['INTRINSICS_RIGHT_CX'],
                    PARAMS['INTRINSICS_RIGHT_CY'],
                    PARAMS['INTRINSICS_RIGHT_FX'],
                    PARAMS['INTRINSICS_RIGHT_FY']                  
                )
                if pc_lh is not None:
                    # Calculates Centroid (x,y), ignore y
                    points_y = pc_rh[1]
                    points_z = pc_rh[2]
                    centroid_y = np.mean(points_y)
                    centroid_z = np.mean(points_z)
                    
                    # Show visualization xz
                    if CONTEXT['init_yz']:
                        if CONTEXT['fig_yz'] is not None:
                            CONTEXT['fig_yz'], CONTEXT['ax_yz'], CONTEXT['plot_yz'], CONTEXT['centroid_plot_yz'] = utils.plot_yz(
                                points_y,
                                points_z,
                                centroid_y,
                                centroid_z,
                                CONTEXT['fig_yz'], 
                                CONTEXT['ax_yz'], 
                                CONTEXT['plot_yz'], 
                                CONTEXT['centroid_plot_yz']
                            )
                    else:
                        CONTEXT['fig_yz'], CONTEXT['ax_yz'], CONTEXT['plot_yz'], CONTEXT['centroid_plot_yz'] = utils.init_plot_yz(
                                depth_res_w,
                                depth_res_h,
                                points_y, points_z,
                                centroid_y,
                                centroid_z,
                                depth_threshold_min,
                                depth_threshold_max,
                                min_y_min_z_lh, max_y_min_z_lh, 
                                min_y_max_z_lh, max_y_max_z_lh
                        )
                        CONTEXT['init_yz'] = True

                    if CONTEXT['fig_yz'] is not None:
                        img_yz = np.frombuffer(CONTEXT['fig_yz'].canvas.tostring_rgb(), dtype=np.uint8)
                        img_yz  = img_yz.reshape(CONTEXT['fig_yz'].canvas.get_width_height()[::-1] + (3,))
                        img_yz = cv2.cvtColor(img_yz, cv2.COLOR_RGB2BGR)

                final_frame[:,depth_res_w:depth_res_w*2] = frame_
                if img_xz is not None:
                    final_frame[:,depth_res_w*2:] = img_xz
                if img_yz is not None:
                    final_frame[:,0:depth_res_w] = img_yz   

                # Pillow expect BGR
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                # Next Frame
                CURRENT_TS_IDX += 1

                # Show Control
                img = Image.fromarray(final_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.after(10, show_frame) 

                        

            # Window
            window = tk.Tk()  #Makes main window
            window.wm_title("Player")
            window.config(background="#FFFFFF")
            # Graphics window
            img_frame = tk.Frame(window, width=depth_res_w*3+20, height=depth_res_h+10)
            img_frame.grid(row=0, column=0, padx=10, pady=2)
            lmain = tk.Label(img_frame)
            lmain.grid(row=0, column=0)

            # First
            show_frame()
            # Tk main loop 
            window.mainloop()


            # Loop
            #ts_idx = 0
            #previous_lh = None
            #previous_rh = None
            # Matplotlib plot
            #init_xz = False
            #fig_xz = ax_xz = plot_xz = centroid_plot_xz = centroid_plot_xz_f = centroid_plot_xz_fingers = None
            #init_yz = False
            #fig_yz = ax_yz = plot_yz = centroid_plot_yz = None

            # while True:
            #     final_frame = np.zeros((depth_res_h, depth_res_w*3, 3), np.uint8)
            #     if ts_idx >= len(timestamps):
            #         ts_idx = 0
            #     ts = timestamps[ts_idx]
            #     lh = None
            #     rh = None
            #     dm = np.zeros((depth_res_h, depth_res_w), np.uint16)
            #     lh = frame_data[ts]['left']
            #     if frame_data[ts]['left'] is None:
            #         lh = previous_lh
            #     else:
            #         previous_lh = lh
            #     rh = frame_data[ts]['right']
            #     if frame_data[ts]['right'] is None:
            #         rh = previous_rh
            #     else:
            #         previous_rh = rh
                
            #     if lh is not None:
            #         dm[topy_lh:bottomy_lh+1, topx_lh:bottomx_lh+1] = lh

            #     if rh is not None:
            #         dm[topy_rh:bottomy_rh+1, topx_rh:bottomx_rh+1] = rh

            #     frame_ = show_depth_map_segmentation(
            #             dm,
            #             antenna_x_abs,
            #             depth_res_h,
            #             topx_rh, 
            #             topy_rh,
            #             bottomx_rh, 
            #             bottomy_rh,
            #             topx_lh, 
            #             topy_lh, 
            #             bottomx_lh, 
            #             bottomy_lh
            #         )

            #     img_xz = None
            #     img_yz = None

            #     # Plot xz
            #     pc_rh = utils.transform_xyz(
            #         dm, topx_rh, bottomx_rh, topy_rh, bottomy_rh,
            #         depth_threshold_min, depth_threshold_max,
            #         PARAMS['INTRINSICS_RIGHT_CX'],
            #         PARAMS['INTRINSICS_RIGHT_CY'],
            #         PARAMS['INTRINSICS_RIGHT_FX'],
            #         PARAMS['INTRINSICS_RIGHT_FY']                  
            #     )
            #     if pc_rh is not None:
            #         # Calculates Centroid (x,y), ignore y
            #         # and distance to 0,0 (where ever it is)
            #         points_x = pc_rh[0]
            #         points_z = pc_rh[2]
            #         centroid_x, centroid_z, distance = utils.distance_(points_x, points_z, antenna_x, antenna_z)
            #         # filter out outliers
            #         centroid_f_x, centroid_f_z, distance_f = utils.distance_filter_out_(points_x, points_z, antenna_x, antenna_z)
            #         # only fingers
            #         fing_centroid_x, fing_centroid_z, distance_fing = utils.distance_filter_fingers_(points_x, points_z, antenna_x, antenna_z)

            #         #q1x, q3x = np.quantile(points_x, [0.25, 0.75])
            #         #q1z, q3z = np.quantile(points_z, [0.25, 0.75])
            #         #iqrx = q3x - q1x
            #         #iqrz = q3z - q1z
            #         # limits to determine outliers
            #         #liminfx = q1x - 1.5*iqrx
            #         #limsupx = q3x + 1.5*iqrx
            #         #liminfz = q1z - 1.5*iqrz
            #         #limsupz = q3z + 1.5*iqrz
            #         # filter outliers (only include points between liminf and limsup)
            #         #points_xz = np.vstack([points_x, points_z])
            #         #cond_x = (points_xz[0,:] >= liminfx) & (points_xz[0,:] <= limsupx)
            #         #cond_z = (points_xz[1,:] >= liminfz) & (points_xz[1,:] <= limsupz)
            #         #filterd_xz = points_xz[:,cond_x | cond_z]
            #         # further filter the 
            #         #cond_z_fingers = (filterd_xz[1,:] <= centroid_z)
            #         #fingers_xz = filterd_xz[:,cond_z_fingers]
            #         #fing_centroid_x = np.mean(fingers_xz[0,:])
            #         #fing_centroid_z = np.mean(fingers_xz[1,:])
            #         #distance_fing = np.sqrt((fing_centroid_x-antenna_x)**2 + (fing_centroid_z-antenna_z)**2)
            #         # projection (todo)


            #         # Show visualization xz
            #         if init_xz:
            #             if fig_xz is not None:
            #                 fig_xz, ax_xz, plot_xz, centroid_plot_xz, centroid_plot_xz_f, centroid_plot_xz_fingers = utils.plot_xz(
            #                     points_x,
            #                     points_z,
            #                     centroid_x,
            #                     centroid_z,
            #                     centroid_f_x,
            #                     centroid_f_z,
            #                     fing_centroid_x,
            #                     fing_centroid_z,
            #                     fig_xz,
            #                     ax_xz,
            #                     plot_xz,
            #                     centroid_plot_xz,
            #                     centroid_plot_xz_f,
            #                     centroid_plot_xz_fingers
            #                 )
            #         else:
            #             fig_xz, ax_xz, plot_xz, centroid_plot_xz, centroid_plot_xz_f, centroid_plot_xz_fingers = utils.init_plot_xz(
            #                     depth_res_w,
            #                     depth_res_h,
            #                     points_x, points_z,
            #                     centroid_x,
            #                     centroid_z,
            #                     centroid_f_x,
            #                     centroid_f_z,
            #                     fing_centroid_x,
            #                     fing_centroid_z,
            #                     depth_threshold_min,
            #                     depth_threshold_max,
            #                     antenna_x, antenna_z,
            #                     min_x_min_z_rh, max_x_min_z_rh, 
            #                     min_x_max_z_rh, max_x_max_z_rh
            #             )
            #             init_xz = True

            #         if fig_xz is not None:
            #             img_xz = np.frombuffer(fig_xz.canvas.tostring_rgb(), dtype=np.uint8)
            #             img_xz  = img_xz.reshape(fig_xz.canvas.get_width_height()[::-1] + (3,))
            #             img_xz = cv2.cvtColor(img_xz, cv2.COLOR_RGB2BGR)
            #             img_xz = cv2.putText(img_xz, f"Distance 1 = {distance}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            #             img_xz = cv2.putText(img_xz, f"Distance 2 = {distance_f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            #             img_xz = cv2.putText(img_xz, f"Distance 3 = {distance_fing}", (320,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        
            #     # Plot yz
            #     pc_lh = utils.transform_xyz(
            #         dm, topx_lh, bottomx_lh, topy_lh, bottomy_lh,
            #         depth_threshold_min, depth_threshold_max,
            #         PARAMS['INTRINSICS_RIGHT_CX'],
            #         PARAMS['INTRINSICS_RIGHT_CY'],
            #         PARAMS['INTRINSICS_RIGHT_FX'],
            #         PARAMS['INTRINSICS_RIGHT_FY']                  
            #     )
            #     if pc_lh is not None:
            #         # Calculates Centroid (x,y), ignore y
            #         points_y = pc_rh[1]
            #         points_z = pc_rh[2]
            #         centroid_y = np.mean(points_y)
            #         centroid_z = np.mean(points_z)
                    
            #         # Show visualization xz
            #         if init_yz:
            #             if fig_yz is not None:
            #                 fig_yz, ax_yz, plot_yz, centroid_plot_yz = utils.plot_yz(
            #                     points_y,
            #                     points_z,
            #                     centroid_y,
            #                     centroid_z,
            #                     fig_yz,
            #                     ax_yz,
            #                     plot_yz,
            #                     centroid_plot_yz
            #                 )
            #         else:
            #             fig_yz, ax_yz, plot_yz, centroid_plot_yz = utils.init_plot_yz(
            #                     depth_res_w,
            #                     depth_res_h,
            #                     points_y, points_z,
            #                     centroid_y,
            #                     centroid_z,
            #                     depth_threshold_min,
            #                     depth_threshold_max,
            #                     min_y_min_z_lh, max_y_min_z_lh, 
            #                     min_y_max_z_lh, max_y_max_z_lh
            #             )
            #             init_yz = True

            #         if fig_yz is not None:
            #             img_yz = np.frombuffer(fig_yz.canvas.tostring_rgb(), dtype=np.uint8)
            #             img_yz  = img_yz.reshape(fig_yz.canvas.get_width_height()[::-1] + (3,))
            #             img_yz = cv2.cvtColor(img_yz, cv2.COLOR_RGB2BGR)

            #     final_frame[:,depth_res_w:depth_res_w*2] = frame_
            #     if img_xz is not None:
            #         final_frame[:,depth_res_w*2:] = img_xz
            #     if img_yz is not None:
            #         final_frame[:,0:depth_res_w] = img_yz
                
            #     cv2.imshow("Data Capture Player", final_frame)
            #     time.sleep(0.001)

            #     key = cv2.waitKey(1) 
            #     if key == ord('q') or key == 27:
            #         # quit
            #         break
            #     ts_idx += 1

            # cv2.destroyAllWindows()






