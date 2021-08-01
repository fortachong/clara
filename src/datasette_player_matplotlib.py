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

# Matplotlib plot Initialization for Right hand xz plot
def init_plot_xz(
        width, height,
        x, z,
        centroid_x,
        centroid_z,
        centroid_f_x,
        centroid_f_z,
        centroid_fing_x,
        centroid_fing_z,
        min_z, max_z, 
        antenna_x, antenna_z, 
        min_x_min_z, max_x_min_z, 
        min_x_max_z, max_x_max_z,
        notes=None
    ):
    c_f_x = centroid_f_x
    if centroid_f_x is None:
        c_f_x = centroid_x
    c_f_z = centroid_f_z
    if centroid_f_z is None:
        c_f_z = centroid_z

    c_fing_x = centroid_fing_x
    if centroid_fing_x is None:
        c_f_x = centroid_x
    c_fing_z = centroid_fing_z
    if centroid_fing_z is None:
        c_fing_z = centroid_z        

    color_bg = '#2C2E43'
    color_region = '#595260'
    color_diag = '#FFD523'
    color_points = '#B2B1B9'
    color_antenna = '#FFD523'
    plt.rcParams['figure.facecolor'] = color_bg
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(width*px, height*px))
    ax.set(facecolor=color_bg)
    ax.axis('off')
    # set limits
    ax.set_xlim((antenna_x - 120, max_x_min_z + 120))
    ax.set_ylim((min_z - 50, max_z + 50))
    # antenna location
    ax.plot(antenna_x, antenna_z, marker='X', color=color_antenna, ms=8)
    ax.text(antenna_x + 5, antenna_z - 8, 'ANTENNA', color=color_antenna, fontsize='large')
    # draw limiting region
    x0, x1 = ax.get_xlim()
    ax.axline((x0, min_z), (x1, min_z), ls='dashed', color=color_region, linewidth=1.2)
    ax.axline((x0, max_z), (x1, max_z), ls='dashed', color=color_region, linewidth=1.2)
    ax.axline((antenna_x, min_z), (min_x_max_z, max_z), ls='dashed', color=color_region, linewidth=1.2)
    ax.axline((antenna_x, antenna_z), (max_x_max_z, max_z), ls='dashed', color=color_diag, linewidth=0.5)
    ax.axline((max_x_min_z, min_z), (max_x_max_z, max_z), ls='dashed', color=color_region, linewidth=1.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    if notes is not None:
        ax.scatter(notes[0], notes[1], marker='o', color=color_diag, s=80, alpha=0.60)
      
    # Random initial plot (will be update every frame)
    plot = ax.scatter(x, z, marker='.', color=color_points, s=90, alpha=0.10)
    centroid_plot = ax.scatter(centroid_x, centroid_z, marker='X', color='r', s=20)
    centroid_f_plot = ax.scatter(c_f_x, c_f_z, marker='X', color='g', s=20)
    centroid_fing_plot = ax.scatter(c_fing_x, c_fing_z, marker='X', color='b', s=20)
    # axis
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    # draw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot, centroid_f_plot, centroid_fing_plot


# Plot xz (Right Hand)
def plot_xz(
        x,
        z,
        centroid_x,
        centroid_z,
        centroid_f_x,
        centroid_f_z,
        centroid_fing_x,
        centroid_fing_z,
        fig,
        ax,
        plot,
        centroid_plot,
        centroid_f_plot,
        centroid_fing_plot
    ):
    plot.set_offsets(np.stack([x,z], axis=1))
    centroid_plot.set_offsets(np.array([centroid_x,centroid_z]))
    centroid_f_plot.set_offsets(np.array([centroid_f_x,centroid_f_z]))
    centroid_fing_plot.set_offsets(np.array([centroid_fing_x,centroid_fing_z]))
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, plot, centroid_plot, centroid_f_plot, centroid_fing_plot

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

            # 3 octaves                
            octs = 3
            ht = 3*12
            nvectors = np.linspace(0, 1, ht) * diag_distance
            notes_ = np.tile(nvectors, 2).reshape((2, -1))
            #print(notes_.shape)
            #print(notes_)
            us = np.broadcast_to(vector_u, (ht, 2))
            #print(us.shape)
            #print(us)
            notes = us.T*notes_ + np.broadcast_to(np.array([antenna_x, antenna_z]), (ht, 2)).T
            #print(notes)
                        
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
                final_frame = np.zeros((depth_res_h*2, depth_res_w*2, 3), np.uint8)
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
                
                img_xz = None
                
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
                            CONTEXT['fig_xz'], CONTEXT['ax_xz'], CONTEXT['plot_xz'], CONTEXT['centroid_plot_xz'], CONTEXT['centroid_plot_xz_f'], CONTEXT['centroid_plot_xz_fingers'] = plot_xz(
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
                        CONTEXT['fig_xz'], CONTEXT['ax_xz'], CONTEXT['plot_xz'], CONTEXT['centroid_plot_xz'], CONTEXT['centroid_plot_xz_f'], CONTEXT['centroid_plot_xz_fingers'] = init_plot_xz(
                                depth_res_w*2,
                                depth_res_h*2,
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
                                min_x_max_z_rh, max_x_max_z_rh,
                                notes
                        )
                        CONTEXT['init_xz'] = True

                    if CONTEXT['fig_xz'] is not None:
                        img_xz = np.frombuffer(CONTEXT['fig_xz'].canvas.tostring_rgb(), dtype=np.uint8)
                        img_xz  = img_xz.reshape(CONTEXT['fig_xz'].canvas.get_width_height()[::-1] + (3,))
                        img_xz = cv2.cvtColor(img_xz, cv2.COLOR_RGB2BGR)
                        img_xz = cv2.putText(img_xz, f"Distance 1 = {distance}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        img_xz = cv2.putText(img_xz, f"Distance 2 = {distance_f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        img_xz = cv2.putText(img_xz, f"Distance 3 = {distance_fing}", (320,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                
                if img_xz is not None:
                    final_frame = img_xz.copy()
                
                # Pillow expect BGR
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                # Next Frame
                CURRENT_TS_IDX += 1

                # Show Control
                img = Image.fromarray(final_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.after(5, show_frame) 

            # Window
            window = tk.Tk()  #Makes main window
            window.wm_title("Player")
            window.config(background="#FFFFFF")
            # Graphics window
            img_frame = tk.Frame(window, width=depth_res_w*2+10, height=depth_res_h*2+10)
            img_frame.grid(row=0, column=0, padx=0, pady=0)
            lmain = tk.Label(img_frame)
            lmain.grid(row=0, column=0)

            # First
            show_frame()
            # Tk main loop 
            window.mainloop()




