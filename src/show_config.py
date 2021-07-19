import depthai as dai
from team import banner
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import config

# Camera options = 400, 720, 800 for depth
MONO_CAM_RES = {
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

if __name__ == "__main__":
    print(banner)

    # Camera Settings
    print("** Camera Intrinsics (Mono Right) **")
    print(f"CX = {config.PARAMS['INTRINSICS_RIGHT_CX']}")
    print(f"FX = {config.PARAMS['INTRINSICS_RIGHT_FX']}")
    print(f"CY = {config.PARAMS['INTRINSICS_RIGHT_CY']}")
    print(f"FY = {config.PARAMS['INTRINSICS_RIGHT_FY']}")

    print()
    # Resolution used
    print("** Resolution used **")
    print(f"Resolution = {config.PARAMS['DEPTH_RESOLUTION']}")
    print(MONO_CAM_RES[config.PARAMS['DEPTH_RESOLUTION']])
    res_w = MONO_CAM_RES[config.PARAMS['DEPTH_RESOLUTION']][1]
    res_h = MONO_CAM_RES[config.PARAMS['DEPTH_RESOLUTION']][2]
    print(f"Width = {res_w}")
    print(f"Height = {res_h}")

    print()
    # ROI files
    print("** ROI Files **")
    print(f"Path: {config.PARAMS['DATASET_PATH']}")
    print(f"Body ROI file: {config.PARAMS['BODY_ROI_FILENAME']}")
    ok_body = ok_antenna = ok_depth = False
    if os.path.isfile(f"{config.PARAMS['DATASET_PATH']}/{config.PARAMS['BODY_ROI_FILENAME']}"):
        print("[Ok]")
        ok_body = True
    else:
        print("[File not found]")
    print(f"Antenna ROI file: {config.PARAMS['ANTENNA_ROI_FILENAME']}")
    if os.path.isfile(f"{config.PARAMS['DATASET_PATH']}/{config.PARAMS['ANTENNA_ROI_FILENAME']}"):
        print("[Ok]")
        ok_antenna = True
    else:
        print("[File not found]")
    print(f"Depth ROI file: {config.PARAMS['DEPTH_ROI_FILENAME']}")
    if os.path.isfile(f"{config.PARAMS['DATASET_PATH']}/{config.PARAMS['DEPTH_ROI_FILENAME']}"):
        print("[Ok]")
        ok_depth = True
    else:
        print("[File not found]")

    print()
    # Show ROIs
    print("** ROIs **")
    print()
    print("Body ROI:")
    if ok_body:
        with open(f"{config.PARAMS['DATASET_PATH']}/{config.PARAMS['BODY_ROI_FILENAME']}", "rb") as f1:
            roi_body = pickle.load(f1)
            for k, v in roi_body.items(): print(f"{k}: {v}")
    print()
    print("Antenna ROI:")
    if ok_antenna:
        with open(f"{config.PARAMS['DATASET_PATH']}/{config.PARAMS['ANTENNA_ROI_FILENAME']}", "rb") as f2:
            roi_antenna = pickle.load(f2)
            for k, v in roi_antenna.items(): print(f"{k}: {v}")
    print()
    print("Depth ROI:")
    if ok_depth:
        with open(f"{config.PARAMS['DATASET_PATH']}/{config.PARAMS['DEPTH_ROI_FILENAME']}", "rb") as f3:
            roi_depth = pickle.load(f3)
            for k, v in roi_depth.items(): print(f"{k}: {v}")

    print()
    # Limits
    print("** Limits **")
    if ok_body and ok_antenna:
        # antenna x
        antenna_x = ((roi_antenna['absolute']['bottomx'] - config.PARAMS['INTRINSICS_RIGHT_CX'])*roi_antenna['z'])/config.PARAMS['INTRINSICS_RIGHT_FX']
        print(f"X_antena = {antenna_x} mm")
        print(f"Z_antenna = {roi_antenna['z']} mm")
        print(f"Antenna_buffer = {config.PARAMS['ANTENNA_BUFFER']} mm")
        print(f"Z_min = Z_antenna + Anntena_buffer = {roi_antenna['z'] + config.PARAMS['ANTENNA_BUFFER']} mm")
        print(f"Z_body = {roi_body['z']} mm")
        print(f"Boddy_buffer = {config.PARAMS['BODY_BUFFER']} mm")
        print(f"Z_max = Z_body - Body_buffer = {roi_body['z'] - config.PARAMS['BODY_BUFFER']} mm")
       
        print()
        if ok_depth:
            topx_rh = int(roi_depth['right_hand']['topx'] * res_w)
            bottomx_rh = int(roi_depth['right_hand']['bottomx'] * res_w)
            topy_rh = int(roi_depth['right_hand']['topy'] * res_h)
            bottomy_rh = int(roi_depth['right_hand']['bottomy'] * res_h)

            topx_lh = int(roi_depth['left_hand']['topx'] * res_w)
            bottomx_lh = int(roi_depth['left_hand']['bottomx'] * res_w)
            topy_lh = int(roi_depth['left_hand']['topy'] * res_h)
            bottomy_lh = int(roi_depth['left_hand']['bottomy'] * res_h)

            print("-- Right Hand:")
            print(f"topx_rh = {topx_rh} px")
            print(f"topy_rh = {topy_rh} px")
            print(f"bottomx_rh = {bottomx_rh} px")
            print(f"bottomy_rh = {bottomy_rh} px")
            print("-- Left Hand:")
            print(f"topx_lh = {topx_lh} px")
            print(f"topy_lh = {topy_lh} px")
            print(f"bottomx_lh = {bottomx_lh} px")
            print(f"bottomy_lh = {bottomy_lh} px")

            zmin = roi_antenna['z'] + config.PARAMS['ANTENNA_BUFFER']
            zmax = roi_body['z'] - config.PARAMS['BODY_BUFFER']
            min_x_min_z_rh = ((topx_rh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmin)/config.PARAMS['INTRINSICS_RIGHT_FX']
            max_x_min_z_rh = ((bottomx_rh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmin)/config.PARAMS['INTRINSICS_RIGHT_FX']
            min_x_max_z_rh = ((topx_rh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmax)/config.PARAMS['INTRINSICS_RIGHT_FX']
            max_x_max_z_rh = ((bottomx_rh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmax)/config.PARAMS['INTRINSICS_RIGHT_FX']
            min_x_min_z_lh = ((topx_lh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmin)/config.PARAMS['INTRINSICS_RIGHT_FX']
            max_x_min_z_lh = ((bottomx_lh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmin)/config.PARAMS['INTRINSICS_RIGHT_FX']
            min_x_max_z_lh = ((topx_lh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmax)/config.PARAMS['INTRINSICS_RIGHT_FX']
            max_x_max_z_lh = ((bottomx_lh - config.PARAMS['INTRINSICS_RIGHT_CX'])*zmax)/config.PARAMS['INTRINSICS_RIGHT_FX']

            print("-- Limiting Region (Right Hand:)")
            print(f"X_min_Z_min_rh = {min_x_min_z_rh} mm")
            print(f"X_max_Z_min_rh = {max_x_min_z_rh} mm")
            print(f"X_min_Z_max_rh = {min_x_max_z_rh} mm")
            print(f"X_max_Z_max_rh = {max_x_max_z_rh} mm")   
            print("-- Limiting Region (Left Hand:)")
            print(f"X_min_Z_min_lh = {min_x_min_z_lh} mm")
            print(f"X_max_Z_min_lh = {max_x_min_z_lh} mm")
            print(f"X_min_Z_max_lh = {min_x_max_z_lh} mm")
            print(f"X_max_Z_max_lh = {max_x_max_z_lh} mm")   
            
            # Draw Region for Right Hand
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            # set limits
            ax1.set_xlim((min(antenna_x, min_x_min_z_rh, min_x_max_z_rh) - 50, max(max_x_max_z_rh, max_x_min_z_rh) + 50))
            # ax1.set_xlim((min_x_max_z_rh - 50, max_x_max_z_rh + 50))
            ax1.set_ylim((roi_antenna['z'] - 100, zmax + 100))
            # antenna location
            ax1.plot(antenna_x, roi_antenna['z'], marker='X', color='b', ms=8)
            ax1.text(antenna_x + 20, roi_antenna['z'] - 10, 'ANTENNA', color='b', fontsize='small')
            # draw limiting region
            x0, x1 = ax1.get_xlim()
            ax1.axline((min_x_min_z_rh, zmin), (max_x_min_z_rh, zmin), ls='dashed', color='r', linewidth=1.2)
            ax1.axline((min_x_max_z_rh, zmax), (max_x_max_z_rh, zmax), ls='dashed', color='r', linewidth=1.2)
            ax1.axline((min_x_min_z_rh, zmin), (min_x_max_z_rh, zmax), ls='dashed', color='r', linewidth=1.2)
            ax1.axline((max_x_min_z_rh, zmin), (max_x_max_z_rh, zmax), ls='dashed', color='r', linewidth=1.2)
            ax1.axline((x0, roi_antenna['z']), (x1, roi_antenna['z']), ls='dashed', color='k', linewidth=0.8)
            ax1.axline((x0, roi_body['z']), (x1, roi_body['z']), ls='dashed', color='k', linewidth=0.8)
            
            ax1.set_xlabel("X")
            ax1.set_ylabel("Z")
            fig1.canvas.draw()

            # Draw Region for Left Hand
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            # set limits
            ax2.set_xlim((min(antenna_x, min_x_min_z_lh, min_x_max_z_lh) - 50, max(max_x_max_z_lh, max_x_min_z_lh) + 50))
            ax2.set_ylim((roi_antenna['z'] - 100, zmax + 100))
            # antenna location
            ax2.plot(antenna_x, roi_antenna['z'], marker='X', color='b', ms=8)
            ax2.text(antenna_x + 20, roi_antenna['z'] - 10, 'ANTENNA', color='b', fontsize='small')
            # draw limiting region
            x0, x1 = ax2.get_xlim()
            ax2.axline((min_x_min_z_lh, zmin), (max_x_min_z_lh, zmin), ls='dashed', color='r', linewidth=1.2)
            ax2.axline((min_x_max_z_lh, zmax), (max_x_max_z_lh, zmax), ls='dashed', color='r', linewidth=1.2)
            ax2.axline((min_x_min_z_lh, zmin), (min_x_max_z_lh, zmax), ls='dashed', color='r', linewidth=1.2)
            ax2.axline((max_x_min_z_lh, zmin), (max_x_max_z_lh, zmax), ls='dashed', color='r', linewidth=1.2)
            ax2.axline((x0, roi_antenna['z']), (x1, roi_antenna['z']), ls='dashed', color='k', linewidth=0.8)
            ax2.axline((x0, roi_body['z']), (x1, roi_body['z']), ls='dashed', color='k', linewidth=0.8)
            
            ax2.set_xlabel("X")
            ax2.set_ylabel("Z")
            fig2.canvas.draw()

            rigth_hand = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
            rigth_hand  = rigth_hand.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            rigth_hand = cv2.cvtColor(rigth_hand, cv2.COLOR_RGB2BGR)

            left_hand = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
            left_hand  = left_hand.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
            left_hand = cv2.cvtColor(left_hand, cv2.COLOR_RGB2BGR)

            cv2.imshow("Right Hand Region", rigth_hand)
            cv2.imshow("Left Hand Region", left_hand)
            cv2.waitKey()
            cv2.destroyAllWindows()