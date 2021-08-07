#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Datasetter: It is like a casette recorder for data training
# A tool for capturing left hand gestures according to the method for
# Theremine playing.
# We follow this method:
# Show Note from a scale (predefined)
# Find the note on theremin
# Press S to save data
# Alternatively (still not implemented)

import cv2
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import threading
import queue as Queue
import argparse
import pickle
import os


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
    #'PALM_DETECTION_MODEL_PATH': "models/palm_detection_6_shaves.blob",
    'PALM_THRESHOLD': 0.8,
    'PALM_NMS_THRESHOLD': 0.5,
    'PALM_DETECTION_INPUT_LENGTH': 128,
    'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
    #'LM_DETECTION_MODEL_PATH': "models/hand_landmark_6_shaves.blob",
    'LM_THRESHOLD': 0.6,
    'LM_INPUT_LENGTH': 224,
    'FPS': 10,
    'ROI_DP_LOWER_TH': 100,
    'ROI_DP_UPPER_TH': 10000,
    'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
    'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6),
    'PREVIEW_WIDTH': 640,
    'PREVIEW_HEIGHT': 400,
    'HAND_BUFFER_PIXELS': 20, # buffer pixels for capturing the right hand
    'HAND_SIZE': 400, # Max size of the hand image
    'DATASET_PATH': 'data/positions/hands/training'
} 

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST)
    return resized.transpose(2,0,1)

# Hand tracker.
# Captures the hand gesture for left hand
class DatasetteHandCapture:
    def __init__(
        self,
        pd_path='',
        pd_score_thresh=0.5,
        pd_nms_thresh=0.3,
        lm_path='',
        lm_score_threshold=0.5,
        fps=20,
        preview_width=640,
        preview_height=400,
        hand_buffer_pixels=20,
        hand_size=400,
        label=''
    ):
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.fps = fps
        self.hand_buffer_pixels = hand_buffer_pixels
        self.hand_size = hand_size
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
        print(f"{self.nb_anchors} anchors have been created")
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
        
        # Region for depth
        self.depth_topx = 1
        self.depth_bottomx = 0
        self.depth_topy = 1
        self.depth_bottomy = 0

        # Data captured:
        self.capture = []
        self.label = label

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

    # Transform xy coordinates to normalized xy
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
                    cv2.circle(frame, (x, y), 3, (0,128,255), -1)

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

    # Save Depth Region Message
    def save_depth_message(self):
        message = {
            'DEPTH': 1,
            'roi': {
                'topx': self.depth_topx,
                'bottomx': self.depth_bottomx,
                'topy': self.depth_topy,
                'bottomy': self.detph_bottomy
            }
        }
        return message

    # Pipeline
    def create_pipeline(self):
        print("[{}]: Creating Pipeline...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        # Pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)
        print("[{}]: Setting up Depth...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        # Color Camera
        print("[{}]: Color Camera...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setFps(self.fps)
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.preview.link(cam_out.input)

        # Palm Detector
        print("[{}]: Mediapipe Palm Detector NN...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
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
        print("[{}]: Mediapipe Hand Landmark NN...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        lm_nn.setNumInferenceThreads(2)
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)           
        
        print("[{}]: Pipeline Created...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )        
        return pipeline        

    def run(self):
        pipeline = self.create_pipeline()
        with dai.Device(pipeline) as device:
            #pn = device.startPipeline()
            #print("Pipeline Started: {}".format(pn))
            
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
                datasette_frame_video = video_frame.copy()
                datasette_frame = video_frame.copy()
                datasette_hand = np.zeros((self.hand_size,self.hand_size,3), np.uint8)

                # flag
                capture_flag = False
                temp_capture = []
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
                        # Get rigth hand square enclosure
                        if region.handedness >= 0.85:
                            self.lm_transform(region)
                            # Save min and max regions
                            dtx_candidate = np.min(region.lm_xy_y_rescaled[:, 0])
                            dbx_candidate = np.max(region.lm_xy_y_rescaled[:, 0])
                            dty_candidate = np.min(region.lm_xy_y_rescaled[:, 1])
                            dby_candidate = np.max(region.lm_xy_y_rescaled[:, 1])
                            if self.depth_topx > dtx_candidate:
                                self.depth_topx = dtx_candidate
                            if self.depth_bottomx < dbx_candidate:
                                self.depth_bottomx = dbx_candidate
                            if self.depth_topy > dty_candidate:
                                self.depth_topy = dty_candidate
                            if self.depth_bottomy < dby_candidate:
                                self.depth_bottomy = dby_candidate                                

                            # Get coordinates for bounding box
                            # region.lm_xy is a list of x,y pixel values
                            # print(region.lm_xy)
                            xs = region.lm_xy[:, 0]
                            ys = region.lm_xy[:, 1]

                            len_xs = np.max(xs) - np.min(xs)
                            len_ys = np.max(ys) - np.min(ys)
                            dif_xs_ys = abs(len_xs-len_ys)
                            # square crop with a buffer for training
                            if len_xs > len_ys:
                                by = dif_xs_ys // 2
                                topx, bottomx = np.min(xs), np.max(xs)
                                topy, bottomy = np.min(ys)-by, np.max(ys)+by
                                topx_b, bottomx_b = topx-self.hand_buffer_pixels, bottomx+self.hand_buffer_pixels
                                topy_b, bottomy_b = topy-self.hand_buffer_pixels, bottomy+self.hand_buffer_pixels
                            else:
                                bx = dif_xs_ys // 2
                                topx, bottomx = np.min(xs)-bx, np.max(xs)+bx
                                topy, bottomy = np.min(ys), np.max(ys)
                                topx_b, bottomx_b = topx-self.hand_buffer_pixels, bottomx+self.hand_buffer_pixels
                                topy_b, bottomy_b = topy-self.hand_buffer_pixels, bottomy+self.hand_buffer_pixels
                    
                            cv2.rectangle(datasette_frame, (topx, topy), (bottomx, bottomy), (255,255,255), 1)
                            crop_hand = datasette_frame_video[topy_b:bottomy_b+1, topx_b:bottomx_b+1]
                            # pad = abs(crop_hand.shape[0] - crop_hand.shape[1]) // 2
                            datasette_hand = crop_hand.copy()
                            #if crop_hand.shape[0] > crop_hand.shape[1]:
                            #    chand = cv2.copyMakeBorder(
                            #                    crop_hand,
                            #                   0,
                            #                    0,
                            #                    pad,
                            #                    crop_hand.shape[0]-pad-crop_hand.shape[1],
                            #                    cv2.BORDER_CONSTANT
                            #                )
                            #else:
                            #    chand = cv2.copyMakeBorder(
                            #                    crop_hand,
                            #                    pad,
                            #                    crop_hand.shape[1]-pad-crop_hand.shape[0],
                            #                    0,
                            #                    0,
                            #                    cv2.BORDER_CONSTANT
                            #
                            #                 )
                            
                            #datasette_hand = cv2.resize(chand, (datasette_hand.shape[1],datasette_hand.shape[0]))
                            # send data message
                            
                            #message = {
                            #    'DATA': 1,
                            #    'xy': region.lm_xy_normalized,
                            #    'xy_y_rescaled': region.lm_xy_y_rescaled,
                            #    'hand_img': crop_hand
                            #}
                            #self.queue.put(message)
                            if capture_flag:
                                data = {
                                    'frame_size': self.frame_size,
                                    'topx': topx, 
                                    'bottomx': bottomx,
                                    'topy': topy,
                                    'bottomy': bottomy,
                                    'xy': region.lm_xy_normalized,
                                    'xy_y_rescaled': region.lm_xy_y_rescaled,
                                    'hand_img': crop_hand,
                                    'frame': frame_number
                                }
                                temp_capture.append(data)

                instr = f"q: quit | r: start capture | s: save ({self.label})"
                self.current_fps.display(datasette_frame, orig=(50,50),color=(240,180,100))
                self.show_instructions(instr, datasette_frame, (50,70), size=0.8)
                cv2.imshow("Landmarks", datasette_frame)
                cv2.imshow("Right Hand", datasette_hand)
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # Save depth limits
                    break

                if key == ord('r'):
                    # start capturing
                    capture_flag = True                   

                if key == ord('s'):
                    # save
                    self.capture = temp_capture
                    capture_flag = False      
                    temp_capture = []

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
    parser.add_argument('--label', default='P0', type=str, help="Label name")
    args = parser.parse_args()

    # OAKD inference processor
    datasette = DatasetteHandCapture(
        pd_path=args.pdblob,
        pd_score_thresh=args.pdth,
        pd_nms_thresh=args.pdnms,
        lm_path=args.lmblob,
        lm_score_threshold=args.lmth,
        fps=args.fps,
        preview_width=args.prwidth,
        preview_height=args.prheight,
        hand_buffer_pixels=args.pixbuff,
        hand_size=args.hsize,
        label=args.label
    )
    datasette.run() 

    # Save capture to file
    if datasette.capture:
        # Save to data path: images - format: label_frame_date
        if not os.path.isdir(PARAMS['DATASET_PATH']):
            print(f"{PARAMS['DATASET_PATH']} directory does not exists")
            exit()

        dt = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        root = PARAMS['DATASET_PATH'] + "/" + args.label
        if not os.path.isdir(root):
            os.mkdir(root)

        for datapoint in datasette:
            filename_img = f"{root}/img_{datapoint['frame']}_{dt}.jpeg"
            cv2.imwrite(filename_img, datapoint['hand_img'])
            position = {
                'xy': datapoint['xy'],
                'xy_y_rescaled': datapoint['xy_y_rescaled']
            } 
            filename_pos = f"{root}/pos_{datapoint['frame']}_{dt}.pkl"
            pickle.dump(position, open(filename_pos, "wb"))


