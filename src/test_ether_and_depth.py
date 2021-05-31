#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Currently: we are having problems creating more input/output nodes
# So for the moment depth is not used but we experimented with it

from os import X_OK
import cv2
import depthai as dai
import numpy as np
from datetime import datetime
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import threading
import queue as Queue
import well_tempered as wtmp
from pythonosc import udp_client
import argparse


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
    'MEDIAPIPE_HANDS_MODE': False,
    'MEDIAPIPE_HANDS_MAXHANDS': 2,
    'MEDIAPIPE_HANDS_DETECTION_CONFIDENCE': 0.5,
    'MEDIAPIPE_HANDS_TRACKING_CONFIDENCE': 0.5,
    'PALM_DETECTION_INPUT_LENGTH': 128,
    'LM_INPUT_LENGTH': 224,
    'VIDEO_RESOLUTION': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    'MONO_LEFT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_800_P,
    'MONO_RIGHT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_800_P,
    # 'MONO_LEFT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_400_P,
    # 'MONO_RIGHT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_400_P,
    # 'PALM_DETECTION_MODEL_PATH': "models/palm_detection_6_shaves.blob",
    # 'LM_DETECTION_MODEL_PATH': "models/hand_landmark_6_shaves.blob",
    'PALM_DETECTION_MODEL_PATH': "models/palm_detection.blob",
    'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
    'ROI_DP_LOWER_TH': 100,
    'ROI_DP_UPPER_TH': 10000,
    'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
    'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6),
    'PREVIEW_WIDTH': 576,
    'PREVIEW_HEIGHT': 324,
    'SCREEN_MIN_FREQ': 0.5,
    'SCREEN_MAX_FREQ': 1,
    'SC_SERVER': '127.0.0.1',
    'SC_PORT': 57121
} 

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST)
    return resized.transpose(2,0,1)

# Hand tracking based theremin (the CV Part)
class Ether:
    def __init__(
        self, 
        queue,
        pd_path=PARAMS['PALM_DETECTION_MODEL_PATH'], 
        pd_score_thresh=0.5, 
        pd_nms_thresh=0.3,
        lm_path=PARAMS['LM_DETECTION_MODEL_PATH'],
        lm_score_threshold=0.5
    ):
        # landmark kernel sizes
        self.kernel = 7

        # 
        self.the_1080_w = 1920
        self.the_1080_h = 1080
        # For cropping of depth
        self.dx = self.the_1080_h / self.the_1080_w
        self.offset_x = ((self.the_1080_w - self.the_1080_h) / 2) / self.the_1080_w
        # Focal Length
        self.focal_length = 860
        # Message queue
        self.queue = queue
        # Palm detector
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        # Landmark detector
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
            
        self.show_landmarks = True
        self.show_handedness = False
        # For debugging
        self.show_depth_map = True

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
        
        # Rendering flags
        self.show_pd_box = True
        self.show_pd_kps = False
        self.show_rot_rect = False
        self.show_scores = False
        self.show_landmarks = True
        
        # Preview Sizes
        self.preview_width = PARAMS['PREVIEW_WIDTH']
        self.preview_height = PARAMS['PREVIEW_HEIGHT']

        # Palm detector input size
        self.pd_input_length = PARAMS['PALM_DETECTION_INPUT_LENGTH']

        # Landmark detector input size
        self.lm_input_length = PARAMS['LM_INPUT_LENGTH']

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

    # Transform the normalized x, y and form a message to send
    # to a queue for further processing (interaction)
    def xy_message(self, region):
        handedness = 'R'
        if region.handedness < 0.5:
            handedness = 'L'
        # Renormalize
        w = self.preview_width
        h = self.preview_height
        y_factor = w/h
        pts = region.lm_xy_normalized.copy()
        pts[:,1] = pts[:,1]/y_factor
        # Message
        message = {
            'handedness': handedness,
            'original_xy': region.lm_xy_normalized,
            'xy': pts
        }
        return message

    # Stop Sending Messages
    def stop_message(self):
        message = {
            'STOP': 1
        }
        return message

    # Draw lines
    def draw_lines(self, frame, region):
        pos = region.lm_xy[0, :]
        w = frame.shape[0]
        cv2.line(frame, (pos[0], 0), (pos[0], w), (0,255,0), 1)
        cv2.line(frame, (0, pos[1]), (w, pos[1]), (0,255,0), 1)

    # Draw mean line (take the mean of x pos and draw a line)
    def draw_mean_line_r(self, frame, region):
        if region.handedness >= 0.5:
            pos = region.lm_xy[PARAMS['LANDMARKS'], 0]
            mean_pos_x = int(np.mean(pos))
            w = frame.shape[0]
            cv2.line(frame, (mean_pos_x, 0), (mean_pos_x, w), (0,255,0), 1)

    # Draw min line (take the min of x pos and draw a line)
    def draw_min_line_r(self, frame, region):
        # print(region.handedness)
        if region.handedness >= 0.5:
            pos = region.lm_xy[PARAMS['LANDMARKS'], 0]
            min_pos_x = int(np.min(pos))
            w = frame.shape[0]
            cv2.line(frame, (min_pos_x, 0), (min_pos_x, w), (0,255,0), 1)

    # Normalize depth map

    # Draw Body center

    # Process depth
    def process_depth(self, depth_map, region):
        # print(region.lm_xy_normalized)
        dm_h, dm_w = depth_map.shape
        dm = (region.lm_xy_normalized * np.array([dm_w, dm_h])).astype(int)
        centers_x = np.clip(dm[:,0], 0, dm_w-1)
        centers_y = np.clip(dm[:,1], 0, dm_h-1)
        lim_inf_x = np.clip(centers_x - self.kernel//2, 0, dm_w-1)
        lim_sup_x = np.clip(centers_x + self.kernel//2, 0 , dm_w-1)
        lim_inf_y = np.clip(centers_y - self.kernel//2, 0, dm_h-1)
        lim_sup_y = np.clip(centers_y + self.kernel//2, 0, dm_h-1)

        depth_center_values = depth_map[centers_y, centers_x]
        depth_avgs = []
        for lix, lsx, liy, lsy in zip(lim_inf_x, lim_sup_x, lim_inf_y, lim_sup_y):
            depth_avgs.append(np.ravel(depth_map[liy:lsy, lix:lsx]).mean())

        xyz = []
        for idx, d in enumerate(depth_avgs):
            pixel_x = region.lm_xy[idx][0]
            pixel_y = region.lm_xy[idx][1]
            x = ((pixel_x - self.frame_center_x) * d) / self.focal_length + self.frame_center_x
            y = ((pixel_y - self.frame_center_y) * d) / self.focal_length + self.frame_center_y
            z = d
            xyz.append((x,y,z))

        #print(dm)
        #print(depth_center_values)
        #print(depth_avgs)
        print(xyz)
        region.depth_centers = dm
        region.depth_center_values = depth_center_values
        region.lim_inf_x = lim_inf_x
        region.lim_sup_x = lim_sup_x
        region.lim_inf_y = lim_inf_y
        region.lim_sup_y = lim_sup_y
        region.depth_avgs = depth_avgs
        region.xyz = xyz


    # Mask to verify depth
    def get_updated_mask(self, mask, region):
        for lix, lsx, liy, lsy in zip(region.lim_inf_x, region.lim_sup_x, region.lim_inf_y, region.lim_sup_y):
            mask[liy:lsy, lix:lsx] = 1
        return mask

    # Pipeline
    def create_pipeline(self):
        print("[{}]: Creating Pipeline...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        # Pipeline
        pipeline = dai.Pipeline()
        print("[{}]: Setting up Depth...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        # Mono Left Camera
        mono_l = pipeline.createMonoCamera()
        # Mono Right Camera
        mono_r = pipeline.createMonoCamera()
        # Mono Camera Settings
        mono_l.setResolution(PARAMS['MONO_LEFT_RESOLUTION'])
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_r.setResolution(PARAMS['MONO_RIGHT_RESOLUTION'])
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_l.setFps(10)
        mono_r.setFps(10)
        # Depth and Depth Calculator
        stereo = pipeline.createStereoDepth()
        xout_depth = pipeline.createXLinkOut()
        # Stream Names
        xout_depth.setStreamName("depth")
        # Stereo Depth parameters
        output_depth = True
        output_rectified = False
        lr_check = False
        subpixel = False
        stereo.setOutputDepth(output_depth)
        stereo.setOutputRectified(output_rectified)
        stereo.setConfidenceThreshold(255)
        stereo.setLeftRightCheck(lr_check)
        stereo.setSubpixel(subpixel)
        # Mono L / R -> Stereo L / R
        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)
        # Passtrough Stereo -> Stereo Calc -> Output Depth
        # Stereo Depth -> Out
        stereo.depth.link(xout_depth.input)
        # Color Camera
        print("[{}]: Color Camera...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
        cam.setResolution(PARAMS['VIDEO_RESOLUTION'])
        # Crop video to square shape (palm detection takes square image as input)
        self.frame_size = min(cam.getVideoSize())
        self.frame_center_x = self.frame_size // 2
        self.frame_center_y = self.frame_center_x
        cam.setVideoSize(self.frame_size, self.frame_size)
        cam.setFps(30)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        # Video output
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.video.link(cam_out.input)

        # Palm Detector
        print("[{}]: Mediapipe Palm Detector NN...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.input.setQueueSize(1)
        pd_nn.input.setBlocking(False)
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        pd_nn.setNumInferenceThreads(2)
        # cam -> pd_in
        cam.preview.link(pd_nn.input)
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
            print(device.startPipeline())
            # exit()
            
            # Queues
            # 1. Out: Video output
            q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            # 2. Out: Palm Detector Output
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=False)
            # 4. Landmarks Out
            q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=False)
            # 5. Landmarks In
            q_lm_in = device.getInputQueue(name="lm_in")
            # 6. Out: Depth
            q_d = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            while True:
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()
                annotated_frame = video_frame.copy()
                # resize

                # Depth
                in_depth = q_d.get()
                depth_frame = in_depth.getFrame()
                df_h, df_w = depth_frame.shape
                dframe = depth_frame.copy()
                dframe = dframe[:,int((df_w - df_h)/2):int((df_w - df_h)/2) + df_h]
                mask = np.zeros(dframe.shape)
                
                # print(depth_frame.shape)
                # (800, 1280)
                
                # Normalize depth map
                depth_frame_color = cv2.normalize(dframe, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depth_frame_color = cv2.equalizeHist(depth_frame_color)
                depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)
                             
                # Palm Detection
                inference = q_pd_out.get()
                self.pd_postprocess(inference)
                # self.pd_render(annotated_frame)

                # Data for landmarks
                for region in self.regions:
                    img_hand = mpu.warp_rect_img(
                        region.rect_points, 
                        video_frame, 
                        self.lm_input_length, 
                        self.lm_input_length
                    )
                    nn_data = dai.NNData()
                    nn_data.setLayer(
                        "input_1", 
                        to_planar(img_hand, (self.lm_input_length, self.lm_input_length))
                    )
                    q_lm_in.send(nn_data)

                # Retrieve Landmarks
                for region in self.regions:
                    inference = q_lm_out.get()
                    self.lm_postprocess(region, inference)
                    self.lm_render(annotated_frame, region)

                    # Print Handedness, and Coordinates
                    if region.lm_score > self.lm_score_threshold:
                        # [0,1] normalization of loandmark points
                        self.lm_transform(region)
                        # self.draw_mean_line(annotated_frame, region)
                        self.draw_min_line_r(annotated_frame, region)

                        # Post message
                        message = self.xy_message(region)
                        # self.queue.put(message)

                        # Retrieve depth values
                        self.process_depth(dframe, region)
                        self.get_updated_mask(mask, region)
                        
                        

                # invert
                inverted_frame = annotated_frame.copy()
                inverted_frame = cv2.flip(inverted_frame, 1)

                # show
                cv2.imshow("Landmarks", inverted_frame)
                cv2.imshow("Depth", depth_frame_color)
                cv2.imshow("Mask", mask)

                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    message = self.stop_message()
                    # self.queue.put(message)

                    break
                elif key == 32:
                    # Pause on space bar
                    cv2.waitKey(0)
            
            # message = self.stop_message()
            # self.queue.put(message)

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
        #sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        #sc_client.send_message("/main/f", frequency)

        
    def set_volume(self, volume):
        print("------> theremin vol: {} <------".format(volume))
        #sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        #sc_client.send_message("/main/a", volume)

# Process messages from inference (specific hand landmarks)
# and send proper parameters to synthesizer
class SynthMessageProcessor(threading.Thread):
    def __init__(
            self, 
            queue, 
            synth, 
            scale,
            min_freq=PARAMS['SCREEN_MIN_FREQ'], 
            max_freq=PARAMS['SCREEN_MAX_FREQ'] 
        ):
        threading.Thread.__init__(self)
        self.synth = synth
        self.queue = queue
        self.active = True
        self.screen_min_freq = min_freq
        self.screen_max_freq = max_freq
        # Scale
        self.scale = scale

        # Vol
        self.volume = 0

    # Process a Hand Landmark Message
    def process(self, message):        
        landmarks = 1.0 - message['original_xy']
        # print(landmarks)
        pos = landmarks[PARAMS['LANDMARKS'], 0]
        # mean_pos_x = np.mean(pos)
        max_pos_x = np.max(pos)

        # Rigth Hand: Tone Control
        if message['handedness'] == 'R':
            if max_pos_x >= self.screen_min_freq:
                # clip and rescale to [0,1]
                tmp_x = np.clip(max_pos_x, self.screen_min_freq, self.screen_max_freq)
                # tmp_x = np.clip(mean_pos_x, self.screen_min_freq, self.screen_max_freq)
                x = (tmp_x - self.screen_min_freq) / (self.screen_max_freq - self.screen_min_freq)
                # convert freq
                freq = self.scale.from_0_1_to_f(x)          
                # send to synth
                self.synth.set_tone(freq)

        # Left Hand: Volume Control
        if message['handedness'] == 'L':
            lm_l = 1.0 - message['original_xy']
            # print(lm_l)
            pos_y = lm_l[PARAMS['LANDMARKS'], 1]
            min_pos_y = np.min(pos_y)
            max_pos_y = np.max(pos_y)
            if max_pos_y > self.volume:
                self.volume = max_pos_y
            else:
                self.volume = min_pos_y
            # clip to [0,1]
            y = np.clip(self.volume, 0, 1)

            if y > 0.8:
                y = 1
            if y < 0.2:
                y = 0

            # send to synth
            self.synth.set_volume(y)
    
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
    args = parser.parse_args()

    scale = wtmp.WellTempered(octaves=3, start_freq=440, resolution=10000)
    # Create Synthesizer
    synth = EtherSynth(args.scserver, args.scport)
    # Message Queues
    #messages = Queue.Queue()
    # Process Thread
    #smp = SynthMessageProcessor(messages, synth, scale)
    #smp.start()
    # OAKD inference processor
    messages = None
    there = Ether(messages)
    there.run() 
