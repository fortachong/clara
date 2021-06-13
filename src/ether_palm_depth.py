#!/usr/bin/env python3

# Team Clara:
# Elisa Andrade
# Jorge Chong

# Experiment with palm detector and depth calculator

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
    'MONO_LEFT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_400_P,
    'MONO_RIGHT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_400_P,
    'PALM_DETECTION_MODEL_PATH': "models/palm_detection.blob",
    'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
    'ROI_DP_LOWER_TH': 100,
    'ROI_DP_UPPER_TH': 10000,
    'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
    'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6),
    'PREVIEW_WIDTH': 640,
    'PREVIEW_HEIGHT': 400,
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
        self.queue = queue
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
            
        self.show_landmarks = True
        self.show_handedness = False

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

    # Render Palm
    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.frame_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.frame_size)
                    y = int(kp[1] * self.frame_size)
                    cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                    cv2.putText(frame, str(i), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Palm score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.frame_size+10), int((r.pd_box[1]+r.pd_box[3])*self.frame_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

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
            'original_xy_y_rescaled': region.lm_xy_y_rescaled,
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
        w = self.preview_width
        cv2.line(frame, (pos[0], 0), (pos[0], w), (0,255,0), 1)
        cv2.line(frame, (0, pos[1]), (w, pos[1]), (0,255,0), 1)

    # Draw mean line (take the mean of x pos and draw a line)
    def draw_mean_line_r(self, frame, region):
        if region.handedness >= 0.5:
            pos = region.lm_xy[PARAMS['LANDMARKS'], 0]
            mean_pos_x = int(np.mean(pos))
            w = self.preview_width
            cv2.line(frame, (mean_pos_x, 0), (mean_pos_x, w), (0,255,0), 1)

    # Draw min line (take the min of x pos and draw a line)
    def draw_min_line_r(self, frame, region):
        # print(region.handedness)
        if region.handedness >= 0.5:
            # pos = region.lm_xy[PARAMS['LANDMARKS'], 0]
            pos = region.lm_xy[:, 0]
            min_pos_x = int(np.min(pos))
            w = self.preview_width
            cv2.line(frame, (min_pos_x, 0), (min_pos_x, w), (0,255,0), 1)

    # Render ROIs for Spatial Calculator
    def roi_render(self, spatial_data, frame):
        for depth_data in spatial_data:
            roi = depth_data.config.roi
            roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color = PARAMS['LANDMARKS_COLOR']
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(frame, f"X: {int(depth_data.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), font, 0.5, color)
            cv2.putText(frame, f"Y: {int(depth_data.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), font, 0.5, color)
            cv2.putText(frame, f"Z: {int(depth_data.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), font, 0.5, color)

    # Pipeline
    def create_pipeline(self):
        # pix_w = 1280
        # pix_h = 800
        pix_w = 640
        pix_h = 400
        
        # For cropping of depth
        self.dy = pix_w / pix_h
        self.offset_y = ((pix_w - pix_h) // 2) / pix_h

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
        #mono_l.setFps(10)
        #mono_r.setFps(10)
        # Depth and Depth Calculator
        stereo = pipeline.createStereoDepth()
        spatial_calc = pipeline.createSpatialLocationCalculator()
        # Depth / Calculator / Config / Outputs and Inputs
        xout_depth = pipeline.createXLinkOut()
        xout_spatial_data = pipeline.createXLinkOut()
        xin_spatial_calc_config = pipeline.createXLinkIn()
        # Stream Names
        xout_depth.setStreamName("depth")
        xout_spatial_data.setStreamName("spatial")
        xin_spatial_calc_config.setStreamName("spatial_calc_config")
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
        spatial_calc.passthroughDepth.link(xout_depth.input)
        # Stereo Depth -> Stereo Calc
        stereo.depth.link(spatial_calc.inputDepth)
        # Configure the initial ROI (Body center) for Spatial Calc
        spatial_calc.setWaitForConfigInput(False)
        first_config = dai.SpatialLocationCalculatorConfigData()
        first_config.depthThresholds.lowerThreshold = PARAMS['ROI_DP_LOWER_TH']
        first_config.depthThresholds.upperThreshold = PARAMS['ROI_DP_UPPER_TH'] 
        first_config.roi = dai.Rect(PARAMS['INITIAL_ROI_TL'], PARAMS['INITIAL_ROI_BR'])
        spatial_calc.initialConfig.addROI(first_config)
        # # Spatial Calc -> Out spatial
        spatial_calc.out.link(xout_spatial_data.input)
        # In spatial_calc_config -> Spatial Calc Config
        xin_spatial_calc_config.out.link(spatial_calc.inputConfig)

        # Color Camera
        print("[{}]: Color Camera...".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(self.preview_width, self.preview_height)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
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
        return pipeline

    def run(self):
        pipeline = self.create_pipeline()
        with dai.Device(pipeline) as device:
            print(device.startPipeline())
                        
            # Queues
            # 1. Out: Video output
            q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            # 2. In: Palm Detector Input
            q_pd_in = device.getInputQueue(name="pd_in")
            # 3. Out: Palm Detector Output
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=False)
            # 6. Out: Depth
            q_d = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            # 7. Out: Spatial Avg Estimation (Calculator)
            q_spatial = device.getOutputQueue(name="spatial", maxSize=4, blocking=False)
            # 8. In: Config for Spatial Calculator
            q_spatial_config = device.getInputQueue("spatial_calc_config")

            while True:
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()

                # Depth
                in_depth = q_d.get()
                # Calculations
                in_depth_avg = q_spatial.get()
                # Depth map frame
                depth_frame = in_depth.getFrame()
                # Save First Spatial data input
                spatial_data = in_depth_avg.getSpatialLocations()

                # Normalize depth map
                depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depth_frame_color = cv2.equalizeHist(depth_frame_color)
                depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)
                # Render Depth + ROI
                self.roi_render(spatial_data, depth_frame_color)

                # Dimensions of the Video Frame
                h, w = video_frame.shape[:2]
                self.h = h
                self.w = w
                # Padding top and bottom
                self.frame_size = max(self.h, self.w)
                self.pad_h = int((self.frame_size - self.h)/2)
                self.pad_w = int((self.frame_size - self.w)/2)

                video_frame = cv2.copyMakeBorder(
                    video_frame, 
                    self.pad_h, 
                    self.pad_h, 
                    self.pad_w, 
                    self.pad_w, 
                    cv2.BORDER_CONSTANT
                )
                
                frame_nn = dai.ImgFrame()
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
                q_pd_in.send(frame_nn)

                annotated_frame = video_frame.copy()

                # Get palm detection
                inference = q_pd_out.get()
                self.pd_postprocess(inference)
                self.pd_render(annotated_frame)

                # Retrieve Landmarks
                for region in self.regions:
                    if region.pd_score > self.pd_score_thresh:
                        # Bounding Boxes from Palm Dectector -> Spatial Configs
                        spatial_configs = []
                        for idx, region in enumerate(self.regions):
                            config = dai.SpatialLocationCalculatorConfigData()
                            config.depthThresholds.lowerThreshold = PARAMS['ROI_DP_LOWER_TH']
                            config.depthThresholds.upperThreshold = PARAMS['ROI_DP_UPPER_TH'] 
                            # Convert to depth frame coordinates for ROI
                            xtl = region.pd_box[0]
                            ytl = region.pd_box[1]*self.dy - self.offset_y
                            xbr = (region.pd_box[0]+region.pd_box[2])
                            ybr = (region.pd_box[1]+region.pd_box[3])*self.dy - self.offset_y
                            config.roi = dai.Rect(
                                dai.Point2f(xtl, ytl),
                                dai.Point2f(xbr, ybr)
                            )
                            rz = config.roi.denormalize(
                                depth_frame_color.shape[1],
                                depth_frame_color.shape[0]
                            )
                            print(rz.x, rz.y, rz.width, rz.height)
                            spatial_configs.append(config)
                        # Add ROIs to Spatial Config Input Queue
                        if len(spatial_configs) > 0:
                            cfg = dai.SpatialLocationCalculatorConfig()
                            cfg.setROIs(spatial_configs)
                            q_spatial_config.send(cfg)
               
                        # Post message
                        #message = self.xy_message(region)
                        #self.queue.put(message)
                        
                #annotated_frame = annotated_frame[self.pad_h:self.pad_h+h, self.pad_w:self.pad_w+w]
                # invert
                inverted_frame = annotated_frame.copy()
                inverted_frame = cv2.flip(inverted_frame, 1)
                # show palm regions
                cv2.imshow("Palm Detection", inverted_frame)
                # show depth
                cv2.imshow("Depth Map", depth_frame_color)
                key = cv2.waitKey(1) 
                if key == ord('q') or key == 27:
                    # message = self.stop_message()
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
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/f", frequency)

        
    def set_volume(self, volume):
        print("------> theremin vol: {} <------".format(volume))
        sc_client = udp_client.SimpleUDPClient(self.sc_server, self.sc_port)
        sc_client.send_message("/main/a", volume)

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
        # pos = landmarks[PARAMS['LANDMARKS'], 0]
        pos = landmarks[:, 0]
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
            lm_l = 1.0 - message['original_xy_y_rescaled']
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
    # messages = Queue.Queue()
    # Process Thread
    # smp = SynthMessageProcessor(messages, synth, scale)
    # smp.start()
    # OAKD inference processor
    messages = None
    there = Ether(messages)
    there.run() 
