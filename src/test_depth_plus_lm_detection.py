#!/usr/bin/env python3

import sys
import cv2
import depthai as dai
import numpy as np
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now

# Credits:
# Palm Detector from: geax
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
    # 'MONO_LEFT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_800_P,
    # 'MONO_RIGHT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_800_P,
    'MONO_LEFT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_400_P,
    'MONO_RIGHT_RESOLUTION': dai.MonoCameraProperties.SensorResolution.THE_400_P,
    'PALM_DETECTION_MODEL_PATH': "models/palm_detection.blob",
    'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
    'ROI_DP_LOWER_TH': 100,
    'ROI_DP_UPPER_TH': 10000,
    'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
    'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6)
} 



# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)


# Hand Theremin Input
class HandTheremin:
    def __init__(
        self, 
        pd_path=PARAMS['PALM_DETECTION_MODEL_PATH'], 
        pd_score_thresh=0.5, 
        pd_nms_thresh=0.3,
        lm_path=PARAMS['LM_DETECTION_MODEL_PATH'],
        lm_score_threshold=0.5
    ):
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
        
        # Body spatial measurement
        self.body_spatial_data = None

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
        mpu.rect_transformation(self.regions, self.video_size, self.video_size)

    # Render Palm Box 
    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.video_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.video_size)
                    y = int(kp[1] * self.video_size)
                    cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                    cv2.putText(frame, str(i), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Palm score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.video_size+10), int((r.pd_box[1]+r.pd_box[3])*self.video_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)    

    # Process Landmarks
    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]    
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add"))
        
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3*i:3*(i+1)]/PARAMS['LM_INPUT_LENGTH'])
        region.landmarks = lm

    # Render Landmarks
    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:
                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
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
                    cv2.circle(frame, (x, y), 6, (0,128,255), -1)
            if self.show_handedness:
                cv2.putText(frame, f"RIGHT {region.handedness:.2f}" if region.handedness > 0.5 else f"LEFT {1-region.handedness:.2f}", 
                        (int(region.pd_box[0] * self.video_size+10), int((region.pd_box[1]+region.pd_box[3])*self.video_size+20)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if region.handedness > 0.5 else (0,0,255), 2)
            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
                        (int(region.pd_box[0] * self.video_size+10), int((region.pd_box[1]+region.pd_box[3])*self.video_size+90)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)


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
        the_1080_w = 1920
        the_1080_h = 1080
        # the_800_p_w = 1280
        # the_800_p_h = 800
        the_800_p_w = 640
        the_800_p_h = 400
        # For cropping of depth
        self.dx = the_1080_h / the_1080_w
        self.offset_x = ((the_1080_w - the_1080_h) / 2) / the_1080_w

        pipeline = dai.Pipeline()
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
        # Spatial Calc -> Out spatial
        spatial_calc.out.link(xout_spatial_data.input)
        # In spatial_calc_config -> Spatial Calc Config
        xin_spatial_calc_config.out.link(spatial_calc.inputConfig)

        # Color Camera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(
            PARAMS['PALM_DETECTION_INPUT_LENGTH'], 
            PARAMS['PALM_DETECTION_INPUT_LENGTH']
        )
        cam.setResolution(PARAMS['VIDEO_RESOLUTION'])
        # Crop video to square shape (palm detection takes square image as input)
        self.video_size = min(cam.getVideoSize())
        cam.setVideoSize(self.video_size, self.video_size)
        cam.setFps(30)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        
        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(PARAMS['PALM_DETECTION_MODEL_PATH']).resolve().absolute()))
        # Increase threads for detection
        pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Palm detection input                 
        pd_nn.input.setQueueSize(1)
        pd_nn.input.setBlocking(False)
        # cam -> pd
        cam.preview.link(pd_nn.input)
        # Palm detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        # Landmark detection
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
        lm_nn.setNumInferenceThreads(2)
        # Hand landmark input
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        # Hand Landmark output
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)

        # Outputs
        # Video output
        cam_out = pipeline.createXLinkOut()
        cam_out.setStreamName("cam_out")
        cam.video.link(cam_out.input)

        return pipeline

    def run(self):
        pipeline = self.create_pipeline()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_rec = cv2.VideoWriter('rgb.avi', fourcc, 20.0, (1080,1080))
        # depth_rec = cv2.VideoWriter('depth.avi', fourcc, 20.0, (1280,800))
        depth_rec = cv2.VideoWriter('depth.avi', fourcc, 20.0, (640,400))

        with dai.Device(pipeline) as device:
            device.startPipeline()

            # Queues
            # 1. Out: Video output
            q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            # 2. Out: Palm detector
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            # 3. Out: Depth
            q_d = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            # 4. Out: Spatial Avg Estimation (Calculator)
            q_spatial = device.getOutputQueue(name="spatial", maxSize=4, blocking=False)
            # 5. In: Config for Spatial Calculator
            q_spatial_config = device.getInputQueue("spatial_calc_config")
            # 6. Landmarks Out
            q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=False)
            # 7. Landmarks In
            q_lm_in = device.getInputQueue(name="lm_in")

            # FPS monitor
            self.fps = FPS(mean_nb_frames=20)

            seq_num = 0
            nb_pd_inferences = 0
            nb_lm_inferences = 0
            glob_pd_rtrip_time = 0
            glob_lm_rtrip_time = 0

            first_measurement = False

            while True:
                self.fps.update()

                # Video
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()
            
                annotated_frame = video_frame.copy()

                # Depth
                in_depth = q_d.get()
                # Calculations
                in_depth_avg = q_spatial.get()
                # Depth map frame
                depth_frame = in_depth.getFrame()
                # Save First Spatial data input
                spatial_data = in_depth_avg.getSpatialLocations()

                if not first_measurement and self.body_spatial_data is None:
                    self.body_spatial_data = spatial_data
                    first_measurement = True
                
                # Normalize depth map
                depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depth_frame_color = cv2.equalizeHist(depth_frame_color)
                depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)
                # Render Depth + ROI
                self.roi_render(spatial_data, depth_frame_color)
                
                # Palm Detection
                palm_inference = q_pd_out.get()
                self.pd_postprocess(palm_inference)
                self.pd_render(annotated_frame)
                nb_pd_inferences += 1

                # Bounding Boxes from Palm Dectector -> Spatial Configs
                spatial_configs = []
                for idx, region in enumerate(self.regions):
                    config = dai.SpatialLocationCalculatorConfigData()
                    config.depthThresholds.lowerThreshold = PARAMS['ROI_DP_LOWER_TH']
                    config.depthThresholds.upperThreshold = PARAMS['ROI_DP_UPPER_TH'] 
                    # Convert to depth frame coordinates for ROI
                    xtl = region.pd_box[0]*self.dx + self.offset_x
                    ytl = region.pd_box[1]
                    xbr = (region.pd_box[0]+region.pd_box[2])*self.dx + self.offset_x
                    ybr = region.pd_box[1]+region.pd_box[3]
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
                # # Landmarks
                for idx, region in enumerate(self.regions):
                    imgh = mpu.warp_rect_img(
                        region.rect_points,
                        video_frame,
                        PARAMS['LM_INPUT_LENGTH'],
                        PARAMS['LM_INPUT_LENGTH']
                    )
                    nn_data = dai.NNData()
                    nn_data.setLayer(
                        "input_1", 
                        to_planar(
                            imgh, 
                            (PARAMS['LM_INPUT_LENGTH'],PARAMS['LM_INPUT_LENGTH'])
                        )
                    )
                    q_lm_in.send(nn_data)
                    if idx == 0: lm_rtrip_time = now()

                # Retrieve hand landmarks
                for idx, region in enumerate(self.regions):
                    inference = q_lm_out.get()
                    if idx == 0: glob_lm_rtrip_time += now() - lm_rtrip_time
                    self.lm_postprocess(region, inference)
                    self.lm_render(annotated_frame, region)
                    nb_lm_inferences += 1

                # Show FPS
                self.fps.display(annotated_frame, orig=(50,50), color=(240,180,100))
                # Show Video
                cv2.imshow("video", annotated_frame)
                # Show Depth Frame
                cv2.imshow("depth", depth_frame_color)

                video_rec.write(annotated_frame)
                depth_rec.write(depth_frame_color)

                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    break
                elif key == 32:
                    # Pause on space bar
                    cv2.waitKey(0)
                elif key == ord('1'):
                    self.show_pd_box = not self.show_pd_box
                elif key == ord('2'):
                    self.show_pd_kps = not self.show_pd_kps
                elif key == ord('3'):
                    self.show_rot_rect = not self.show_rot_rect
                elif key == ord('4'):
                    self.show_scores = not self.show_scores
                if cv2.waitKey(1) == ord('q'):
                    break

        video_rec.release()
        depth_rec.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    there = HandTheremin()
    there.run() 