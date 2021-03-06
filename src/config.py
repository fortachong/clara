import depthai as dai

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
    'PALM_THRESHOLD': 0.5,
    'PALM_NMS_THRESHOLD': 0.3,
    'PALM_DETECTION_INPUT_LENGTH': 128,
    'LM_DETECTION_MODEL_PATH': "models/hand_landmark.blob",
    'LM_THRESHOLD': 0.5,
    'LM_INPUT_LENGTH': 224,
    'FPS': 30,
    'ROI_DP_LOWER_TH': 100,
    'ROI_DP_UPPER_TH': 10000,
    'INITIAL_ROI_TL': dai.Point2f(0.4, 0.4),
    'INITIAL_ROI_BR': dai.Point2f(0.6, 0.6),
    'PREVIEW_WIDTH': 640,
    'PREVIEW_HEIGHT': 400,
    'HAND_BUFFER_PIXELS': 20,
    'HAND_SIZE': 400,
    'DATASET_PATH': 'data/positions',
    'DEPTH_ROI_FILENAME': 'roi.pkl',
    'DEPTH_CAPTURE_FILENAME': 'depth.pkl',
    'BODY_ROI_FILENAME': 'roi_position.pkl',
    'ANTENNA_ROI_FILENAME': 'antenna_position.pkl',
    'BODY_BUFFER': 50,
    'ANTENNA_BUFFER': 5,
    'ROI_HAND_ID': 2,
    'DEPTH_CAMERA_RESOLUTIONS': {
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
    },
    'DEPTH_RESOLUTION': '400',
    'INTRINSICS_RIGHT_CX': 318.04592896,
    'INTRINSICS_RIGHT_CY': 198.99064636,
    'INTRINSICS_RIGHT_FX': 427.05795288,
    'INTRINSICS_RIGHT_FY': 427.38696289,
    #'INTRINSICS_RIGHT_CX': 636.0918579101562,
    #'INTRINSICS_RIGHT_CY': 357.9812927246094,
    #'INTRINSICS_RIGHT_FX': 854.1159057617188,
    #'INTRINSICS_RIGHT_FY': 854.77392578125,
    'SC_SERVER': '127.0.0.1',
    'SC_PORT': 57121
} 

