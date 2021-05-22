import numpy as np
import cv2
import depthai as dai

# Parameters
PARAMS = {
    'CAPTURE_DEVICE': 0,
    'KEY_QUIT': 'q',
    'HUD_COLOR': (153,219,112),
    'VIDEO_DIMS_W': 640,
    'VIDEO_DIMS_H': 480
}

# Draw a Hud on top of frame
def draw_hud(img, n_segments=4):
    imgd = img.copy()
    cv2.line(
        imgd, (0,0),
        (PARAMS['VIDEO_DIMS_W'],PARAMS['VIDEO_DIMS_H']),
        PARAMS['HUD_COLOR'],
        1,cv2.LINE_8)
    cv2.line(
        imgd, (PARAMS['VIDEO_DIMS_W'],0),
        (0, PARAMS['VIDEO_DIMS_H']),
        PARAMS['HUD_COLOR'],
        1,cv2.LINE_8)

    # Draw rectangles for the HUD
    if n_segments % 2 == 0:
        lx = PARAMS['VIDEO_DIMS_W']//(2*n_segments)
        ly = PARAMS['VIDEO_DIMS_H']//(2*n_segments)

        pxs = np.arange(0, PARAMS['VIDEO_DIMS_W'], lx)
        pys = np.arange(0, PARAMS['VIDEO_DIMS_H'], ly)

        lpxs = pxs.shape[0]
        lpys = pys.shape[0]
        if lpxs > 1 and lpys > 1:
            starts_x = list(pxs[1:lpxs//2])
            starts_y = list(pys[1:lpys//2])
            ends_x = list(pxs[lpxs//2 + 1:][::-1])
            ends_y = list(pys[lpys//2 + 1:][::-1])
            for sx,sy,ex,ey in zip(starts_x, starts_y, ends_x, ends_y):
                cv2.rectangle(imgd, (sx,sy), (ex,ey), PARAMS['HUD_COLOR'],1,cv2.LINE_8)
        
    cv2.imshow('hud', imgd)        

# Define the pipeline
def define_pipeline():
    pipeline = dai.Pipeline()
    # Source - colour camera
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(PARAMS['VIDEO_DIMS_W'], PARAMS['VIDEO_DIMS_H'])
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Output
    out = pipeline.createXLinkOut()
    out.setStreamName("video_rgb")
    cam.preview.link(out.input)

    return pipeline

# Creates pipeline and process
def process():
    pipeline = define_pipeline()

    with dai.Device(pipeline) as device:
        # Start Pipeline
        device.startPipeline()

        # Output queue
        qrgb = device.getOutputQueue(name='video_rgb', maxSize=4, blocking=False)

        while True:
            # Blocking call
            imrgb = qrgb.get()

            type(imrgb)

            # Retrieve bgr opencv format
            cv2.imshow("bgr", imrgb.getCvFrame())

            # Press q for quitting
            if cv2.waitKey(1) == ord(PARAMS['KEY_QUIT']):
                break

if __name__ == "__main__":
    process()