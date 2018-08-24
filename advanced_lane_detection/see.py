import numpy as np
import cv2
import os
import glob
import time
from PIL import ImageGrab, Image
from grabscreen import grab_screen
from lane_detection import Lane_Detector # process_screengrab

# Preallocating screengrab array
screengrab = np.zeros((480,640,3))

def find_lane_markers(img):

    #converted = convert_hls(img)
    image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0,   100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_img = cv2.bitwise_and(img, img, mask = mask)
    return masked_img


LD = Lane_Detector()

while(True):
    tic = time.time()

    # First Try - using PIL ImageGrab method - gives less frame rate
    # orig_screengrab = np.array(ImageGrab.grab(bbox=(0,40,640,520)))
    # Grabbing + Displaying takes around 0.15 secs per frame of size(480,640)

    # Capture Region based on resolution
    # 40px to allow for the window title bar, bottom = 480 + 40 = 520
    capture_region = (0,40,640,520)
    orig_screengrab = grab_screen(capture_region)
    # Grabbing + Displaying takes around 0.03 secs per frame of size(480,640)

    mask_birdeye, mask_1ch, lane_masked_image, roi_extracted_image, orig_image = LD.process_screengrab(orig_screengrab, capture_region)
    #print(processed_screengrab.shape)


    cv2.imshow('Original ScreeGrab', cv2.cvtColor(orig_screengrab, cv2.COLOR_RGB2BGR))
    cv2.imshow('Region of Interest Extracted', cv2.cvtColor(roi_extracted_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Lane Masked Image',  cv2.cvtColor(lane_masked_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Binary Mask',mask_1ch)
    cv2.imshow('Mask after Bird Eye Transform', mask_birdeye)

    lane_masked_image = find_lane_markers(orig_screengrab)
    # output of find_lane_markers
    #cv2.imshow('Raw Lane Detection', lane_masked_image)

    print('Grabbing + Displaying took {} seconds'.format(time.time() - tic))
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
