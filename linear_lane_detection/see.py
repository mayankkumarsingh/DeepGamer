import numpy as np
import cv2
import os
import glob
import time
from PIL import ImageGrab, Image
from grabscreen import grab_screen
from draw_lanes import draw_lanes
# Preallocating screengrab array
screengrab = np.zeros((480,640,3))

def mask_image(img, vertices):
    mask = np.zeros_like(img)
    # Deciding a 3-channel or 1-channel color to fill the mask depending on
    # whether the input image is in color or grayscale.
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_vertices(l,t,b,r,w,h):

    # Saints Row 3rd Person Cam vertices for bike
    # vertices = np.array([ [l,b], [l+np.floor(w/10),t+np.floor(2*h/10)], [r-np.floor(w/10), \
    # t+np.floor(2*h/10)], [r,b], \
    # [r-np.floor(w/3),b], [r-np.floor(w/2)+35, b-np.floor(w/2)],\
    #  [r-np.floor(w/2)-35, b-np.floor(w/2)], [r-np.floor(2*w/3), b] ], np.int32)

    # Vertices for 1st Person Cam NFS Run
    vertices = np.array([ [l,b], [l,t], [r,t], [r,b] ], np.int32)

    # return the chosen vertices
    return vertices

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

def process_screengrab(orig_image, capture_region):
    # topleftx, toplefty, botrightx, botrighty = capture_region
    tlx, tly, brx, bry = capture_region
    w = brx - tlx
    h = bry - tly
    t = tly     # t = tly = 40
    b = bry     # b = bry = 520
    l = tlx     # t = tlx = 0
    r = brx     # r = brx = 640

    processed_image = find_lane_markers(orig_image)
    # colored image = [[0-255],[0-255],[0-255]] GRAY = [0-255]
    processed_image = cv2.cvtColor(processed_image,cv2.COLOR_BGR2GRAY)
    # Applying GAUSSIAN BLUR
    kernel_size = (5, 5)
    processed_image = cv2.GaussianBlur(processed_image, kernel_size, 0)
    # processed_image = cv2.adaptiveThreshold(processed_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,25, 15)
    #processed_image = cv2.Canny(processed_image,threshold1=50,threshold2=150)

    # Applying HOUGHLINESP TRANSFORM
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # 50 minimum number of pixels making up a line
    max_line_gap = 10  # 20 maximum gap in pixels between connectable line segments
    line_image = np.copy(orig_image) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(processed_image, rho, theta, threshold, np.array([]), \
                    min_line_length, max_line_gap)
    #print(lines.shape)
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        orig_image = cv2.addWeighted(orig_image, 0.8, line_image, 1, 0)
    except Exception as e:
        print(str(e))
        pass

    vertices = mask_vertices(l,t,b,r,w,h)
    processed_image = mask_image(processed_image, [vertices])
    return processed_image, orig_image


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
    #processed_image, original_image = process_screengrab(orig_screengrab, capture_region)
    processed_image, image_gray, image_with_lanes_bgr = draw_lanes(orig_screengrab, capture_region)
    #print(processed_screengrab.shape)

    lane_masked_image = find_lane_markers(orig_screengrab)
    cv2.imshow('Original ScreeGrab', cv2.cvtColor(orig_screengrab, cv2.COLOR_RGB2BGR))
    # output of find_lane_markers
    cv2.imshow('Lane Detection', lane_masked_image)
    # Draw Lane outputs
    cv2.imshow('Grayscale Image ', image_gray)
    cv2.imshow('Processed Screengrab',processed_image)
    cv2.imshow('Processed with lanes',image_with_lanes_bgr)
    print('Grabbing + Displaying took {} seconds'.format(time.time() - tic))
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
