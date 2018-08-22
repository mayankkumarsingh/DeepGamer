import os
os.chdir('advanced_lane_detection')
pwd


import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from image_processing_utils import *



mask_1ch = np.load('bin_mask.npy')
lane_masked_image = np.load('lane_masked_img.npy')

capture_region = (0,40,640,520)
mask_birdeye = BirdsEyePerspective(mask_1ch, capture_region)

cv2.imshow('BINARY MASK',mask_1ch)
cv2.imshow('LANE MASKED_IMAGE', cv2.cvtColor(lane_masked_image, cv2.COLOR_RGB2BGR))
cv2.imshow('BIRD EYE MASK',mask_birdeye)
cv2.waitKey(0)

img = mask_birdeye
steps = 10
search_window = (0, mask_birdeye.shape[1])
h_window = 7
masked_img = img[:, search_window[0]:search_window[1]]
pixels_per_step = img.shape[0] // steps
peak_threshold = 1500
frame_debug = True

histogram_overlay, all_x, all_y = histogram_lane_detection(img, steps, search_window, h_window, peak_threshold, frame_debug)

cv2.imshow('histogram',cv2.cvtColor(histogram_overlay,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)


i=1

histogram_overlay, all_x, all_y = histogram_lane_detection(img, steps, search_window, h_window, peak_threshold=2000, frame_debug = True)

cv2.imshow('OVERLAY',cv2.cvtColor(histogram_overlay,cv2.COLOR_RGB2BGR))

start = masked_img.shape[0] - (i * pixels_per_step)
end = start - pixels_per_step
histogram = np.sum(masked_img[end:start, :], axis=0)
histogram_smooth = signal.medfilt(histogram, h_window)
peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 10)))
print(peaks)
peaks = peaks[np.nonzero(histogram_smooth[peaks] > peak_threshold)]
print(peaks)
print(histogram_smooth)
print(histogram_smooth/max(histogram_smooth))
histogram_overlay = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
COLOR_BOX_BORDER = np.array([100, 160, 255], np.uint8)
COLOR_HIST_PLOT = np.array([225, 150, 50], np.uint8)
COLOR_POSITIVE_DETECTION = np.array([0, 150, 0], np.uint8)
COLOR_THRESHOLD = np.array([50, 225, 225], np.uint8)
COLOR_GRID = np.array([100, 150, 150], np.uint8)
print(histogram_smooth.shape)
print(start,end)
histogram_overlay[start-2:start,:,:] = COLOR_BOX_BORDER
histogram_overlay[end-2:end,:,:] = COLOR_BOX_BORDER
histogram_overlay[end:start,0:2,:] = COLOR_BOX_BORDER
histogram_overlay[end:start,-2:,:] = COLOR_BOX_BORDER
histogram_overlay[end:start:(start-end)//4,1:-1:5,:] = COLOR_GRID
histogram_overlay[start-np.uint8(np.log1p(peak_threshold)*(pixels_per_step-4)/max(np.log1p(histogram_smooth))),1:-1:5,:] = COLOR_THRESHOLD
peak = peaks[0]
histogram_overlay[end:start:2,peak-1:peak+1,:] = COLOR_GRID

hst = np.uint8(np.log1p(histogram_smooth)*(pixels_per_step-4)/max(np.log1p(histogram_smooth)))
for i in range(len(histogram_smooth)-4):
    histogram_overlay[start-hst[i]-2,i+2,:] = COLOR_HIST_PLOT

print(hst)

cv2.destroyAllWindows()
cv2.imshow('histogram',cv2.cvtColor(histogram_overlay,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyWindow('histogram')



del histogram_lane_detection
histplot,allx, ally = histogram_lane_detection(mask_birdeye,steps=10,search_window=(0,640),h_window=7,peak_threshold=2000,frame_debug=True)

cv2.imshow('HISTOGRAM',cv2.cvtColor(histplot,cv2.COLOR_RGB2BGR))
