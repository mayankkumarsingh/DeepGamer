import numpy as np
import cv2
import os
import glob
import time
from PIL import ImageGrab, Image
from grabscreen import grab_screen
from lane_detection import process_screengrab
# Preallocating screengrab array
screengrab = np.zeros((480,640,3))


capture_region = (0,40,640,520)
orig_screengrab = grab_screen(capture_region)
portion = orig_screengrab[240:250,150:170,:]
print(orig_screengrab[250,150,:])
print(portion)
print(orig_screengrab.shape)
while True:
    cv2.imshow('Original ScreeGrab', cv2.cvtColor(orig_screengrab, cv2.COLOR_RGB2BGR))
    cv2.imshow('PORTION', cv2.cvtColor(portion, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
