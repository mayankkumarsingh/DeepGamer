import numpy as np
import cv2
import os
import glob
import time
from PIL import ImageGrab, Image
from direct_keys import PressKey, ReleaseKey, W, A, S, D
# Preallocating screengrab array
screengrab = np.zeros((480,640,3))

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

print('down')
PressKey(W)
time.sleep(5)
ReleaseKey(A)
print('up')
PressKey(S)
time.sleep(3)

def process_screengrab(orig_image):
    # colored image = [[0-255],[0-255],[0-255]] GRAY = [0-255]
    processed_image = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
    processed_image = cv2.adaptiveThreshold(processed_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,5)
    # processed_image = cv2.Canny(processed_image,threshold1=100,threshold2=300)
    return processed_image


while(True):
    tic = time.time()
    orig_screengrab = np.array(ImageGrab.grab(bbox=(0,40,640,520)))
    processed_screengrab = process_screengrab(orig_screengrab)
    #print(screengrab.shape)
    cv2.imshow('Captured Screen', orig_screengrab)
    cv2.imshow('Processed Screen', processed_screengrab)
    #print('Grabbing + Displaying took {} seconds'.format(time.time() - tic))
    # Grabbing + Displaying takes around 0.15 secs per frame of size(480,640)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break
