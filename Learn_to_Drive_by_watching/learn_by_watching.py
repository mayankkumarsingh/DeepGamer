import numpy as np
import cv2
import time
import pyautogui
from direct_keys import PressKey, ReleaseKey
from grabscreen import grab_screen
from read_keyboard import key_check
import os

TRAINING_DATA_FILE = 'training_data.npy'

if os.path.isfile(TRAINING_DATA_FILE):
    print('File exists, loading previous data')
    training_data = list(np.load(TRAINING_DATA_FILE))
    existing_length = len(training_data)
    print('Length of existing Training Data', existing_length)
else:
    print('File does not exist, starting fresh')
    existing_length = 0
    training_data = []

def keys_to_output(keys):
    # outputs = one hot representation of [A,W,D,S]
    output = [0,0,0,0]
    if 'A' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'S' in keys:
        output[3] = 1

    return output

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    print('Watching you Drive now')
    tic = time.time()

    while True:

        tic = time.time()
        capture_region = (0,40,640,520)
        image = grab_screen(capture_region)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([image,output])
        #print('All this took {} secs'.format(time.time() - tic))

        if len(training_data) % 500 == 0 :
            print('Training Data Length: ', len(training_data), '    Saving...')
            np.save(TRAINING_DATA_FILE, training_data)
        if len(training_data) - existing_length > 5000:
            print('Training_data Max length of 5000 reached')
            print('Qutting now')
            break
        keys = key_check()
        if 'T' in keys:
            print('Quit without saving')
            break

main()
