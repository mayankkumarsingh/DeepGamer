import numpy as np
import cv2
import time
import pyautogui
from direct_keys import PressKey, ReleaseKey, W, A, S, D
from grabscreen import grab_screen
from read_keyboard import key_check
#from learn_by_watching import keys_to_output
from alexnet import alexnet
import os

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 5
m = '119K'
MODEL_NAME = 'nfs-run-{}-LR-{}-{}-epochs-{}-trsize.model'.format(LR,'alexnet_mdfd',EPOCHS,m)

def accelerate():
    PressKey(W)
    time.sleep(0.2)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)

def left():
    PressKey(A)
    time.sleep(0.175)
    PressKey(W)
    time.sleep(0.025)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def right():
    PressKey(D)
    time.sleep(0.175)
    PressKey(W)
    time.sleep(0.025)
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)

def brake():
    PressKey(S)
    time.sleep(0.2)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    tic = time.time()

    paused = False
    while True:

        if not paused:
            tic = time.time()
            capture_region = (0,40,640,520)
            image = grab_screen(capture_region)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (80,60))
            cv2.imshow('Test', image)
            prediction = model.predict([image.reshape(WIDTH, HEIGHT,1)])

            moves = np.around(prediction)
            # Taking the move to be arg max of predictions
            moves = np.int32(prediction == np.max(prediction))

            print('Moves: ', moves[0], '   Prediction: ',prediction[0])
            print((moves[0] == [0,1,0,0]).all())

            if ((moves[0] == [1,0,0,0]).all()):
                print('Taking Left')
                left()
            elif ((moves[0] == [0,1,0,0]).all()):
                print('Going Straight')
                accelerate()
            elif ((moves[0] == [0,0,1,0]).all()):
                print('Taking Right')
                right()
            elif ((moves[0] == [0,0,0,1]).all()):
                print('Braking')
                brake()

        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)


main()
