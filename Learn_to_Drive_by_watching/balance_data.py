import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import os, time, glob


train_data = np.load('training_data.npy')

# for data in train_data:
#     img = data[0]
#     choice = data[1]
#     cv2.imshow('test',img)
#     print(choice)
#     if cv2.waitKey(25) & 0xFF==ord('q'):
#         cv2.destroyAllWindows()
#         break

df = pd.DataFrame(train_data)
print(df.head())

print(Counter(df[1].apply(str)))

lefts = []
rights = []
accelerate = []
brake = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0,0]:
        lefts.append([img,choice])
    elif choice == [0,1,0,0]:
        accelerate.append([img,choice])
    elif choice == [0,0,1,0]:
        rights.append([img,choice])
    elif choice == [0,0,0,1]:
        brake.append([img,choice])
    #else:
    #    print('No Keypress for this frame')
print('ORIGINAL DATA COMPOSITION')
print('Length of Training Data: ', len(train_data))
print('Length of accelerate: ', len(accelerate))
print('Length of Lefts: ', len(lefts))
print('Length of Rights: ', len(rights))
print('Length of Brakes: ', len(brake))
print('----------------------------------------------')
accelerate = accelerate[:2*len(lefts)][:2*len(rights)]
lefts = lefts[:len(lefts)][:len(rights)]
rights = rights[:len(lefts)]
balanced_data = accelerate + lefts + rights + brake
shuffle(balanced_data)
print('\n\nBALANCED DATA COMPOSITION')
print('Length of accelerate: ', len(accelerate))
print('Length of Lefts: ', len(lefts))
print('Length of Rights: ', len(rights))
print('Length of Brakes: ', len(brake))
print('TOTAL Length of Balanced data', len(balanced_data))
print('----------------------------------------------')

np.save('training_data_balanced.npy', balanced_data)
