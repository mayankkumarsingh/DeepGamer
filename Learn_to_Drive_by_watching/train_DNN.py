import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 5


# Training

model = alexnet(WIDTH, HEIGHT, LR)
train_data = np.load('training_data_augmented.npy')
m = str(int(len(train_data)/1000)) + 'K'
MODEL_NAME = 'nfs-run-{}-LR-{}-{}-epochs-{}-trsize.model'.format(LR,'alexnet_mdfd',EPOCHS,m)

TEST_DATA_LEN = 500

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_Y = [i[1] for i in test]

model.fit({'input':X}, {'targets':Y}, n_epoch = EPOCHS,
        validation_set = ({'input':test_X}, {'targets':test_Y}),
        snapshot_epoch = True, snapshot_step = 500, show_metric = True, run_id = MODEL_NAME)

# tensorboard --logdir=foo:C:/Mayank/Coursera/Projects/DeepGamer/logfilename

model.save(MODEL_NAME)

# tensorboard --logdir=foo:C:\Mayank\Coursera\Projects\DeepGamer\Learn_to_Drive_by_watching\log\nfs-run-0.001-alexnet_mdfd-8-epochs.model
