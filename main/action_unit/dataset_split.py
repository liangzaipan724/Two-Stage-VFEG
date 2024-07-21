
import os

Date_LOCATION ='/data/DM'
ids = [d for d in os.listdir(Date_LOCATION)]

train_num = 32 * 200
val_num =  32 * 26

train = ids[0:0.75*len(ids))]
val = ids[int(0.75*len(ids)):int(0.9*len(ids))]
test = ids[int(0.9*len(ids)):]

import numpy as np
np.save('datasets/train.npy', np.array(train))
np.save('datasets/test.npy', np.array(test))
np.save('datasets/val.npy', np.array(val))
print "train shape: {0}, example:{1}".format(str(np.array(train).shape), train[0])
print "test shape: {0}, example:{1}".format(str(np.array(test).shape), test[0])
print "val shape: {0}, example:{1}".format(str(np.array(val).shape), val[0])
print(val[0])
