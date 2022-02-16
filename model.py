#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
import random
import numpy as np
from keras import optimizers
from keras.layers import LSTM
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers.wrappers import TimeDistributed
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D


# In[3]:


height = 120
width = 240
path = '../datasets/'
allfiles = glob.glob(path+'sensory/tile/*.npy')
sali = glob.glob(path+'content/saliencyImages/*.npy')
motion = glob.glob(path+'content/motionImages/*.npy')
prob = glob.glob(path+'sensory/tileProb/*.npy')


# In[4]:


def myGenerator():
    while True:
        index_list = random.sample(range(1, 30000), 5)
        alldata_x = []
        alldata_y = []
        for i in index_list:
            f = allfiles[i]
            s = f.split('_')
            saliFile = '../datasets/content/saliencyImages/'+s[0][25:]+'_saliency_'+s[2].split('.')[0]+'.npy'
            motionFile = '../datasets/content/motionImages/'+s[0][25:]+'_motion_'+s[2].split('.')[0]+'.npy'
            probFile = '../datasets/sensory/tileProb/'+s[0][25:]+'_user'+s[1][4:]+'_'+s[2].split('.')[0]+'.npy'
            a = np.load(f)
            b = np.load(saliFile)
            c = np.load(motionFile)
            d = [a, b, c]
            alldata_x.append(d)
            alldata_y.append(np.load(probFile))
        alldata_x = np.array(alldata_x)
        alldata_x = np.rollaxis(alldata_x, 1, 5)  
        #alldata_x = alldata_x.reshape((32, 30, height, width, 3))
        #alldata_x = np.swapaxes(alldata_x, 1, 4)
        alldata_y = np.array(alldata_y)
        yield alldata_x, alldata_y
# x = myGenerator()
# xtrain, ytrain = next(x)
# print('xtrain shape:',xtrain.shape)
# print('ytrain shape:',ytrain.shape)


# In[4]:


import matplotlib.pyplot as plt
plt.imshow(xtrain[0, 0][:, :, 2])


# In[12]:


print(allfiles[1])
f = allfiles[1]
s = f.split('_')
print(s)
print('../datasets/sensory/tileProb/'+s[0][25:]+'_user'+s[1][4:]+'_'+s[2].split('.')[0]+'.npy')


# In[2]:


path = '../datasets/'
videoNames = os.listdir(path+'content/saliency/')
videoNames = [i[:-13] for i in videoNames]


# In[5]:


# load the numpy arrays from saliency, motion maps and sensor data
sali = glob.glob(path+'content/saliencyImages/*.npy')
motion = glob.glob(path+'content/motionImages/*.npy')
sensory = glob.glob(path+'sensory/tile/*.npy')


# In[6]:


from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils

input_shape=(30, height, width, 3)
def mySegNet(input_shape):
    base_model  = MobileNet(input_shape=(224,224,3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs=base_model.input, outputs=x)
    
    model = Sequential();
    model.add(TimeDistributed(cnn_model, input_shape=input_shape))
    model.add(TimeDistributed(Flatten()))
    
    model.add(LSTM(200, return_sequences=True))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #print(model.summary())
    return model
    

#mySegNet(input_shape)


# In[ ]:


input_shape=(30, height, width, 3)
model = mySegNet(input_shape)
model.fit_generator(generator=myGenerator(),
                    use_multiprocessing=True,
                   steps_per_epoch=5, nb_epoch=5)


# In[5]:


base_model  = VGG16(input_shape=(224,224,3), include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
cnn_model = Model(inputs=base_model.input, outputs=x)
print(cnn_model.summary())


# In[4]:


videoNames = sorted(videoNames)[5:]
print(videoNames)
#print(sorted(sali))

for video in videoNames:
    npys = [s for s in sali if video in s]
    for npy in npys:
        data = np.load(npy)
        print(data.shape)
        break
    break


# In[16]:


# get the pre-trained VGG model
def loadVGG16Model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    #print ("Model loaded..!")
    #print (base_model.summary())
    return base_model
vgg_model = loadVGG16Model()


# In[2]:


def getBaseModel():
    #base_model  = MobileNet(input_shape=(224,224,3), include_top=False)
    #base_model  = ResNet50(input_shape=(224,224,3), include_top=False)
    base_model  = VGG16(input_shape=(224,224,3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    sgd = optimizers.SGD(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print (model.summary())
    return model
getBaseModel()


# In[ ]:


def buildModel(): 
    nFilters=32
    kernelSize=(3,3)
    poolSize=(2,2)
    batchSize=64

    model=Sequential()

    model.add(TimeDistributed(Conv2D(40, (3, 3), activation='relu'), input_shape=[224, 224, 1]))
    #model.add(TimeDistributed(Conv2D(nFilters, kernel_size = kernelSize, activation="relu"), input_shape=[1920, 3840,1]))
    model.add(TimeDistributed(Conv2D(nFilters*2, kernel_size = kernelSize, activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize)))

    model.add(TimeDistributed(Conv2D(nFilters, kernel_size = kernelSize, activation="relu")))
    model.add(TimeDistributed(Conv2D(nFilters*2, kernel_size = kernelSize, activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize)))

    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(5))
    #model.add(Dense(, input_dim=, activation='relu'))
    print(model.summary())
    
    return model
model = buildModel()


# In[ ]:




