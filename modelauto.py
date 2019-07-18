#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
#import cv2
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import pandas as pd
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import scipy.misc
import numpy.random as rng
#from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
#import nibabel as nib #reading MR images
#from sklearn.cross_validation import train_test_split
import math
from sklearn.model_selection import train_test_split
#import glob
import tensorflow as tf
#from skimage.transform import resize
#import data as data



# In[ ]:


batch_size = 12
epochs = 30
inChannel = 3
x, y = 512, 512
input_img = Input(shape = (x, y, inChannel))


# In[ ]:





def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded



# In[ ]:



pat_1 = "/storage/research/Intern19_v2/AutomatedDetectionWSI/data/extract/"
pat_2 = "/storage/research/Intern19_v2/AutomatedDetectionWSI/data/extract2/"


def image_generator(files, batch_size = 32, sz = (512, 512)):
  
  while True: 
    
    #extract a random batch 
    #print(files)
    batch1 = np.random.choice(files.iloc[:,0], size = batch_size)
    batch2 = np.random.choice(files.iloc[:,1], size = batch_size)
    #print(batch1)
    
    #variables for collecting batches of inputs and outputs 
    batch_x = []
    batch_y = []
    
    
    for f in zip(batch1,batch2):
        #print(f)
        sz = (512,512)
        #get the masks. Note that masks are png files 
        mask = np.load(pat_2 +f[1][:16]+'/'+f[1])
        mask = np.resize(mask,sz)


        #preprocess the mask 
        #mask[mask >= 2] = 0 
        #mask[mask != 0 ] = 1
        
        batch_y.append(mask)
        sz = (512,512,3)
        #preprocess the raw images 
        raw = np.load(pat_1 +f[0][:10]+'/'+f[0])
        raw = np.resize(raw,sz)
        raw = np.array(raw)

        #check the number of channels because some of the images are RGBA or GRAY
        if len(raw.shape) == 2:
          raw = np.stack((raw,)*3, axis=-1)

        else:
          raw = raw[:,:,0:3]

        batch_x.append(raw)
    
   
    #preprocess a batch of images and masks 
    batch_x = np.array(batch_x)/255.
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3)

    yield (batch_x, batch_y)      

modelauto = Model(input_img, autoencoder(input_img))
#modelauto.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])


modelauto.compile(loss='mean_squared_error',optimizer='sgd',metrics=['mae', 'acc'])

modelauto.summary()
"""
train = pd.read_csv('/storage/research/Intern19_v2/AutomatedDetectionWSI/data/my_csv.csv')


split = int(0.95 * len(train)) #95% for training

#split into training and testing
train_files = train[0:split]
test_files  = train[split:]
#print(train_files.shape,test_files.shape)

train_generator = image_generator(train_files, batch_size = batch_size)
test_generator  = image_generator(test_files, batch_size = batch_size)

def build_callbacks():
        checkpointer = ModelCheckpoint(filepath='unetauto_64.h5', verbose=1, save_best_only=True, save_weights_only=True)
        red = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        csvv = CSVLogger('loggerauto_64.csv', separator=',', append=True)
        callbacks = [checkpointer,red,csvv]
        return callbacks

train_steps = len(train_files) //batch_size
test_steps = len(test_files) //batch_size
modelauto.fit_generator(train_generator, 
                    epochs = 30, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps,
                    callbacks = build_callbacks(), verbose = 1)



#autoencoder_train = modelauto.fit_generator(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)
#pred = modelauto.predict(valid_X)
"""
