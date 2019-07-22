#print("unet loaded")

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import groupby
#from imageio import imread
from random import randint
#from tqdm import tqdm_notebook

from keras.models import Model
from keras.utils import Sequence
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.regularizers import l2
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
#from IPython.display import SVG
from keras.utils import multi_gpu_model
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def build_unet(shape):
    input_layer = Input(shape = shape)
    
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(input_layer)
    conv1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation = 'relu', padding = 'same')(conv5)
    
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides = (2, 2), padding = 'same')(conv5), conv4], axis = 3)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(up6)
    conv6 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same')(conv6), conv3], axis = 3)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(up7)
    conv7 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same')(conv7), conv2], axis = 3)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(up8)
    conv8 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same')(conv8), conv1], axis = 3)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(up9)
    conv9 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)
    
    return Model(input_layer, conv10)



from sklearn.utils import shuffle
#df = shuffle(df)

#import numpy as np
#import pandas as pd

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


batch_size = 10

train = pd.read_csv('/storage/research/Intern19_v2/AutomatedDetectionWSI/data/my_csv.csv')


split = int(0.90 * len(train)) #90% for training

#split into training and testing
train_files = train[0:split]
test_files  = train[split:]
#print(train_files.shape,test_files.shape)

train_generator = image_generator(train_files, batch_size = batch_size)
test_generator  = image_generator(test_files, batch_size = batch_size)

def build_callbacks():
        checkpointer = ModelCheckpoint(filepath='uunet_64_multi.h5', verbose=1, save_best_only=True, save_weights_only=False)
        red = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0001)
        csvv = CSVLogger('logger_64_multi.csv', separator=',', append=True)
        callbacks = [checkpointer,red,csvv]
        return callbacks

szz = (512,512,3)
model = build_unet(szz)


try:
    parallel_model = multi_gpu_model(model, cpu_merge=False)
    print("Training using multiple GPUs..")
except:
    parallel_model = model
    print("Training using single GPU or CPU..")

#model.compile(optimizer = Adam(lr = 1e-5), loss = 'mean_squared_error', metrics = ['accuracy'])
parallel_model.compile(optimizer = Adam(lr = 1e-5), loss = 'mean_squared_error', metrics = ['accuracy'])


#parallel_model.summary()


train_steps = len(train_files) //batch_size
test_steps = len(test_files) //batch_size
parallel_model.fit_generator(train_generator,
                    epochs = 30, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps,use_multiprocessing=True,
                    callbacks = build_callbacks(), verbose = 1)
