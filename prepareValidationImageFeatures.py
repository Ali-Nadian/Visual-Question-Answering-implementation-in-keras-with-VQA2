# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:55:04 2020

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:50:22 2020

@author: admin
"""


import numpy as np
from keras.applications import InceptionV3
from keras.models import  Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
import json

""" define model : run this if your feature extraction from scrath """
"""
im_width = 299
im_height = 299

def conv_net2():
    deepmodel = InceptionV3(weights = 'imagenet',  input_shape=(im_width,im_height,3))
    deepmodel.layers.pop()
    deepmodel = Model(inputs = deepmodel.inputs,outputs = deepmodel.layers[-1].output)
    return deepmodel
my_model  = conv_net2()
print('model built')
print('starting feature extraction....')


# load unique path
Uniq_path_val = json.load(open('val_uniq_path.json','r'))
fvval = np.zeros((len(Uniq_path_val),2048))
count = 0
for i,QID in enumerate(Uniq_path_val):
   
    count = count + 1
    if count% 500==0:
        print('number of processed training images= '+str(count))
    im = load_img(QID,target_size=(im_height,im_width))
    im = np.asarray(im,dtype='float32')
    im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
    im=  preprocess_input(im)
    features = my_model.predict(im)
    fvval[i]= features

tem   = np.sqrt(np.sum(np.multiply(fvval, fvval), axis=1))
fvval = np.divide(fvval, np.transpose(np.tile(tem,(2048,1))))

        
np.save('VQA2_inceptionV3_val_features_normalized.npy',fvval)
"""

fvval  = np.load('VQA2_inceptionV3_val_features_normalized.npy')
val_im_pos = np.load('val_im_pos.npy')
Uniq_path_val = json.load(open('val_uniq_path.json','r'))
IMfeatures = np.zeros((len(val_im_pos),2048))

for i in range(len(Uniq_path_val)):
    temp = np.where(val_im_pos==i)
    IMfeatures[temp]=fvval[i]    
    
np.save('ImageFeatures_validation_VQA2_inceptionV3.npy',IMfeatures)