# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:33:00 2020

@author: Ali Nadian
"""


import numpy as np
from keras.applications import InceptionV3
from keras.models import  Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img


""""" run this segment if you want to feature extract """


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

""" upload the files that are required """"
""" We need two things
1. A list that contains image paths
2. A list that contains the image positions. 

"""
# this is the unique image path data 
Uniq_path_Train= np.load('unique_path.npy')


""" the following part does feature extraction and save the features for each image, 
# if you wish to do feature extraciton your self, you have to run the following part

fvtrain = np.zeros((len(Uniq_path_Train),2048))
count = 0
for i,QID in enumerate(Uniq_path_Train):
   
    count = count + 1
    if count% 500==0:
        print('number of processed training images= '+str(count))
    im = load_img(QID,target_size=(im_height,im_width))
    im = np.asarray(im,dtype='float32')
    im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2])
    im=  preprocess_input(im)
    features = my_model.predict(im)
    fvtrain[i]= features

tem = np.sqrt(np.sum(np.multiply(fvtrain, fvtrain), axis=1))
fvtrain = np.divide(fvtrain, np.transpose(np.tile(tem,(2048,1))))
       
np.save('VQA2_inceptionV3_train_features.npy',fvtrain)

"""
""" run this section if your want to create the large list of image features ready for the training algorithm """
#it loads the features that were created in the above region and creates a large list of iamge feature relavant to each question
# if your have the features just run this part. 

# these are the image position list/ 
im_pos =np.load('im_pos.npy')
fvtrain= np.load('VQA2_inceptionV3_train_features.npy')
Uniq_path_Train= np.load('unique_path.npy')

IMfeatures = np.zeros((len(Uniq_path_Train),2048))

for i in range(82575):
    temp = np.where(im_pos==i)
    IMfeatures[temp]=fvtrain[i]    
    
np.save('ImageFeatures_Train_VQA2_inceptionV3.npy',IMfeatures)




