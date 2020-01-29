# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:41:58 2020

@author: Ali Nadian
"""



import json
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding,  Input, LSTM,Dropout ,Multiply
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.models import load_model
import h5py


model = load_model('model_77acc.h5py')

Ifv= np.load('ImageFeatures_Train_VQA2_inceptionV3.npy')
Que = np.load('tokenized_questions.npy')
ans = np.load('ans_tokens.npy')

ans_to_cat = np.zeros((len(Que),1000))
for i in range(len(Que)):
    ans_to_cat[i][ans[i]]=1

x=[Ifv,Que]
ans_idx= json.load(open('ans_index.json','r'))

data = json.load(open('validation_filtered.json','r'))
t = model.predict(x)
t= list(t)


Valresults =[]
predicted_classes=[]


for i in range(len(x[0])):
    
    predicted_classes = list(t[i]).index(max(list(t[i])))
    Valresults.append({
            'answer'     :ans_idx[str(predicted_classes)],
            'question_id':int(data[i]['question_id']),
            'gt_ans'     :data[i]['ans']
            })
    
json.dump(Valresults,open('Results\v2_OpenEnded_mscoco_val2014_val_results.json','w'))
  


