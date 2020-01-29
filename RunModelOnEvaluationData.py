# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:35:48 2020

@author: admin
"""


import json
import numpy as np
from keras.models import load_model


model = load_model('model_77acc.h5py')

Ifv= np.load('ImageFeatures_validation_VQA2_inceptionV3.npy')
Que = np.load('tokenized_val_questions.npy')
ans = json.load(open('val_ans_tokens.json','r'))

ans_to_cat = np.zeros((len(Que),1000))
for i in range(len(Que)):
    ans_to_cat[i][ans[i]]=1

x=[Ifv,Que]
ans_idx= json.load(open('ans_index.json','r'))

data = json.load(open('validation_filtered.json','r'))
t = model.predict(x)
t= list(t)


results =[]
predicted_classes=[]


for i in range(len(x[0])):
    
    predicted_classes = list(t[i]).index(max(list(t[i])))
    results.append({
            'answer'     :ans_idx[str(predicted_classes)],
            'question_id':int(data[i]['question_id']),
            'gt_ans'     :data[i]['ans']
            })
    
json.dump(results,open('Results/v2_OpenEnded_mscoco_val2014_val_results.json','w'))