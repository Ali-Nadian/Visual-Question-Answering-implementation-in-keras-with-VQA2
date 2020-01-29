# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 19:44:15 2020

@author: admin
"""

    # -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:37:36 2020

@author: Ali Nadian
"""
# The process performed for training data should be repeated. 
# we use the tokenizer, and top_ans from the training set to keep the consistency between data


import json


"""   create the validation data with fields : question, answer, image_path, question_id   """
    
valid_Q = json.load(open('TextData/v2_OpenEnded_mscoco_val2014_questions.json','r'))
valid_A = json.load(open('TextData/v2_mscoco_val2014_annotations.json','r'))
rootImagesVal= 'E:/Datasets/VQA2 dataset/VQA/Images/MSCOCO/val2014'

validation=[]

imdir1='%s/COCO_%s_%012d.jpg'

for items in range(len(valid_Q['questions'])):
    question    = valid_Q['questions'][items]['question']
    question_id = valid_Q['questions'][items]['question_id']
#    img_path    = imdir1%(rootImages,'val2014',valid_Q['questions'][items]['image_id'])
    img_path    = imdir1%(rootImagesVal,'val2014',valid_A['annotations'][items]['image_id'])
    ans         = valid_A['annotations'][items]['multiple_choice_answer']
    validation.append({'question':question,'question_id':question_id,'img_path':img_path,'ans':ans})

json.dump(validation,open('validation.json','w'))    


""" all the steps shold be repeated here """"""
step 1: filter validation data with top answers 
step 2: filter validation annotation 
step 3: tokenize quesiont and svae a list of tokenized questions with propper padding
step 4: create a unique list of image and the image postion list
"""


import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences


#val      = json.load(open('validation.json','r'))
ans_idx  = json.load(open('ans_index.json','r'))
top_ans  = json.load(open('top_ans.json','r'))
#val_ants = json.load(open('textdata/v2_mscoco_val2014_annotations_original.json','r'))

""" functions """
#these function are copied from prepare question text files
def filter_data(top_ans,train):
    train_cleaned=[]
    for i in range(len(train)):
        if train[i]['ans'] in top_ans:
            train_cleaned.append(train[i])
    return train_cleaned

def filter_annotations(top_ans,ants):
    new_annotaitons = []
    for i in range(len(ants['annotations'])):
        if ants['annotations'][i]['multiple_choice_answer']  in top_ans:
            new_annotaitons.append(ants['annotations'][i])
    del ants['annotations']
    ants['annotations']= new_annotaitons
    return ants

def get_im_path(data):
    img_path_unique={}
    N = len(data)
    im_pos = np.zeros(N,dtype = 'uint32')
    k = 0 
    for i,items in enumerate(data):
        if items['img_path'] not in img_path_unique.keys():
            k=k+1
            img_path_unique[items['img_path']]=k
            unique_path=[Key for Key,Val in img_path_unique.items()]
            path_to_int = {w:i for i,w in enumerate(unique_path)}
    for i,items in enumerate(data):
        im_pos[i] = path_to_int.get(data[i]['img_path'])
    return unique_path,im_pos

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# filter question with top answers
filter_validation_questions = filter_data(top_ans,validation)

# filter annotaiton files
filter_ants = filter_annotations(top_ans,valid_A)

# tokenize questions using the tokenizer
data = filter_validation_questions
import pickle   # load the tokenizer

  # import tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

val_questions = [data[i]['question'] for i in range(len(data))] # extract the list of validaiton questions

tokenized_val_questions = tokenizer.texts_to_sequences(val_questions)
tokenized_val_questions = pad_sequences(tokenized_val_questions,maxlen=26,padding='pre',truncating='pre')
tokenized_val_questions=list(tokenized_val_questions)
   

# do image path
unique_path,im_pos= get_im_path(data)
val_uniq_path = unique_path
val_im_pos = im_pos

# extract answers

index_ans = {key:ans for ans,key in enumerate(top_ans)}
val_ans = [data[i]['ans'] for i in range(len(data))]
val_ans_tokens = [index_ans[data[i]['ans']] for i in range(len(data))] 


# extract embedding matrix
json.dump(filter_ants,open('annotations/v2_mscoco_val2014_annotations.json','w'))
json.dump(val_ans_tokens,open('val_ans_tokens.json','w'))
np.save('val_im_pos.npy',val_im_pos)
np.save('tokenized_val_questions.npy',tokenized_val_questions)
json.dump(val_uniq_path,open('val_uniq_path.json','w'))
json.dump(filter_validation_questions,open('validation_filtered.json','w'))

