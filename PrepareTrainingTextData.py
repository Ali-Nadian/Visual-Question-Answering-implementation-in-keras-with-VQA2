# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:24:48 2020

@author: Ali_Nadian
"""


import json

# enter your own path to the coco image set directory
rootImages   = 'E:/Datasets/VQA2 dataset/VQA/Images/MSCOCO'

# leave this be
imdir='%s/%s/%012d.jpg'


"""""""""""step 1: read and create your data in an appropriate format"""""""""""""""""""""""""""""""""""""""""""""

"""
# their are two files creates by the creators of the VQA dataset that contain the 
 questions : train_A in this script
 annotations: train_Q in this script
 these file are python dictionaris
 we require the train_A['questions'] field with the following keys : image_id, question, question_id
 from train_A we need to extract the answer
"""
train_A = json.load(open('TextData/v2_mscoco_train2014_annotations.json','r'))
train_Q = json.load(open('TextData/v2_OpenEnded_mscoco_train2014_questions.json','r'))

# annotation file is loaded in train_A 
# train_A has 5 fields: 'info', 'license', 'data_subtype', 'annotations', 'data_type'
# we require the annotations files which is a list of dictionaries
# the filds of each item in the list which is a dictionary has the following fields
# 'question_type', 'multiple_choice_answer', 'answers', 'image_id', 'answer_type', 'question_id'
# for VQA2 we only have the open ended task and the answers for each question are sotred in  the field: multiple_choice_answer


# questions are loaded in train_Q : 'info', 'task_type', 'data_type', 'license', 'data_subtype', 'questions'
# we are intrested in the field : questions
# questions is a list of dictionaries, where each dictionary as the following fields: 
# 'image_id', 'question', 'question_id' 
# the following files show the keys in both files

print('\n','These are the fields of the train questions dictionary:   ','\n',train_Q.keys())
print('\n','questions filed has the following keys:   ','\n', train_Q['questions'][0].keys())

# what data do we require?
    #1 questions  (this is a key in train_Q['questions'][index_to_a_row]['question'])
    #2 the answer (this is a key in train_A['annotations'][index_to_a_row]['multiple_choice_answer'])
    #3 the adress to the image that corresponds to the question (we use the image id to construct the image path)
    #4 question_id (this is key in train_A['questions'][index_to_a_row]['questions_id'])


# next we buld a list (the number of rows equals to the number of quesitons in the train_Q)
    # then for each question we extract the required data and append it in a list we call train. 
    
train =[]

for items in range(len(train_Q['questions'])):
    question    = train_Q['questions'][items]['question']
    question_id = train_Q['questions'][items]['question_id']
    img_path    = imdir%(rootImages,'train2017',train_Q['questions'][items]['image_id'])
    ans         = train_A['annotations'][items]['multiple_choice_answer']
    train.append({'question':question,'question_id':question_id,'img_path':img_path,'ans':ans})

# save your data in json format 
json.dump(train,open('train.json','w'))    

# Run this code if you want to see samples of your data
#import cv2
#import random
#import matplotlib.pyplot as plt
#
#item=random.randint(0,len(train))
#im = cv2.imread(train[item]['img_path'])
#plt.imshow(im)
#print(train[item]['question'])
#print(train[item]['ans'])
    
""""""""""""""""""""""""""""""""""""" step 2: tokenize data"""""""""""""""""""""""""""""""""""""""""""""""""
"""
the train variable we create above is a python list. 
each item of the list is a dictionary which holds: question id, question, answer, image_path

in this step we want to perform the followinf tasks

prepare answers:
An answer could be 'yes' or 'on the bench'. we want to convert them into integers so that the machine undestants it

1. Turn each anwer to an integer. For example, we want the answer 'yes' to be represented with 0 and answer 'no' to 
    be represented as 1 and so on. 

2. Keep top N repeated answers. There are many questions that are only repeated once or twice so the system wont learn 
    much from these quesionts 
    
3. Filter quesitons based on the top answers (we must omit items in the train list which have answers not included 
   in the top answers). So we first find the top answers, we then sort the answers based on the occurance frequency 
   and then assin 0 to the most frequent answer and so on.
4. we convert this data to a dictionary so later we can convert answers of validation and test sets into integers and 
   also convert answers the results of the model which are integers into string(words) so that we can evaluate our results.


prepaer questions:
questions are strings : 'what is on the bus?' This string needs to be converted into integers. 
and each words in all the questions should be represented with the same integer. 
Read more about Tokenization for further details. 

the following steps should be implemented. 
1. preprocess text data. 

2. create a Tokenizer object using keras 

3. convert qustions into tokens 
    so a questions 'what is on the table ?' is represented as '12 23 43 123 431 2'
    and question 'what seems to be is on the bed ?' is represented as     '12 888 213 5555 23 43 123 499 2'
    the largest number shows the size of the vocobluary (i.e. the number of unique words in the questions)
4. pad sequences
    now questions(sequence) might have different length. so we must convert them to sequnces with similar lentgh
    so much pad each sequence 

5. we need a dictionary to convert each integer to a word if required. this is provided by the keras tokenizer object. 

at the the end we have a list of questions that look like : 
        [0 0 0 34 12 43 2]
        [0 0 2 23 23 4 345]
        .
        .
        .
        # in this case zeros are add at the begginig of the sequence
     

prepare images:
1. create a list of unique image path. 
    we create the a list of image addresses to all images that are used

2. we also create a list of image positions. 
    the size of this list equals to the number of questions.     
    each element of the list indicates which image address belonggs to this question. 
    
    unique image list look like 
    A[0]= [d:\...\imga.jpg']
    A[1]= [d:\...\imga1.jpg']
    A[2]= [d:\...\imgd1.jpg']
    A[3]= [d:\...\img1av.jpg']
    A[4]= [d:\...\img1fs.jpg']
    .
    .
    .
    .
    
 image postion list looks liks:
     [0
     0
     0
     1
     2
     2
     .
     .
     ]
     which means the first 3 questions require image address stored in A[0]
"""
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



num_ans = 1000  # the number of answers , these are the 1000 most repeated answers

# create a dictionary ans_dict{} , loop over all answers and if answer is not in the 
# dictionary add it as a dictionary key 
# also add 1 to the value of the inserted key so you have the count of the repeated answers
# when the loop is processed , convert the dictionary into tuple (so your can sort it, note your cant sort dictionaries)
# the loop over the sorted tuple and create a list of the top answers. 
# the returned list top_ans is a list of strings


def get_top_ans(train,num_ans):
    top_ans = []
    ans_dict={}
    for items in train:
        if items['ans'] not in ans_dict:
            ans_dict[str(items['ans'])] = 1
        ans_dict[str(items['ans'])]=ans_dict.get(items['ans'])+1
    sorted_answer = sorted([(ans,i) for i,ans in ans_dict.items()],reverse=True)
    for i in range(num_ans):
        top_ans.append(sorted_answer[i][1])
    return top_ans

# go bad to vairable train, loop over all of its elements and if for an element, the answer to the question
# is in the top answers, added that list item to the tran_cleaned[] , return the cleaned data. 
def filter_data(top_ans,train):
    train_cleaned=[]
    for i in range(len(train)):
        if train[i]['ans'] in top_ans:
            train_cleaned.append(train[i])
    return train_cleaned
        

# your should also filter the annotaitons file (this step is not required unless you want to evaluate your code with evaluaion toolkit provided by the VQA team)
def filter_annotations(top_ans,ants):
    new_annotaitons = []
    for i in range(len(ants['annotations'])):
        if ants['annotations'][i]['multiple_choice_answer']  in top_ans:
            new_annotaitons.append(ants['annotations'][i])
    del ants['annotations']
    ants['annotations']= new_annotaitons
    return ants

# this file reads all filtered train data, and creates a keras tokenizer object
# fits it on the data

    
    
def create_tokenizer(data):
    question_list = []
    for i in range(len(data)):
        question = data[i]['question'].lower() # convert each quesiton to lower cases
        q=re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", question ) # convert a string into a list of strings , so: 'what is my name?' is converted to ['what','is','my','name','?']
        word_splited_question=[q[i] for i in range(len(q)-1) if q[i]!=" " ]
        data[i]['question_list']=word_splited_question # this step is not required. 
        question_list.append(word_splited_question) # this is a list of questions. 
#       question_list =   [  ['what','is','my','name','?'],
#                            ['what','is','on','table','?'],
#                            ['who','is','that','?']]
        
    tokenizer = Tokenizer(split=' ', oov_token=12380) # create a tokenizer object , oov_token:later on, when input a test question, it might have a word that was not present in the training quesitons, this will be replaced with oov_token
    tokenizer.fit_on_texts(question_list) # fit your tokenizer on the quesion_list
    return tokenizer,data
# we can now apply the tokenizer to a question list and convert the text into integers where each word is represented with a unique integer. 
    

# in the following function we loop over the filtered data

def get_im_path(data):

    img_path_unique={} # we create a dictionary that each key is the image path, 

    N = len(data)
    im_pos = np.zeros(N,dtype = 'uint32')
    k = 0 
    for i,items in enumerate(data):
        if items['img_path'] not in img_path_unique.keys():
            k=k+1 #use a counter to set aunique integer value for each image address
            img_path_unique[items['img_path']]=k
            unique_path=[Key for Key,Val in img_path_unique.items()]
            path_to_int = {w:i for i,w in enumerate(unique_path)}
    for i,items in enumerate(data): # loop over data 
        im_pos[i] = path_to_int.get(data[i]['img_path']) # assign the value of each key(image path) to the im_pos list
    return unique_path,im_pos


#################
# get the top answers
top_ans = get_top_ans(train,num_ans)

# build dictionaries to convert your answer to integers and viseversa
ans_index = {ans:key for ans,key in enumerate(top_ans)}
index_ans = {key:ans for ans,key in enumerate(top_ans)}



# clean annotations files for evaluation
ants = json.load(open('D:/python projects/VQA/annotations/v2_mscoco_train2014_annotations_original.json','r'))
ants_cleaned = filter_annotations(top_ans,ants)

# clean training data (remove unwanted questions)
train_cleaned = filter_data(top_ans,train)

# tokenize_questions
data = train_cleaned
tokenizer,data = create_tokenizer(data)

# tokenize questions of the filtered training data
ans_tokens  = [index_ans[data[i]['ans']] for i in range(len(data))]

# convert questions a list 
questions = [data[i]['question_list'] for i in range(len(data))] # create a list of questions
# tokenz the list and do padding
tokenized_questions = tokenizer.texts_to_sequences(questions) 
tokenized_questions = pad_sequences(tokenized_questions,maxlen=26,padding='pre',truncating='pre')
tokenized_questions=list(tokenized_questions) # covert to list


# create unique image path
unique_path,im_pos=get_im_path(data)

""" to do list """
"""
change the name of the saved files to reflect that they are training data. 
"""


# save data
json.dump(top_ans,open('top_ans.json','w'))  
np.save('unique_path.npy',unique_path)   # may change this to unique training image path
np.save('im_pos.npy',im_pos) # change this to training image positions. 
np.save('tokenized_questions.npy',tokenized_questions)  
np.save('top_ans.npy',top_ans)
json.dump(data,open('filtered_train.json','w'))
np.save('ans_tokens.npy',ans_tokens)
np.save('ans_index.npy',ans_index)
np.save('index_ans.npy',index_ans)
json.dump(ans_index,open('ans_index.json','w'))

import pickle
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

## save your annotations file the annotation folder which is later used in the evalutation step 
json.dump(ants_cleaned,open('annotations/v2_mscoco_train2014_annotations.json','w'))

