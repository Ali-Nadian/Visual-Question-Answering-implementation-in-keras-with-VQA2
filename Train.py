# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:59:53 2020

@author: admin
"""

import json
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding,  Input, LSTM,Dropout ,Multiply
from keras.utils import plot_model
from keras.utils import to_categorical






Ifv= np.load('ImageFeatures_Train_VQA2_inceptionV3.npy')
Que = np.load('tokenized_questions.npy')
ans = np.load('ans_tokens.npy')

ans_to_cat = np.zeros((len(Que),1000))
for i in range(len(Que)):
    ans_to_cat[i][ans[i]]=1

x=[Ifv,Que]




d_rate = 0.5
Vocab_length = 12818

ImInput = Input(shape=(2048,))
ImF = Dense(1024,activation='tanh')(ImInput)

X = Input(shape=(26,))
#E =Embedding(12379,100,weights=[Embedding_matrix], input_length=26, trainable=False)(X)
E =Embedding(17965,100,input_length=26)(X)
LSTM1 = LSTM(512,return_sequences=True)(E)
LSTM1 = Dropout(0.5)(LSTM1)
LSTM1 = LSTM(512,return_sequences=False)(LSTM1)
LSTM1 = Dropout(0.5)(LSTM1)
Ques_D=Dense(1024,activation='tanh')(LSTM1)

fv = Multiply()([ImF,Ques_D])

F = Dense(1024,activation='tanh')(fv)
F = Dropout(0,5)(F)
F = Dense(1000,activation='tanh')(F)
F = Dropout(0,5)(F)
output=Dense(1000,activation='softmax')(F)
model = Model([ImInput,X],output)
model.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics = ['accuracy'])
model.summary()
plot_model(model,'my_model.png')

history = model.fit(x,ans_to_cat,batch_size=512,epochs=15,verbose=1,validation_split=.05)

