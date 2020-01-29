# Visual-Question-Answering-implementation-in-keras-with-VQA2

This is an implementation of visual question answering baseline method in keras. 
The VQA2 has been in this implementation.


Data collection

In order to use this repo you need to download the following files from the VQA2 web site
from https://visualqa.org/download.html download:

Text data: 

Training annotations 2017 v2.0*
Validation annotations 2017 v2.0*

Training questions 2017 v2.0*
Validation questions 2017 v2.0*
Testing questions 2017 v2.0

Images:
If your want to do everything from scratch download the Training and validation images 

If you only want to use features donwload train/validation features extracted from the InceptionV3
DOI: 10.6084/m9.figshare.11763636
------------------
Running (in the following order) :
1. run 
prepareTrainingTextData.py
The code has comments to explain everything step by step.
The code creates top answers, answer<->integer dictionary, filtered training data by top answers, answer tokens, question tokens, image path, image position 


2. run 
prepareValidaitonTextData.py

3. run
prepareTrainingImageFeatures.py (you can extract features from scratch or use pretrained features)
prepareValidationImageFeatures.py (the same as above)

-----------------------
now you have your data ready
run train.py to train your model 
run RunModelOnEvaluationData.py to get the class probability distribution over all validation images. 
run EvaluationDemoOnValidationData.py to evalute your results 
* training data were used for training and validation
* validition data were used for test



