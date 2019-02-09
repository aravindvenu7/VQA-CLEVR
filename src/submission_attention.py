from __future__ import print_function
# %matplotlib inline
import json
import os.path
import random 
import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization, Reshape, Lambda, Embedding, LSTM, Conv2D, MaxPooling2D, TimeDistributed, RepeatVector, Concatenate, Multiply, Flatten
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, TensorBoard
from scipy import ndimage, misc
from imageio import imread
from keras.utils.vis_utils import plot_model
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
# import pkg_resources
# pkg_resources.require("OneHotEncoder==0.20.0")
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd 
import pickle
from keras.models import load_model

encoder = LabelEncoder()
encoder.classes__ = np.load('Classes_Clevr.npy')
with open('tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)
  
encoder.fit(encoder.classes__)
integer_encoded = encoder.fit_transform(encoder.classes__)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

model = load_model('AttentionX.h5')#name

def load_data(sequence_length):

    questions_path = 'Questions.json'
    images_path = 'images/' 

    x_text = []     # List of questions
    x_image = []    # List of images
    num_labels = 0  # Current number of labels, used to create index mapping
    labels = {}     # Dictionary mapping of ints to labels
    images = {}     # Dictionary of images, to minimize number of imread ops
    target_height, target_width = 96, 96
    print('Loading data...')

 
    with open(questions_path) as f:
        data = json.load(f)

    data = data['questions']
    
    for q in data:

        # Create an index for each image
        if not q['Image'] in images:
            images[q['Image']] = imread(images_path + q['Image'] + '.png', pilmode='RGB')
            images[q['Image']] = cv2.resize(images[q['Image']], (target_height, target_width))

        x_text.append(q['Question'])
        x_image.append(images[q['Image']])

    # Convert question corpus into sequential encoding for LSTM    
    sequences = tokenizer.texts_to_sequences(x_text)
    x_text = sequence.pad_sequences(sequences, maxlen=sequence_length)

    # Convert x_image to np array
    x_image = np.array(x_image)
    # Convert labels to categorical labels
    
    return [x_text, x_image]

x_test = load_data(42)
results = model.predict(x_test)

preds = []
for i in range(len(results)):
  preds.append( np.argmax(results[i]))
  
preds_text = dict()
preds_text["Answer"] = []
preds_text["Index"] = []
for i in range(len(results)):
  preds_text["Answer"].append(encoder.inverse_transform([preds[i]])[0].lower())
  preds_text['Index'].append(i) 
  
ans=pd.DataFrame()
ans['Index']=preds_text['Index']
ans['Answer']=preds_text['Answer']
ans.to_csv('solution.csv',index=False)