from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import os.path
import random as ra
import cv2
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder


def load_data(n, vocab_size, sequence_length, tokenizer=None):
	questions_path = path + '/Quest_Answers.json'
	images_path = path + '/images/' 

	x_text = []     # List of questions
	x_image = []    # List of images
	y = []          # List of answers
	num_labels = 0  # Current number of labels, used to create index mapping
	labels = {}     # Dictionary mapping of ints to labels
	images = {}     # Dictionary of images, to minimize number of imread ops
	target_height, target_width = 96, 96
	print('Loading data...')

 
	with open(questions_path) as f:
		data = json.load(f)

	data = data['quest_answers'][0:n]
	for i in range(len(data)):
		if (data[i]['Answer'] is False):
			data[i]['Answer'] = 'False'
		if (data[i]['Answer'] is True):
			data[i]['Answer'] = 'True'
	print('JSON subset saved to file...')

	print('Storing image data...')

	for q in data[0:n]:
		if not q['Answer'] in labels :
			labels[q['Answer']] = num_labels
			num_labels += 1
		

		# Create an index for each image
		if not q['Image'] in images:
			images[q['Image']] = imread(images_path + q['Image'] + '.png', pilmode='RGB')
			images[q['Image']] = cv2.resize(images[q['Image']], (target_height, target_width))

		x_text.append(q['Question'])
		x_image.append(images[q['Image']])
		y.append(labels[q['Answer']])

	# Convert question corpus into sequential encoding for LSTM
	print('Processing text data...')
	
	tokenizer = Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(x_text)
	sequences = tokenizer.texts_to_sequences(x_text)
	x_text = sequence.pad_sequences(sequences, maxlen=sequence_length)

	# Convert x_image to np array
	x_image = np.array(x_image)
	
	# Convert labels to categorical labels
	y = keras.utils.to_categorical(y, num_labels)
	print(type(x_image))
	print(len(data))
	print('Text: ', x_text.shape)
	print('Image: ', x_image.shape)
	print('Labels: ', y.shape)
	print(num_labels)
	return ([x_text, x_image], y), num_labels, tokenizer