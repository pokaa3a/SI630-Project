from utility import *
import string
import tensorflow as tf
import numpy as np
from cnn import CNN
import sys

EVAL_DIR = 'data/gold_data/'
TRIAL_DIR = 'data/trial_dir/trial_data/' 
TRAIN_DIR = 'data/train_dir/train_data/'

def train(train_sentences, trial_sentences, test_sentences, embeddings, epochs, batch_size):

	with tf.Session() as sess:
		model = CNN(sess)
		model.init()

		for e in range(epochs):
			print 'Epoch: {}'.format(e)
			nData = len(train_sentences[1])
			offset = 0

			# training
			progress = 0.0
			while offset < nData:
				size = min(nData-offset, batch_size/2)
				idx = range(offset, offset+size)
				batch_x, batch_y = indices_to_vectors(idx, train_sentences, embeddings)
				model.train(batch_x, batch_y,0.01-e*0.001)
				offset += size

				# testing
				idx = range(len(trial_sentences[1]))
				trial_x, trial_y = indices_to_vectors(idx, trial_sentences, embeddings)
				print 'Trial - ',
				model.test(trial_x, trial_y)

				idx = range(len(test_sentences[1]))
				test_x, test_y = indices_to_vectors(idx, test_sentences, embeddings)
				print 'Test - ',
				model.test(test_x, test_y)

if __name__ == '__main__':

	# fetch train/trial/evaluation data
	train_x, train_y = get_data(TRAIN_DIR)
	trial_x, trial_y = get_data(TRIAL_DIR)
	eval_x, eval_y = get_data(EVAL_DIR)

	# categorize sentence by their labels
	train_sentences = {0:[], 1:[]}
	trial_sentences = {0:[], 1:[]}
	test_sentences = {0:[], 1:[]}
	for x,y in zip(train_x, train_y):
		train_sentences[1 if y > 0 else 0].append(x)
	for x,y in zip(trial_x, trial_y):
		trial_sentences[1 if y > 0 else 0].append(x)
	for x,y in zip(eval_x, eval_y):
		test_sentences[1 if y > 0 else 0].append(x)

	# build word embedding table
	embeddings = {}
	with open('embedding_fastText.txt') as f:
		for line in f.readlines():
			word = line.split()[0]
			vector = [float(x) for x in line.split()[1:]]
			embeddings[word] = np.array(vector)

	# training
	train(train_sentences, trial_sentences, test_sentences, embeddings, 10, 20)

