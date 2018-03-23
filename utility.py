import glob
import re
import random
import numpy as np

len_vec = 16

def filter_sentence(sentence):
	p = r"http\S+|[a-z]+|\!|\?|@\S+|#\S+"
	return [token for token in re.findall(p, sentence.lower()) if not (token[0]=='@' or token[0]=='#' or (len(token)>=len('http') and token[0:4]=='http'))]
	
def get_data(dir_name):
	files = glob.glob(dir_name + '*.tsv')
	x = []
	y = []
	for file in files:
		with open(file) as f:
			for line in f.readlines():
				filtered_line = filter_sentence(line)
				if len(filtered_line):
					x.append(filtered_line)
					y.append(float(line.split()[-1]))
	return x, y

def sentence_to_vector(sentence, embeddings):
	vector = embeddings[sentence[0]].reshape(1,300)
	count = 1
	for token in sentence[1:]:
		new_vector = embeddings[token].reshape(1,300)
		vector = np.concatenate((vector, new_vector), axis=0)
		count += 1
		if count >= len_vec:
			break
	for i in range(count,len_vec):
		new_vector = embeddings['</s>'].reshape(1,300)
		vector = np.concatenate((vector, new_vector), axis=0)
	vector = vector.reshape(1,len_vec,300)
	return vector

def indices_to_vectors(index, sentences, embeddings):
	vector_0 = sentence_to_vector(sentences[0][index[0]], embeddings)
	x = vector_0
	y = np.array([1,0]).reshape([1,2])
	vector_1 = sentence_to_vector(sentences[1][index[0]], embeddings)
	x = np.concatenate((x, vector_1), axis=0)
	y = np.concatenate((y, np.array([0,1]).reshape([1,2])), axis=0)

	for i in index[1:]:
		vector_0 = sentence_to_vector(sentences[0][i], embeddings)
		x = np.concatenate((x, vector_0), axis=0)
		y = np.concatenate((y, np.array([1,0]).reshape([1,2])), axis=0)
		vector_1 = sentence_to_vector(sentences[1][i], embeddings)
		x = np.concatenate((x, vector_1), axis=0)
		y = np.concatenate((y, np.array([0,1]).reshape([1,2])), axis=0)
	return x, y

def next_batch(sentences, embeddings, batch_size):
	random_0_idx = random.sample(range(len(sentences[0])), batch_size/2)
	random_1_idx = random.sample(range(len(sentences[1])), batch_size/2)

	vector_0 = sentence_to_vector(sentences[0][random_0_idx[0]], embeddings)
	batch_x = vector_0
	batch_y = np.array([1,0]).reshape([1,2])
	vector_1 = sentence_to_vector(sentences[1][random_1_idx[0]], embeddings)
	batch_x = np.concatenate((batch_x, vector_1), axis=0)
	batch_y = np.concatenate((batch_y, np.array([0,1]).reshape([1,2])), axis=0)

	for idx_0, idx_1 in zip(random_0_idx[1:], random_1_idx[1:]):
		vector_0 = sentence_to_vector(sentences[0][idx_0], embeddings)
		batch_x = np.concatenate((batch_x, vector_0), axis=0)
		batch_y = np.concatenate((batch_y, np.array([1,0]).reshape([1,2])), axis=0)
		vector_1 = sentence_to_vector(sentences[1][idx_1], embeddings)
		batch_x = np.concatenate((batch_x, vector_1), axis=0)
		batch_y = np.concatenate((batch_y, np.array([0,1]).reshape([1,2])), axis=0)

	return batch_x, batch_y
		

