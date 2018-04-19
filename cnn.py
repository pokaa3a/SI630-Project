import numpy as np
import tensorflow as tf
import random
import scipy.misc
import os
import time

def init_weights(shape):
	init_random_dist = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_random_dist)

def init_bias(shape):
	init_bias_vals = tf.constant(0.1, shape=shape)
	return tf.Variable(init_bias_vals)

def conv2d(x,W):
	# x --> [batch,H,W,C]
	# W --> [filter H, filter W, Channels In, Channels Out]
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_to_1(x):
	# shape --> [batch,H,W,C]
	return tf.nn.max_pool(x,ksize=[1,x.shape[1],x.shape[2],1],strides=[1,1,1,1],padding='VALID')

def convolutional_layer(input_x,shape):
	W = init_weights(shape)
	b = init_bias([shape[3]])
	return tf.nn.relu(conv2d(input_x,W)+b)

def normal_full_layer(input_layer,size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size,size])
	b = init_bias([size])
	return tf.matmul(input_layer,W) + b

class CNN(object):
	def __init__(self, sess, size_H=16, size_W=300, batch_size=10):
		self.sess = sess
		self.size_H = size_H
		self.size_W = size_W
		self.batch_size = batch_size
		self.step = 0
		self.build_model()

	def build_model(self):
		# Placeholders
		self.x = tf.placeholder(tf.float32, shape=[None, self.size_H, self.size_W])
		self.y = tf.placeholder(tf.float32, shape=[None, 2])
		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		# Filters
		nFilter = 2
		x_input = tf.reshape(self.x,[-1,self.size_H,self.size_W,1])

		# length = 2
		conv2 = convolutional_layer(x_input, shape=[2, self.size_W, 1, 1])
		conv2_pool = max_pool_to_1(conv2)
		x_concat = conv2_pool
		for i in range(nFilter-1):
			conv2 = convolutional_layer(x_input, shape=[2, self.size_W, 1, 1])
			conv2_pool = max_pool_to_1(conv2)
			x_concat = tf.concat((x_concat, conv2_pool),1)

		# length = 3
		for i in range(nFilter):
			conv3 = convolutional_layer(x_input, shape=[3, self.size_W, 1, 1])
			conv3_pool = max_pool_to_1(conv3)
			x_concat = tf.concat((x_concat, conv3_pool),1)

		# length = 4
		for i in range(nFilter):
			conv4 = convolutional_layer(x_input, shape=[4, self.size_W, 1, 1])
			conv4_pool = max_pool_to_1(conv4)
			x_concat = tf.concat((x_concat, conv4_pool),1)

		concat_flat = tf.reshape(x_concat, [-1, nFilter*3])
		full_layer_one = tf.nn.relu(normal_full_layer(concat_flat, 20))

		# Dropout
		self.hold_prob = tf.placeholder(tf.float32)
		full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=self.hold_prob)
		self.y_pred = normal_full_layer(full_one_dropout, 2)

		# Softmax
		self.y_pred_prob = tf.nn.softmax(self.y_pred)

		# Loss function
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

		# Optimizer
		self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

		# Saver
		self.saver = tf.train.Saver()

	def init(self):
		init = tf.global_variables_initializer()
		self.sess.run(init)

	def train(self, batch_x, batch_y, learning_rate):
		self.sess.run(self.trainer,feed_dict={self.x:batch_x, self.y:batch_y, self.learning_rate:learning_rate, self.hold_prob:0.5})

	def test(self, x_test, y_test):
		matches = tf.equal(tf.argmax(self.y_pred,1), tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
		# print 'Accuracy:',
		return self.sess.run(accuracy, feed_dict={self.x:x_test, self.y:y_test, self.hold_prob:1.0})

	def predict(self, x):
		return self.sess.run(self.y_pred_prob, feed_dict={self.x: x, self.y: np.array([1,0]).reshape([1,2]), self.hold_prob:1.0})

	def save(self, checkpoint_dir):
		model_name = 'model'
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=0)

	def load(self, checkpoint_dir):
		# checkpoint_dir = 'checkpoint'
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False


if __name__ == '__main__':
	with tf.Session() as sess:
		model = CNN(sess)
		model.init()
		x = np.ones([10, 16, 300])
		y = np.ones([10, 2])
		model.train(x, y)
