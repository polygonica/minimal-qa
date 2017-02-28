# Works with Tensorflow release 1.0
# cut down, fixed, and adapted nivwusquorum's (more commented) code at https://gist.github.com/nivwusquorum/b18ce332bde37e156034e5d3f60f8a23

# adapt this file to operate on doc2vec vectors

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import argparse
import os.path

################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 2       # 2 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)

cell = tf.contrib.rnn.LSTMCell(RNN_HIDDEN)

# Create initial state.
batch_size    = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))

# a function for getting prediction values out of the network, added by Matt (TODO: is this correct?)
normalized_outputs = tf.cast(tf.abs(predicted_outputs) > 0.5, tf.int32)

################################################################################
##                           ARGPARSE                                         ##
################################################################################

argparser = argparse.ArgumentParser(description='Trains an LSTM if necessary, then takes two numbers and outputs sum with NN')
argparser.add_argument('--retrain',action='store_true',help='force retraining of the LSTM')
argparser.add_argument('-a',type=int,help='first number')
argparser.add_argument('-b',type=int,help='second number')
args = argparser.parse_args()

################################################################################
##                         TRAINING/RESTORING THE MODEL                       ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

sess = tf.Session()

saver = tf.train.Saver()
checkpointpath = "/home/matth/Desktop/ross/minimal-qa/lstmmodel.ckpt"

if (not args.retrain) and (os.path.isfile("checkpoint")): #TODO: file check necessary?
	# restore model
	saver.restore(sess, checkpointpath)
	print("Model restored from file: %s" % checkpointpath)
else:
	# For some reason it is our job to do this:
	sess.run(tf.initialize_all_variables())
	# train model
	for epoch in range(15):
		epoch_error = 0
		for _ in range(ITERATIONS_PER_EPOCH):
			# here train_fn is what triggers backprop. error and accuracy on their
			# own do not trigger the backprop.
			x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
			epoch_error += sess.run([error, train_fn], {
				inputs: x,
				outputs: y,
			})[0]
		epoch_error /= ITERATIONS_PER_EPOCH
		valid_accuracy = sess.run(accuracy, {
			inputs:  valid_x,
			outputs: valid_y,
		})
		print "Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)

	# save model
	save_path = saver.save(sess, checkpointpath)
	print("Model saved in file: %s" % save_path)

################################################################################
##                           USING THE MODEL                                  ##
################################################################################

# process input: sum, wrangle everything into byte form and pretty-printed byte form, then tell the user
a = args.a
abytes = as_bytes(a, NUM_BITS)
abytestext = ''.join([str(i) for i in abytes])
b = args.b
bbytes = as_bytes(b, NUM_BITS)
bbytestext = ''.join([str(i) for i in bbytes])
c = a + b
cbytes = as_bytes(c, NUM_BITS)
cbytestext = ''.join([str(i) for i in cbytes])
print("A, B and A + B are %d, %d, and %d" % (a, b, c))
print("A bytes and B bytes and A + B bytes are %s and %s and %s" % (abytestext, bbytestext, cbytestext))

# reshape the data so it's ready for the model (right now I'm doing the dumb thing and running two copies of the same data through the network to demonstrate dimensionality)
input_data_1 = np.concatenate((np.reshape(abytes,(-1,1,1)), np.reshape(bbytes,(-1,1,1))), axis=2) # a pair of numbers to add together
input_data_2 = np.concatenate((np.reshape(abytes,(-1,1,1)), np.reshape(bbytes,(-1,1,1))), axis=2) # another (identical) pair of numbers to add together
input_data = np.concatenate((input_data_1, input_data_2), axis=1)

# run model to get predictions
output_data = sess.run(normalized_outputs, feed_dict={inputs: input_data})

# print inputs and predictions
# first pair
print(input_data[:,0,0])
print(input_data[:,0,1])
# first sum
print(output_data[:,0,0])
# second pair
print(input_data[:,1,0])
print(input_data[:,1,1])
# second sum
print(output_data[:,1,0])

# rnn outputs seem to be the hidden layer, predicted outputs seem to be softmax pooled
print(sess.run(rnn_outputs, feed_dict={inputs: input_data}))
print(sess.run(predicted_outputs, feed_dict={inputs: input_data}))