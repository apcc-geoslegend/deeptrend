from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from ult.stock_data import StockData
import tensorflow as tf

FLAGS = None

def main(_):
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  db = StockData()
  db.readDataSet("../pdata/", 0.2)

  input_size = db.getInputSize()
  output_size = db.getOutputSize()

  h1_nodes_num = 100
  h2_nodes_num = 50

  # Create the model
  x = tf.placeholder(tf.float32, shape=[None, input_size])
  W1 = tf.Variable(tf.zeros([input_size, h1_nodes_num]))
  b1 = tf.Variable(tf.zeros([h1_nodes_num]))
  W2 = tf.Variable(tf.zeros([h1_nodes_num, h2_nodes_num]))
  b2 = tf.Variable(tf.zeros([h2_nodes_num]))
  Wo = tf.Variable(tf.zeros([h2_nodes_num, output_size]))
  bo = tf.Variable(tf.zeros([output_size]))

  vh1 = tf.matmul(x, W1) + b1
  vh2 = tf.matmul(vh1, W2) + b2
  y = tf.matmul(vh2, Wo) + bo

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, shape=[None, output_size])

  # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  # use L2 loss 
  l2_loss = tf.reduce_mean(tf.square(y - y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(l2_loss)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = db.nextBatch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_input, test_label = db.getTestData()
  l1_loss = tf.reduce_mean(tf.abs(y - y_))
  output = sess.run(l1_loss, feed_dict={x: test_input, y_:test_label})
  print("The mean L1 loss of test data is",output)

if __name__ == '__main__':
  tf.app.run()
