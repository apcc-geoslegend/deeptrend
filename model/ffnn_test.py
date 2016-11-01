from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from ult.stock_data import StockData
import tensorflow as tf
import numpy

FLAGS = None

def data_type():
  return tf.float32

def main(_):
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  db = StockData()
  db.readDataSet("../pdata/", test_precentage = 0.4, backtest_precentage = 0.03)

  input_size = db.getInputSize()
  output_size = db.getOutputSize()

  ###
  # h1_nodes_num = 100
  # h2_nodes_num = 50

  # x = tf.placeholder(tf.float32, shape=[None, input_size])

  # layers = [100,50,25,12,6]
  # weights = []
  # bias = []
  # for id,layer in enumerate(layers):
  #   if id == 0:
  #     W = tf.Variable(tf.truncated_normal([input_size, layer]))
  #     b = tf.Variable(tf.truncated_normal([layer]))
  #   elif id == len(layers):
  #     W = tf.Variable(tf.truncated_normal([layer, output_size]))
  #     b = tf.Variable(tf.truncated_normal([output_size]))
  #   else:
  #     W = tf.Variable(tf.truncated_normal([layers[id-1],layers[id]]))
  #     b = tf.Variable(tf.truncated_normal([layers[id]]))
  #   weights.append(W)
  #   bias.append(b)
  # # print(weights)
  # # print(bias)
  # for id in range(len(weights)):
  #   if id == 0:
  #     vh = tf.matmul(x, weights[id]) + bias[id]
  #   elif id == len(weights):
  #     vh = tf.matmul(vh,weights[id]) + bias[id]
  #   else:
  #     y = tf.matmul(vh, weights[id]) + bias[id]

  ####
  h1_nodes_num = 100
  h2_nodes_num = 50

  x = tf.placeholder(tf.float32, shape=[None, input_size])
  # Create the model
  W1 = tf.Variable(tf.truncated_normal([input_size, h1_nodes_num],  stddev=0.1,dtype=data_type()))
  b1 = tf.Variable(tf.truncated_normal([h1_nodes_num],              stddev=0.1,dtype=data_type()))
  W2 = tf.Variable(tf.truncated_normal([h1_nodes_num, h2_nodes_num],stddev=0.1,dtype=data_type()))
  b2 = tf.Variable(tf.truncated_normal([h2_nodes_num],              stddev=0.1,dtype=data_type()))
  Wo = tf.Variable(tf.truncated_normal([h2_nodes_num, output_size], stddev=0.1,dtype=data_type()))
  bo = tf.Variable(tf.truncated_normal([output_size],               stddev=0.1,dtype=data_type()))

  W1 = tf.nn.dropout(W1, 0.5)
  W2 = tf.nn.dropout(W2, 0.5)
  Wo = tf.nn.dropout(Wo, 0.5)

  vh1 = tf.matmul(x, W1) + b1
  vh2 = tf.matmul(vh1, W2) + b2
  y = tf.matmul(vh2, Wo) + bo

  #####

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, shape=[None, output_size])

  # use L2 loss 
  if db.classification == True:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  else:
    loss = tf.reduce_mean(tf.square(y - y_))

  # add regularizer
  regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                  tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
                  tf.nn.l2_loss(Wo) + tf.nn.l2_loss(bo)
                  )
  loss += 5e-4 * regularizers
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
  ####

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in xrange(1000):
    batch_xs, batch_ys = db.nextBatch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  test_input, test_label = db.getTestData()
  # Test trained model
  if db.classification:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    evaluation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    output = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
    print("Accuracy is %f"%output)
  else:
    evaluation = tf.reduce_mean(tf.abs(y - y_))
    output = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
    print("The mean L1 loss of test data is",output)
  backtest_data = db.getBacktestData()

  def backTest():
    acc_return = 0
    for row in range(backtest_data.shape[1]):
      input = backtest_data[:,row, 0:35].reshape(backtest_data.shape[0],35)
      output = sess.run(y, feed_dict={x:input})
      class1 = output[:,0]
      # argsort the class1
      sort_ids = numpy.argsort(class1)
      # -20: is the last 20 row in sorted id
      # col -4 is the next month return
      acc_return += numpy.sum(backtest_data[sort_ids[-20:],row,-4])
      print("Accumulated return in month %d is %f"%(row, acc_return))
    # print(output)

  backTest()


if __name__ == '__main__':
  tf.app.run()
