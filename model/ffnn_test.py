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
  db.readDataSet("../pdata/", classification = True, test_precentage = 0.4, backtest_precentage = 0.03)

  input_size = db.getInputSize()
  output_size = db.getOutputSize()

  ####
  layers = [100,100,100]
  weights = []
  bias = []

  x = tf.placeholder(tf.float32, shape=[None, input_size])
  weights.append(tf.Variable(tf.truncated_normal([input_size, layers[0]])))
  bias.append(tf.Variable(tf.truncated_normal([layers[0]])))
  for id,layer in enumerate(layers):
    if id == 0:
      continue
    W = tf.Variable(tf.truncated_normal([layers[id-1],layers[id]]))
    b = tf.Variable(tf.truncated_normal([layers[id]]))
    weights.append(W)
    bias.append(b)
  weights.append(tf.Variable(tf.truncated_normal([layers[-1], output_size])))
  bias.append(tf.Variable(tf.truncated_normal([output_size])))

  for id in range(len(weights)):
    if id == 0:
      vh = tf.matmul(x, weights[id]) + bias[id]
    elif id == len(weights)-1:
      y = tf.matmul(vh, weights[id]) + bias[id]
    else:
      vh = tf.matmul(vh, weights[id]) + bias[id]

  # y = tf.clip_by_value(y,1e-10,1.0)
  y = (y - tf.reduce_min(y) + 1e-10)/(tf.reduce_max(y)-tf.reduce_min(y))
  y_ = tf.placeholder(tf.float32, shape=[None, output_size])

  # use L2 loss 
  if db.classification == True:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
  else:
    loss = tf.nn.l2_loss(y - y_)

  # add regularizer
  regularizers = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights] + [tf.nn.l2_loss(b) for b in bias])
  loss += 5e-4 * regularizers
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
  # train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

  ####
  # h1_nodes_num = 100
  # h2_nodes_num = 50

  # x = tf.placeholder(tf.float32, shape=[None, input_size])
  # # Create the model
  # W1 = tf.Variable(tf.truncated_normal([input_size, h1_nodes_num],  stddev=0.1,dtype=data_type()))
  # b1 = tf.Variable(tf.truncated_normal([h1_nodes_num],              stddev=0.1,dtype=data_type()))
  # W2 = tf.Variable(tf.truncated_normal([h1_nodes_num, h2_nodes_num],stddev=0.1,dtype=data_type()))
  # b2 = tf.Variable(tf.truncated_normal([h2_nodes_num],              stddev=0.1,dtype=data_type()))
  # Wo = tf.Variable(tf.truncated_normal([h2_nodes_num, output_size], stddev=0.1,dtype=data_type()))
  # bo = tf.Variable(tf.truncated_normal([output_size],               stddev=0.1,dtype=data_type()))

  # W1 = tf.nn.dropout(W1, 0.5)
  # W2 = tf.nn.dropout(W2, 0.5)
  # Wo = tf.nn.dropout(Wo, 0.5)

  # vh1 = tf.matmul(x, W1) + b1
  # vh2 = tf.matmul(vh1, W2) + b2
  # y = tf.matmul(vh2, Wo) + bo

  # # Define loss and optimizer
  # y_ = tf.placeholder(tf.float32, shape=[None, output_size])

  # # use L2 loss 
  # if db.classification == True:
  #   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  # else:
  #   loss = tf.reduce_mean(tf.square(y - y_))

  # # add regularizer
  # regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
  #                 tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) +
  #                 tf.nn.l2_loss(Wo) + tf.nn.l2_loss(bo)
  #                 )
  # loss += 5e-4 * regularizers
  # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
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
  num_stock_to_buy = 15

  def backTest():
    acc_return = 0
    for row in range(backtest_data.shape[1]):
      input = backtest_data[:,row, db.x_ids].reshape(backtest_data.shape[0], len(db.x_ids))
      output = sess.run(y, feed_dict={x:input})
      if db.classification:
        class1 = output[:,0]
        # argsort the class1
        sort_ids = numpy.argsort(class1)
        # -20: is the last 20 row in sorted id
        # col -4 is the next month return
        acc_return += numpy.sum(backtest_data[sort_ids[-num_stock_to_buy:],row,-4])
      else:
        sort_ids = numpy.argsort(output)
        acc_return += numpy.sum(backtest_data[sort_ids[0:num_stock_to_buy], row, -4])
      print("Accumulated return in month %d is %f"%(row, acc_return))
    # print(output)
    # print(backtest_data[:,row,db.y_ids])
    return acc_return

  backTest()

if __name__ == '__main__':
  tf.app.run()
