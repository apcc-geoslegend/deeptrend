from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
# from ult.stock_data import StockData
from ult.momentum_reader import MomentumReader
import tensorflow as tf
import numpy
import time

FLAGS = None

def data_type():
  return tf.float32

def main(_):
  layer_depth = 10
  layer_units = 100
  epoch = 20
  batch_size = 100
  buying_precentage = 0.05
  ##
  regularizers_weights = 5e-4
  start_learning_rate = 0.2
  lr_decay = False
  ##
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  db = MomentumReader("../pdata/")
  # db.read_dataset("../pdata/", classification = True, test_precentage = 0.3, backtest_precentage = 0.12)

  input_size = db.get_input_size()
  output_size = db.get_output_size()
  data_size = db.get_train_data_size()
  print("Input size is %s, Output size is %s, Trainning Data Size is %s"%(input_size,output_size,data_size))
  # print(data_size)

  max_train_steps = int(data_size/batch_size*epoch)

  ####
  layers = []
  for _ in range(layer_depth):
    layers.append(layer_units)
  weights = []
  bias = []

  x = tf.placeholder(tf.float32, shape=[None, input_size])
  weights.append(tf.Variable(tf.truncated_normal([input_size, layers[0]])))
  bias.append(tf.Variable(tf.truncated_normal([layers[0]])))
  for id,layer in enumerate(layers):
    if id == 0:
      continue
    W = tf.Variable(tf.truncated_normal([layers[id-1],layers[id]]))
    W = tf.nn.dropout(W, 0.5)
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

  global_step = tf.Variable(0, dtype=data_type())
  # add regularizer
  regularizers = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights] + [tf.nn.l2_loss(b) for b in bias])
  loss += regularizers_weights * regularizers
  
  if lr_decay:
    learning_rate = tf.train.exponential_decay(
        start_learning_rate,            # Base learning rate.
        global_step,                    # Current index into the dataset.
        max_train_steps/batch_size,     # Decay steps.
        0.96,                           # Decay rate.
        staircase=True)
  else:  
    learning_rate = tf.constant(start_learning_rate)

  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  # train_step = tf.train.AdadeltaOptimizer(learning_rate, 0.9).minimize(loss,global_step=global_step)
  # train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
  # train_step = tf.train.AdagradDAOptimizer(learning_rate, global_step=global_step).minimize(loss)
  # train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
  # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
  # train_step = tf.train.FtrlOptimizer(learning_rate).minimize(loss.global_step=global_step)
  # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss,global_step=global_step)
  # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss,global_step=global_step)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  total_start_time = time.time()
  evaluation_frequency = 100
  for _ in xrange(max_train_steps):
    batch_xs, batch_ys = db.next_batch(batch_size)
    oput,l,lr,gs = sess.run([train_step,loss,learning_rate,global_step], feed_dict={x: batch_xs, y_: batch_ys})
    if gs%evaluation_frequency == 0:
      duration = time.time()-total_start_time
      print("loss is % 2.3f, learning rate is % 2.3f, time used is % 3.3f"%(l,lr,duration))

  test_input, test_label = db.get_test_data()
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
  backtest_input, backtest_output, backtest_value = db.get_backtest_data()

  def back_test():
    acc_return = 0
    for date in range(len(backtest_input)):
      input = backtest_input[date]
      output = sess.run(y, feed_dict={x:input})
      value = backtest_value[date]
      num_stock_to_buy = int(buying_precentage*len(input))
      if num_stock_to_buy < 1:
        num_stock_to_buy = 1
      if db.classification:
        class1 = output[:,0]
        # argsort the class1
        sort_ids = numpy.argsort(class1)
        # -20: is the last 20 row in sorted id
        # col -4 is the next month return
        acc_return += numpy.sum(value[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
      else:
        sort_ids = numpy.argsort(output)
        acc_return += numpy.sum(value[sort_ids[0:num_stock_to_buy]])
      print("Accumulated return at month %d is % 3.3f%%"%(date, acc_return))
    # print(output)
    # print(backtest_data[:,row,db.y_ids])
    return acc_return

  back_test()

if __name__ == '__main__':
  tf.app.run()
