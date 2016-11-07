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

## TODO:
## Better to have a param supper class, but we can do it later
class DeepLinearNNParams():
  pass

class DeepLinearNN(object):

  """
  @phil,
  here instead of initialize params by some nested list structure,
  better to have a defined parameters class, then we can set default value in the
  param class
  """
  def __init__(self, params):
    """
    Initializes all hyper parameters and model params

    Args:
      params: A list indexed as followers: 0 = layers (list), 1 = epoch, 2 = batch_size, 3 = learning_rate, 4 = optimizer,
      5 = classify, 6 = test_pct, 7 = backtest_pct, 8 = buying_precentage
    """

    #Input parameters
    self.layers = map(int, params[0])
    print("Using layer:        ",self.layers)

    self.epoch = int(params[1][0])
    print("Using epoch:        ",self.epoch)

    self.batch_size = int(params[2][0])
    print("Batch size:         ",self.batch_size)

    self.base_learning_rate = float(params[3][0])
    print("Base Learning Rate: ", self.base_learning_rate)

    self.optimizer = params[4][0]
    print("Using Optimizer:    ", self.optimizer)

    self.classify = params[5][0]
    print("Claissify:          ", self.classify)

    self.test_pct = float(params[6][0])
    print("Test Precentage:    ", self.test_pct)

    self.backtest_pct = float(params[7][0])
    print("Backtest Precentage:", self.backtest_pct)

    self.buying_precentage = float(params[8][0])
    print("Buying precentage:  ",self.buying_precentage)

    #Output results
    self.output_accuracy = 0.0
    self.output_loss = 0.0
    self.loss = 0.0
    self.acc_monthly_returns = []

  def run_model(self):
    """
    Builds the neural network variables required and fills critical object
    variables
    """

    #epoch = 20 #NOTE
    #batch_size = 100 #NOTE

    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    db = MomentumReader("../pdata/")
    # db.readDataSet("../pdata/", self.classify, self.test_pct, self.backtest_pct)

    input_size = db.get_input_size()
    output_size = db.get_output_size()
    data_size = db.get_train_data_size()
    print("Input size is %s, Output size is %s, Trainning Data Size is %s"%(input_size,output_size,data_size))

    max_train_steps = int(data_size/self.batch_size*self.epoch)
    #layers = [100,100,100,100,100,100,100] #NOTE
    weights = []
    bias = []

    # with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    weights.append(tf.Variable(tf.truncated_normal([input_size, self.layers[0]])))
    bias.append(tf.Variable(tf.truncated_normal([self.layers[0]])))

    for id, self.layer in enumerate(self.layers):
      if id == 0:
        continue
      W = tf.Variable(tf.truncated_normal([self.layers[id-1], self.layers[id]]))
      b = tf.Variable(tf.truncated_normal([self.layers[id]]))
      weights.append(W)
      bias.append(b)

    weights.append(tf.Variable(tf.truncated_normal([self.layers[-1], output_size])))
    bias.append(tf.Variable(tf.truncated_normal([output_size])))

    # af = lambda x: tf.nn.relu(x)
    # af = lambda x: tf.nn.relu6(x)
    # af = lambda x: tf.nn.crelu(x)
    # af = lambda x: tf.nn.elu(x)
    # af = lambda x: tf.tanh(x)
    af = lambda x: tf.sigmoid(x)
    # af = lambda x: tf.nn.softplus(x)
    # af = lambda x: tf.nn.softsign(x)
    for id in range(len(weights)):
      if id == 0:
        vh = af(tf.matmul(x, weights[id]) + bias[id])
      elif id == len(weights)-1:
        y = af(tf.matmul(vh, weights[id]) + bias[id])
      else:
        vh = af(tf.matmul(vh, weights[id]) + bias[id])

    # y = tf.clip_by_value(y,1e-10,1.0)
    # TBD: whether we want to normalize the output from 0 to 1
    # y = (y - tf.reduce_min(y) + 1e-10)/(tf.reduce_max(y)-tf.reduce_min(y))
    y_ = tf.placeholder(tf.float32, shape=[None, output_size])

    # choose the loss function
    # loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    # loss = tf.reduce_mean(tf.nn.log_poisson_loss(y,y_))
    loss = tf.reduce_mean(tf.nn.l2_loss(y - y_))

    global_step = tf.Variable(0, dtype=tf.float32)
    # add regularizer
    regularizers = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights] + [tf.nn.l2_loss(b) for b in bias])
    loss += 5e-4 * regularizers

    """
    @phil,
    here learning rate decay can be another parameter
    """
    lr_decay = False
    if lr_decay:
      learning_rate = tf.train.exponential_decay( 
            self.base_learning_rate,        # Base learning rate.
            global_step,                    # Current index into the dataset.
            max_train_steps/self.batch_size,# Decay steps.
            0.96,                           # Decay rate.
            staircase=True)
    else:  
      learning_rate = tf.constant(self.base_learning_rate)

    if(self.optimizer == 'gd'):
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    elif(self.optimizer == 'ad'):
      train_step = tf.train.AdadeltaOptimizer(learning_rate, 0.9).minimize(loss,global_step=global_step)
    elif(self.optimizer == 'ag'):
      train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
    elif(self.optimizer == 'agd'):
      train_step = tf.train.AdagradDAOptimizer(learning_rate, global_step=global_step).minimize(loss)
    elif(self.optimizer == 'm'):
      train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    elif(self.opitimizer == 'ao'):
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    elif(self.optimizer == 'fo'):
      train_step = tf.train.FtrlOptimizer(learning_rate).minimize(loss, global_step=global_step)
    elif(self.optimizer == 'rpo'):
      train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss ,global_step=global_step)
    else: #NOTE Specialty optimizer to try out here
      pass

    # if want to show what devices are using, turn log_device_placement True
    config = tf.ConfigProto(log_device_placement=False)
    # this is already the default
    # config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    # if use Session() you have to put with a as_default()
    # sess = tf.Session(config=config)
    # with sess.as_default():
    evaluation_frequency = 100
    # Training
    total_start_time = time.time()
    loop_strat_time = time.time()
    tf.initialize_all_variables().run()
    for step in xrange(max_train_steps):
      batch_xs, batch_ys = db.next_batch(self.batch_size)
      opout,l,lr,gs,output = sess.run([train_step,loss,learning_rate,global_step,y], feed_dict={x: batch_xs, y_: batch_ys})
      if gs%evaluation_frequency == 0:
        duration = time.time()-total_start_time
        loop_duration = time.time() - loop_strat_time
        operation_precentage = step/max_train_steps*100
        print("loss: % 2.3f, learning rate: % 2.3f, operation precentage:% 2.2f%% loop time used:% 3.3f, total time used:% 3.3f"
          %(l,lr,operation_precentage,loop_duration,duration))
        print("output is ",output[0])
        loop_strat_time = time.time()
    print("Total Time Used For Trainning: %f"%(time.time()-total_start_time))
    # log the final loss
    self.loss = l

    # Test trained model
    test_input, test_label = db.get_test_data()
    if db.classification:
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      evaluation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      self.output_accuracy = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
      print("Accuracy is %f"%self.output_accuracy)
    else:
      evaluation = tf.reduce_mean(tf.abs(y - y_))
      self.output_loss = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
      print("The mean L1 loss of test data is %f"%self.output_loss)

    # Backtest
    backtest_input, backtest_output, backtest_value = db.get_backtest_data()
    acc_return = 0
    for date in range(len(backtest_input)):
      input = backtest_input[date]
      num_stock_to_buy = int(self.buying_precentage*len(input))
      output = sess.run(y, feed_dict={x:input})
      bvalue = backtest_value[date]
      # print("output of backtest: ", output)
      if db.classification:
        class1 = output[:,0]
        # argsort the class1
        sort_ids = numpy.argsort(class1)
        acc_return += numpy.sum(bvalue[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
        self.acc_monthly_returns.append(acc_return)
      else:
        sort_ids = numpy.argsort(output)
        acc_return += numpy.sum(bvalue[sort_ids[0:num_stock_to_buy]])/num_stock_to_buy
        self.acc_monthly_returns.append(acc_return)
      print("Accumulated return at month %d is % 3.3f%%"%(date, acc_return))
    return acc_return