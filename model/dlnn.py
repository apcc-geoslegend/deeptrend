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
import datetime

import dlnn_util

## TODO:
## Better to have a param supper class, but we can do it later
class DeepLinearNNParams(object):

  def __init__(self):
    # NOTE initialize params with default values
    self.layer = []
    self.epoch = 0
    self.batch_size = 0
    self.learning_rate = 0.0
    self.optimizer = ''
    self.classify = True
    self.test_pct = 0.0
    self.backtest_pct = 0.0
    self.buying_pct = 0.0
    self.loss_func = 'l2_loss'
    self.activation = 'relu'
    # NOTE make if elif structure for activation and loss functions

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
    self.layers = params.layers #map(int, params[0])
    print("Using layer:        ",self.layers)

    self.epoch = params.epoch #int(params[1][0])
    print("Using epoch:        ",self.epoch)

    self.batch_size = params.batch_size #)int(params[2][0])
    print("Batch size:         ",self.batch_size)

    self.base_learning_rate = params.learning_rate #float(params[3][0])
    print("Base Learning Rate: ", self.base_learning_rate)

    self.optimizer = params.optimizer #params[4][0]
    print("Using Optimizer:    ", self.optimizer)

    self.classify = params.classify #params[5][0]
    print("Claissify:          ", self.classify)

    self.test_pct = params.test_pct #float(params[6][0])
    print("Test Precentage:    ", self.test_pct)

    self.backtest_pct = params.backtest_pct #float(params[7][0])
    print("Backtest Precentage:", self.backtest_pct)

    self.buying_precentage = params.buying_pct #float(params[8][0])
    print("Buying precentage:  ", self.buying_precentage)

    self.activation = params.activation
    print("Activation set as:   ", self.activation)

    self.loss_func = params.loss
    print("Loss function:   ", self.loss_func)

    #Output results
    self.output_accuracy = 0.0
    self.output_loss = 0.0
    self.loss = 0.0
    self.acc_monthly_returns = []
    self.acc_return = 0.0
    self.total_time = 0.0

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

    max_train_steps = int(data_size*self.epoch/self.batch_size)
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
      W = tf.nn.dropout(W, 0.5)
      b = tf.Variable(tf.truncated_normal([self.layers[id]]))
      weights.append(W)
      bias.append(b)

    weights.append(tf.Variable(tf.truncated_normal([self.layers[-1], output_size])))
    bias.append(tf.Variable(tf.truncated_normal([output_size])))

    if(self.activation == 'relu'):
      af = lambda x: tf.nn.relu(x)
    elif(self.activation == 'relu6'):
      af = lambda x: tf.nn.relu6(x)
    elif(self.activation == 'crelu'):
      af = lambda x: tf.nn.crelu(x)
    elif(self.activation == 'elu'):
      af = lambda x: tf.nn.elu(x)
    elif(self.activation == 'tanh'):
      af = lambda x: tf.tanh(x)
    elif(self.activation == 'sigmoid'):
      af = lambda x: tf.sigmoid(x)
    elif(self.activation == 'softplus'):
      af = lambda x: tf.nn.softplus(x)
    elif(self.activation == 'softsign'):
      af = lambda x: tf.nn.softsign(x)
    else: #Used for special functions
      pass

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
    if(self.loss_func == 'clip_by_value'):
      loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    elif(self.loss_func == 'sigmoid_cross_entropy_with_logits'):
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))
    elif(self.loss_func == 'softmax_cross_entropy_with_logits'):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    elif(self.loss_func == 'sparse_softmax_cross_entropy_with_logits'):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    elif(self.loss_func == 'log_poisson_loss'):
      loss = tf.reduce_mean(tf.nn.log_poisson_loss(y,y_))
    elif(self.loss_func == 'l2_loss'):
      loss = tf.nn.l2_loss(y - y_)/self.batch_size
    else:
      pass

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
      if step%evaluation_frequency == 1:
        now = time.time()
        duration = now - total_start_time
        loop_duration = now - loop_strat_time
        operation_precentage = step/max_train_steps*100
        loop_strat_time = now
        average_loop_time = duration / step
        time_left = average_loop_time * (max_train_steps-step)
        print("loss: % 2.3f, learning rate: % 2.3f, operation precentage:% 2.2f%% loop time used:% 3.3f, total time used:% 3.3f"
          %(l,lr,operation_precentage,loop_duration,duration))
        print("Estimated time left is: % .2f mins"%(time_left/60))
        print("output sample is ",output[0])
    self.total_time = time.time()-total_start_time
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
    #acc_return = 0
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
        self.acc_return += numpy.sum(bvalue[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
        self.acc_monthly_returns.append(self.acc_return)
      else:
        sort_ids = numpy.argsort(output)
        self.acc_return += numpy.sum(bvalue[sort_ids[0:num_stock_to_buy]])/num_stock_to_buy
        self.acc_monthly_returns.append(self.acc_return)
      print("Accumulated return at month %d is % 3.3f%%"%(date, self.acc_return))
    return self.acc_return

if __name__ == '__main__':
  #
  params = DeepLinearNNParams() #Param object

  params.layers = [100, 100, 100, 100, 100]
  print(params.layers)

  params.epoch = 100
  print(params.epoch)

  params.batch_size = 1000
  print(params.batch_size)

  params.learning_rate = 0.1
  print(params.learning_rate)

  params.optimizer = 'gd'
  print(params.optimizer)

  params.classify = True
  print(params.classify)

  params.test_pct = 0.3
  print(params.test_pct)

  params.backtest_pct = 0.1
  print(params.backtest_pct)

  params.buying_pct = 0.01
  print(params.buying_pct)

  params.activation = 'relu'
  print(params.activation)

  params.loss = 'l2_loss'
  print(params.loss)

  dlnn = DeepLinearNN(params)
  results = dlnn.run_model()

  #just pass dlnn and you can access all results.
  dlnn_util.save(params, dlnn, 'results/')
