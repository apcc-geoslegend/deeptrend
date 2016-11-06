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

class ModelFF(object):

  #NOTE @Di why did you make this it's own function?
  #Why not just give it tf.float32 ?
  """
  def data_type():
    return tf.float32
  """

  def __init__(self, params):
    """
    Initializes all hyper parameters and model params

    Args:
      params: A list indexed as followers: 0 = layers (list), 1 = epoch, 2 = batch_size, 3 = learning_rate, 4 = optimizer,
      5 = classify, 6 = test_pct, 7 = backtest_pct, 8 = num_stock_to_buy
    """
    #Input parameters
    self.layers = map(int, params[0])
    print(self.layers)

    self.epoch = int(params[1][0])
    print(self.epoch)

    self.batch_size = int(params[2][0])
    print(self.batch_size)

    self.learning_rate = tf.constant(float(params[3][0]))
    print(self.learning_rate)

    self.optimizer = params[4][0]
    print(self.optimizer)

    self.classify = params[5][0]
    print(self.classify)

    self.test_pct = float(params[6][0])
    print(self.test_pct)

    self.backtest_pct = float(params[7][0])
    print(self.backtest_pct)

    self.num_stock_to_buy = int(params[8][0])
    print(self.num_stock_to_buy)

    #Output results
    self.output_accuracy = 0.0
    self.output_loss = 0.0
    self.loss = 0.0
    self.acc_monthly_returns = {}

  def run_model(self):
    """
    Builds the neural network variables required and fills critical object
    variables
    """

    #epoch = 20 #NOTE
    #batch_size = 100 #NOTE

    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    db = StockData()
    db.readDataSet("../pdata/", self.classify, self.test_pct, self.backtest_pct)

    print("\n\n\n")
    print("Fetching input data size")
    input_size = db.getInputSize()
    print(input_size)

    print("Getting output size")
    output_size = db.getOutputSize()
    print("Getting training data size")
    data_size = db.getTrainDataSize()
    print(data_size)

    max_train_steps = int(data_size/self.batch_size*self.epoch)
    print("Max training steps", max_train_steps)

    print("\n\n\n")
    print("End of data fetching")
    print("\n\n\n")

    print("establishing neural net architecture\n\n")
    #layers = [100,100,100,100,100,100,100] #NOTE
    weights = []
    bias = []

    print("Creating inputs\n")
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    weights.append(tf.Variable(tf.truncated_normal([input_size, self.layers[0]])))
    bias.append(tf.Variable(tf.truncated_normal([self.layers[0]])))
    print("End of neural net creation. 33 inputs --> 100 node layers ... so 100 weights and biases\n\n")

    print("Enumerate 7 layers of 100 neurons\n")
    for id, self.layer in enumerate(self.layers):
      print("id: ", id, " layer:", self.layer)

      if id == 0:
        continue

      W = tf.Variable(tf.truncated_normal([self.layers[id-1], self.layers[id]]))
      b = tf.Variable(tf.truncated_normal([self.layers[id]]))
      print("Appending W: ", W, "\n")
      print("Appending b: ", b, "\n")

      weights.append(W)
      bias.append(b)

    weights.append(tf.Variable(tf.truncated_normal([self.layers[-1], output_size])))
    bias.append(tf.Variable(tf.truncated_normal([output_size])))

    print("connecting network layers via x*w+b")
    for id in range(len(weights)):
      if id == 0:
        vh = tf.matmul(x, weights[id]) + bias[id]
      elif id == len(weights)-1:
        y = tf.matmul(vh, weights[id]) + bias[id]
      else:
        vh = tf.matmul(vh, weights[id]) + bias[id]

    # y = tf.clip_by_value(y,1e-10,1.0)
    print("Getting y output estimates\n")
    y = (y - tf.reduce_min(y) + 1e-10)/(tf.reduce_max(y)-tf.reduce_min(y))
    print("Getting actual output targets to calculate loss against\n")
    y_ = tf.placeholder(tf.float32, shape=[None, output_size])

    # use L2 loss
    if db.classification == True:
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
      # loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    else:
      self.loss = tf.nn.l2_loss(y - y_)

    global_step = tf.Variable(0, dtype=tf.float32)
    # add regularizer
    regularizers = tf.reduce_mean([tf.nn.l2_loss(w) for w in weights] + [tf.nn.l2_loss(b) for b in bias])
    self.loss += 5e-4 * regularizers

    # learning_rate = tf.train.exponential_decay(
    #     0.01,                # Base learning rate.
    #     global_step,         # Current index into the dataset.
    #     max_train_steps,     # Decay step.
    #     0.95,                # Decay rate.
    #     staircase=True)
    #self.learning_rate = tf.constant(0.1) NOTE

    if(self.optimizer == 'gd'):
      train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
    elif(self.optimizer == 'ad'):
      train_step = tf.train.AdadeltaOptimizer(self.learning_rate, 0.9).minimize(self.loss,global_step=global_step)
    elif(self.optimizer == 'ag'):
      train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
    elif(self.optimizer == 'agd'):
      train_step = tf.train.AdagradDAOptimizer(self.learning_rate, global_step=global_step).minimize(self.loss)
    elif(self.optimizer == 'm'):
      train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=global_step)
    elif(self.opitimizer == 'ao'):
      train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
    elif(self.optimizer == 'fo'):
      train_step = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)
    elif(self.optimizer == 'rpo'):
      train_step = tf.train.RMSPropOptimizer(self.earning_rate).minimize(self.loss ,global_step=global_step)
    else: #NOTE Specialty optimizer to try out here
      pass

    sess = tf.InteractiveSession()
    # Training
    tf.initialize_all_variables().run()
    print("initailized all variables")
    for _ in xrange(max_train_steps):
      batch_xs, batch_ys = db.nextBatch(self.batch_size)
      output,lr = sess.run([train_step, self.learning_rate], feed_dict={x: batch_xs, y_: batch_ys})
      # print("learning rate is",lr)
      print("LR: %f"%lr )

    test_input, test_label = db.getTestData()
    # Test trained model

    if db.classification:
      print("entering classification")
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      evaluation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      self.output_accuracy = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
      print("Accuracy is %f"%self.output_accuracy)
    else:
      print("Entering else with mean loss")
      evaluation = tf.reduce_mean(tf.abs(y - y_))
      self.output_loss = sess.run(evaluation, feed_dict={x: test_input, y_:test_label})
      print("The mean L1 loss of test data is",self.output_loss)

    print("Backtesting\n")
    backtest_data = db.getBacktestData()

    #self.back_test(backtest_data, self.num_stock_to_buy, y, sess)
    #self.num_stock_to_buy = 150 #NOTE

    #def back_test(self, backtest_data, num_stock_to_buy, y, sess):

    """
    @Di

    I didn't want to spend too much time re-engineering your code here into another
    modular function. That can be a next steps
    """

    acc_return = 0
    for row in range(backtest_data.shape[1]):
      input = backtest_data[:,row, db.x_ids].reshape(backtest_data.shape[0], len(db.x_ids))
      output = sess.run(y, feed_dict={x:input})
      print("output of backtest: ", output)
      if db.classification:
        class1 = output[:,0]
        # argsort the class1
        sort_ids = numpy.argsort(class1)
        # -20: is the last 20 row in sorted id
        # col -4 is the next month return
        acc_return += numpy.sum(backtest_data[sort_ids[-self.num_stock_to_buy:],row,-4])

        self.acc_monthly_returns[row] = acc_return #add to returns list
        #print("Created month return dict space in 1")

      else:
        sort_ids = numpy.argsort(output)
        acc_return += numpy.sum(backtest_data[sort_ids[0:self.num_stock_to_buy], row, -4])

        self.acc_monthly_returns[row] = acc_return #add to returns list
        #print("Created month return dict space in 2")

      #print("Accumulated return in month %d is %f"%(row, acc_return))

    #print("returning all accumulative returns from acc_monthly_returns: \n\n")
    #print(self.acc_monthly_returns)

    # print(output)
    # print(backtest_data[:,row,db.y_ids])
    #return acc_return
