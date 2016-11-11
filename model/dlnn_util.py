import sys
import os.path
import time
import datetime
import tensorflow as tf

class DeepLinearNNParams():

  def __init__(self):
    # NOTE initialize params with default values
    self.layer = [100,100,100]
    self.epoch = 100
    self.batch_size = 100
    self.base_learning_rate = 0.1
    self.optimizer = 'gd'
    self.classification = True
    self.test_pct = 0.3
    self.backtest_pct = 0.1
    self.buying_pct = 0.01
    self.loss_func = 'sigmoid'
    self.activation = 'relu'
    self.dropout = 0.5
    self.opt_dict = {
      "gd": tf.train.GradientDescentOptimizer(self.base_learning_rate),
      "add": tf.train.AdadeltaOptimizer(self.base_learning_rate, 0.9),
      "adg": tf.train.AdagradOptimizer(self.base_learning_rate),
      "mome": tf.train.MomentumOptimizer(self.base_learning_rate, 0.9),
      "adam": tf.train.AdamOptimizer(self.base_learning_rate),
      "ftrl": tf.train.FtrlOptimizer(self.base_learning_rate),
      "rms": tf.train.RMSPropOptimizer(self.base_learning_rate)
      }
    self.af_dict = {
      "relu":     tf.nn.relu,
      "relu6":    tf.nn.relu6,
      'crelu':    tf.nn.crelu,
      'elu':      tf.nn.elu,
      'tanh':     tf.tanh,
      'sigmoid':  tf.sigmoid,
      'softplus': tf.nn.softplus,
      'softsign': tf.nn.softsign
    }
    self.loss_dict = {
      'sigmoid':
        lambda y,y_: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_)),
      'softmax':
        lambda y,y_: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)),
      'sparse_softmax':
        lambda y,y_: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)),
      'log_poisson':
        lambda y,y_: tf.reduce_mean(tf.nn.log_poisson_loss(y,y_)),
      'l2':
        lambda y,y_: tf.nn.l2_loss(y - y_)/self.batch_size
    }

    if self.optimizer not in self.opt_dict:
      raise KeyError("ERROR: Can't find this optimizer %s"%self.optimizer)
    
    if self.activation not in self.af_dict:
      raise KeyError("ERROR: Can't find this activation function %s"%self.activation)

    if self.loss_func not in self.loss_dict:
      raise KeyError("ERROR: Can't find this loss function %s"%self.loss_func)

  def __str__(self):
    string = "This Parameters is:     \n"
    string += ("Layer:              %s\n"%str(self.layers))
    string += ("Epoch:              %d\n"%self.epoch)
    string += ("Batch size:         %d\n"%self.batch_size)
    string += ("Base Learning Rate: %f\n"%self.base_learning_rate)
    string += ("Optimizer:          %s\n"%str(self.optimizer))
    string += ("Claissify:          %s\n"%str(self.classify))
    string += ("Test Precentage:    %f\n"%self.test_pct)
    string += ("Backtest Precentage:%f\n"%self.backtest_pct)
    string += ("Buying precentage:  %f\n"%self.buying_pct)
    string += ("Activation set as:  %s\n"%str(self.activation))
    string += ("Loss function:      %s"%str(self.loss_func))
    # NOTE make if elif structure for activation and loss functions
    return string

  def get_optimizer(self):
    return self.opt_dict[self.optimizer]

  def get_activation_function(self):
    return self.af_dict[self.activation]

  def get_loss_function(self):
    return self.loss_dict[self.loss_func]

def save(params, result, file_path):
  """
  Takes in object of paramers, results object and file_path to write results
  from a test ran in /results directory

  Args:
    params: input parameters accessed via .item notation
    results: results parameters object from test
    path_file: complete string path to directory

  Results:
    A new file with datestamp in /results directory

  """
    
  print("Saving input and results now")
  file_name = time.strftime("%m_%d_%Y-%H:%M:%S"+".txt")
  file_loc = os.path.join(file_path, file_name)

  writer = open(file_loc, 'w')
  #Write input parameters first
  layer_string = ','.join(map(str, params.layers))
  writer.write("--Input Parameters:\n")
  writer.write("layers:        %s\n"%layer_string)
  writer.write("epoch:         %d\n"%params.epoch)
  writer.write("batch_size:    %d\n"%params.batch_size)
  writer.write("learning_rate: %f\n"%params.learning_rate)
  writer.write("optimizer:     %s\n"%params.optimizer)
  writer.write("classify:      %s\n"%str(params.classify))
  writer.write("test_pct:      %f\n"%params.test_pct)
  writer.write("backtest_pct:  %f\n"%params.backtest_pct)
  writer.write("buying_pct:    %s\n"%str(params.buying_pct))
  writer.write("activation:    %s\n"%params.activation)
  writer.write("loss_func:     %s\n"%params.loss_func)
  writer.write("\n")
  writer.write("--Output Results:\n")
  if result is None:
    writer.write("No result for this param\n")
    return None
  else:
    writer.write("Runtime:       %f\n"%result["Total Time"])
    writer.write("Accuracy:      %f\n"%result["Accuracy"])
    writer.write("Loss:          %f\n"%result["Loss"])
    #For loop of x months that's flexible if we extend from 12 months to say 24 or even less months
    for idx, val in enumerate(result["AMR"]):
      writer.write("Accumulated Return: %d : %s\n"%(idx, str(val)))