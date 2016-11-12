import sys
import os.path
import time
import datetime
import tensorflow as tf
import yaml

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
    self.run = False
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
    if self.optimizer not in self.opt_dict:
      raise KeyError("ERROR: Can't find this optimizer %s"%self.optimizer)
    return self.opt_dict[self.optimizer]

  def get_activation_function(self):
    if self.activation not in self.af_dict:
      raise KeyError("ERROR: Can't find this activation function %s"%self.activation)
    return self.af_dict[self.activation]

  def get_loss_function(self):
    if self.loss_func not in self.loss_dict:
      raise KeyError("ERROR: Can't find this loss function %s"%self.loss_func)
    return self.loss_dict[self.loss_func]

  def load_yaml(self, file_path):
    file = open(file_path, 'rt')
    yaml_dict = yaml.load(file)
    self.optimizer = yaml_dict['optimizer']
    self.layers = yaml_dict['layers']
    self.run = yaml_dict['run']
    self.classification = yaml_dict['classification']
    self.activation = yaml_dict['activation']
    self.epochs = yaml_dict['epochs']
    self.buying_pct = yaml_dict['buying_pct']
    self.test_pct = yaml_dict['test_pct']
    self.backtest_pct = yaml_dict['backtest_pct']

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
  writer.write("layers:             %s\n"%layer_string)
  writer.write("epoch:              %d\n"%params.epoch)
  writer.write("batch_size:         %d\n"%params.batch_size)
  writer.write("base_learning_rate: %f\n"%params.base_learning_rate)
  writer.write("optimizer:          %s\n"%params.optimizer)
  writer.write("classify:           %s\n"%str(params.classify))
  writer.write("test_pct:           %f\n"%params.test_pct)
  writer.write("backtest_pct:       %f\n"%params.backtest_pct)
  writer.write("buying_pct:         %s\n"%str(params.buying_pct))
  writer.write("activation:         %s\n"%params.activation)
  writer.write("loss_func:          %s\n"%params.loss_func)
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

def run(self,config_dir,re_run=True):
  """
  Reads parameters from a file in the config folder
  returns a list of parameters via a list

  Returns:
    holders: A list indexed as followers: 0 = layers (list), 1 = epoch, 2 = learning_rate, 3 = optimizer,
    4 = classify, 5 = test_pct, 6 = backtest_pct
  """
  nn = NN.DeepLinearNN()
  # Iterate through files
  path = os.path.abspath(config_dir)
  for file_name in os.listdir(path):
    if file_name.endswith(".yaml"):
        print("Found Config File",os.path.join(path, file_name))
        file_path = os.path.join(path, file_name)
        params = DeepLinearNNParams()
        params.load_yaml(file_path)
        if (not re_run) and params.run = True:
          print("%s config has already ran"%file_path)
          continue
        result = nn.run_model(params)
        with open(file_path, "a") as yaml_file:
          yaml.dump(result, yaml_file)