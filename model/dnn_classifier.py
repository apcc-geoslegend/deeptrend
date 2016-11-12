from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import sys
import os
import time
sys.path.insert(0, os.path.abspath(".."))
from util.momentum_reader import MomentumReader
import dlnn_util

def main(params):
  classification = params.classification
  db = MomentumReader(classification=classification, test_precentage=params.test_pct, validation_precentage=0, backtest_precentage=params.backtest_pct)
  trX, trY = db.get_all_train_data()
  vlX, vlY = db.get_validation_data()
  teX, teY = db.get_test_data()
  print("Number of Classes is :",db.get_number_classes())
  print("Input size is ", db.get_input_size())

  model_dir = os.path.abspath("/tmp/stock")
  optimizer = params.get_optimizer()
  af = params.get_activation_function()
  layers = params.layers

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(trX)
  if classification:
    NN = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=layers,
      n_classes=db.get_number_classes(),
      dropout=params.dropout,
      activation_fn=af,
      optimizer=optimizer,
      model_dir=model_dir)
  else:
    NN = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_columns,
      hidden_units=layers,
      dropout=params.dropout,
      activation_fn=af,
      optimizer=optimizer,
      model_dir=model_dir)

  start_time = time.time()
  # Fit model.
  print("Start to train the model")
  NN.fit(x=trX, y=trY, steps=params.epoch, batch_size=params.batch_size)

  # Evaluate accuracy.
  if classification:
    accuracy_score = NN.evaluate(x=teX, y=teY)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
  else:
    score = NN.evaluate(x=teX, y=teY)["loss"]
    print('Score: {0:f}'.format(accuracy_score))
    
  # Backtest
  backtest_input, backtest_output, backtest_value = db.get_backtest_data()
  acc_return = 0
  amrs = [] # acumulated montly return
  for date in range(len(backtest_input)):
    vinput = backtest_input[date]
    num_stock_to_buy = int(params.buying_pct*len(vinput))
    if num_stock_to_buy < 1:
      num_stock_to_buy = 1

    bvalue = backtest_value[date]
    output = NN.predict_proba(x=vinput)
    output_list = []
    for a in output:
      output_list.append(a)
    output = numpy.array(output_list)
    if classification:
      # argsort the class1
      sort_ids = numpy.argsort(output[:,0])
    else:
      sort_ids = numpy.argsort(output)
    acc_return += numpy.sum(bvalue[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
    amrs.append(acc_return)
    print("Accumulated return at month %d is % 3.3f%%"%(date, acc_return))
  print('AMRS:',amrs)
  result = {}
  result["Total Time"] = time.time()-start_time
  result["Accuracy"] = accuracy_score
  result["AMR"] = amrs
  # result["Loss"] = NN.get_variable_value('loss')

if __name__ == '__main__':
  params = dlnn_util.DeepLinearNNParams()
  params.layers = [40, 4, 50]
  params.epoch = 1
  params.batch_size = 100
  params.base_learning_rate = 0.1
  # gd add adg mome adam ftrl rms
  params.optimizer = 'add'
  params.classification = True
  params.test_pct = 0.3
  params.backtest_pct = 0.1
  params.buying_pct = 0.01
  params.activation = 'relu'
  params.dropout = 0.5
  params.loss_func = 'sigmoid'

  main(params)
