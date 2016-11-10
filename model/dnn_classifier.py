from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy

import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from util.momentum_reader import MomentumReader

# # Data sets
# IRIS_TRAINING = "../data/iris_training.csv"
# IRIS_TEST = "../data/iris_test.csv"

# # Load datasets.
# training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TRAINING,
#     target_dtype=np.int,
#     features_dtype=np.float32)
# test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TEST,
#     target_dtype=np.int,
#     features_dtype=np.float32)


# print(training_set.data.shape)
# print(training_set.target.shape)
# print(test_set.data.shape)
# print(test_set.target.shape)

db = MomentumReader("../pdata/",test_precentage=0.3,validation_precentage=0)
trX, trY = db.get_all_train_data()
vlX, vlY = db.get_validation_data()
teX, teY = db.get_test_data()
# print("trX:",trX.shape,"trY", trY.shape)
# print("teX:",teX.shape,"teY", teY.shape)
print("Number of Classes is :",db.get_number_classes())
print("Input size is ", db.get_input_size())
# print(trX.shape)
# print(trY.shape)
# print(trY)
model_dir = os.path.abspath("/tmp/stock")

# if tf.gfile.Exists(model_dir):
#   tf.gfile.DeleteRecursively(model_dir)
# tf.gfile.MakeDirs(model_dir)

# feature_columns = [tf.contrib.layers.real_valued_column("", dimension=db.get_input_size())]
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(trX)
print(feature_columns)

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[40, 20, 10],
                                            n_classes=2,
                                            dropout=0.5,
                                            activation_fn=tf.nn.sigmoid,
                                            model_dir=model_dir)

# Fit model.
print("Start to train the model")
classifier.fit(x=trX, y=trY, steps=300000, batch_size=10000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=teX,
                                     y=teY)["accuracy"]

# Backtest
backtest_input, backtest_output, backtest_value = db.get_backtest_data()
acc_return = 0
amrs = [] # acumulated montly return
for date in range(len(backtest_input)):
  input = backtest_input[date]
  num_stock_to_buy = int(0.01*len(input))
  if num_stock_to_buy < 1:
    num_stock_to_buy = 1
  output = classifier.predict_proba(x=input)
  output_list = []
  for a in output:
    output_list.append(a)
  output = numpy.array(output_list)
  bvalue = backtest_value[date]
  # print("output of backtest: ", output)

  class1 = output[:,0]
  # argsort the class1
  sort_ids = numpy.argsort(class1)
  acc_return += numpy.sum(bvalue[sort_ids[-num_stock_to_buy:]])/num_stock_to_buy
  amrs.append(acc_return)
  print("Accumulated return at month %d is % 3.3f%%"%(date, acc_return))
print('Accuracy: {0:f}'.format(accuracy_score))
print('AMRS:',amrs)
# result = {}
# result["Total Time"] = total_time_used
# result["Accuracy"] = output_accuracy
# result["AMR"] = amrs
# result["Loss"] = final_loss