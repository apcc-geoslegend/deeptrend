import sys
import os.path
import time
import datetime

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
  print("Saving input and results now\n")
  file_name = time.strftime("%m_%d_%Y-%H:%M:%S"+".txt")
  file_loc = os.path.join(file_path, file_name)

  writer = open(file_loc, 'w')

  #Write input parameters first
  layer_string = ','.join(map(str, params.layers))
  writer.write("--Input Parameters: ")
  writer.write("layers: %s"%layer_string+"\n")
  writer.write("epoch: %s"%str(params.epoch)+"\n")
  writer.write("batch_size: %s"%str(params.batch_size)+"\n")
  writer.write("learning_rate: %s"%str(params.learning_rate)+"\n")
  writer.write("optimizer: %s"%params.optimizer+"\n")
  writer.write("classify: %s"%str(params.classify)+"\n")
  writer.write("test_pct: %s"%str(params.test_pct)+"\n")
  writer.write("backtest_pct: %s"%str(params.backtest_pct)+"\n")
  writer.write("buying_pct: %s"%str(params.buying_pct)+"\n")
  writer.write("activation: %s"%params.activation+"\n")
  writer.write("loss_func: %s"%params.loss_func+"\n")
  writer.write("\n")
  writer.write("--Output Results:\n")
  writer.write("Runtime: %s"%str(result.total_time)+"\n")
  writer.write("Accuracy: %s"%str(result.output_accuracy)+"\n")
  #For loop of x months that's flexible if we extend from 12 months to say 24 or even less months
  for idx, val in enumerate(result.acc_monthly_returns):
    writer.write("Accumulated Return: %d : %s"%(idx, str(val)+"\n"))

  writer.close()
