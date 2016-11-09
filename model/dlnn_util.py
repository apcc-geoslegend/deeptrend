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