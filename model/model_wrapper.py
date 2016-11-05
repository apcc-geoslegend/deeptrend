# Calls ffnn_test.py
#Change ffnn to a library
import ffnn_lib
import csv
import os.path

"""
Retrieves hyperparams and network parameters from a config file in the /config folder
and returns them to ModelUI variabels to be used as parameters for models

Args:
  layers: An array representing number of hidden layers. [100, 100,...]
  epoch: Number of epochs to run
  learning_rate: initial learning rate
  optimizer: Type of optimizer
  classify: Boolean to run with classification or not
  test_pct: Percent of input_data to test with
  backtest_pct: Percent of input data to backtest on

TODO:
  train_data: Dataset to train on?
  test_data: Dataset to test on?
  backtest (aka validate): Dataset to backtest on?

"""

class ModelUI(object):

  def __init__(self):
  # layers, epoch, batch_size, learning_rate, optimizer, classify, test_pct, backtest_pct
    pass


  def config_execute(self):
    """
    Reads parameters from a file in the config folder
    returns a list of parameters via a list

    Returns:
      holders: A list indexed as followers: 0 = layers (list), 1 = epoch, 2 = learning_rate, 3 = optimizer,
      4 = classify, 5 = test_pct, 6 = backtest_pct
    """
    holder = []
    # Iterate through files
    for filename in os.listdir('./config'):
      if filename.endswith(".csv"):
          print(os.path.join('./config', filename))
          print(filename)
          f = open(os.path.join('./config',filename), 'r')
          reader = csv.reader(f, delimiter=',', quotechar='|')
          for row in reader: #each row list
            if(row[0]=="#"): ## means not executed, skip first line
              continue
            holder.append(row)

        #Confirming list
          print("holder list check")
          for i in holder:
            print(i)

          nn = ffnn_lib.ModelFF(holder)
          nn.run_model()
          print(nn.acc_monthly_returns)
          f.close()

          #Write back to the same file f
          fw = open(os.path.join('./config',filename), 'a')
          fw.write("\n")
          fw.write("Test results: \n")
          fw.write("Accuracy: "+str(nn.output_accuracy)+"\n")
          fw.write("Loss (0 if not ran): "+str(nn.output_loss)+"\n")
          fw.write("Accumulative Monthly Returns\n")
          for key, value in nn.acc_monthly_returns.items():
              print key
              print value
              fw.write(str(key)+": "+str(value)+"\n")

          fw.close()
          #current unexecuted file holder params
          #NOTE initialize ModelFF object with params list
          #NOTE execute run_model
          #NOTE run_model model should return list of outputs
          #NOTE append new line then dictionary line by line to current file
