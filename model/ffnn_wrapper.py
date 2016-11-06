# Calls ffnn_test.py
#Change ffnn to a library
import ffnn
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


  def run(self,config_dir):
    """
    Reads parameters from a file in the config folder
    returns a list of parameters via a list

    Returns:
      holders: A list indexed as followers: 0 = layers (list), 1 = epoch, 2 = learning_rate, 3 = optimizer,
      4 = classify, 5 = test_pct, 6 = backtest_pct
    """
    token = []
    # Iterate through files
    path = os.path.abspath(config_dir)
    for file_name in os.listdir(path):
      if file_name.endswith(".csv"):
          print("Found Config File",os.path.join(path, file_name))
          file_path = os.path.join(path,file_name)
          write_rows = []
          lines = open(file_path, 'rt').readlines()
          read_file = open(file_path, 'rt')
          reader = csv.reader(read_file, delimiter=',')
          run = False
          for id,row in enumerate(reader): #each row list
            if id == 0:
              if row[0] == '0':
                write_rows.append("1\n")
                run = True
                continue
              else:
                print("This config has already ran")
                break
            else:
              write_rows.append(lines[id])
            if row[0] == "###":
              break
            token.append(row)
          if run:
            nn = ffnn.ModelFF(token)
            nn.run_model()
            write_file = open(file_path,'wt')
            write_rows.append("Test results: \n")
            write_rows.append("Accuracy:    %s\n"%str(nn.output_accuracy))
            write_rows.append("Final Loss:  %s\n"%str(nn.output_loss))
            write_rows.append("AMR:         "+"".join([str(x)+"," for x in nn.acc_monthly_returns]))
            # amr_row = ""
            # for x in nn.acc_monthly_returns:
            #   amr_row += str(x) + ","
            # write_rows.append(amr_row+"\n")
            for row in write_rows:
              write_file.write(row)

          #current unexecuted file holder params
          #NOTE initialize ModelFF object with params list
          #NOTE execute run_model
          #NOTE run_model model should return list of outputs
          #NOTE append new line then dictionary line by line to current file

if __name__ == '__main__':
  wrapper = ModelUI()
  wrapper.run("./config")