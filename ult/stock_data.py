import csv
import os.path
import numpy
import copy
import random

class StockData:

	def __init__(self):
		self.all_datas = []
		self.randidx = []
		self.train_data = None
		self.test_data = None
		self.backtest_data = None
		self.current_id = 0
		self.found_data = False
		self.x_ids = []
		self.y_ids = []
		self.train_size = 0
		self.test_precentage = 0
		self.classification = False

	def getInputSize(self):
		if self.found_data == True:
			return len(self.x_ids)
	
	def getOutputSize(self):
		if self.found_data == True:
			if type(self.y_ids) == type(int(0)):
				return 1
			else:
				return len(self.y_ids)

	def getTrainDataSize(self):
		if self.found_data == True:
			return self.train_data.shape[0]

	def getTestDataSize(self):
		if self.found_data == True:
			return self.test_data.shape[0]

	def setClassification(self, classification):
		self.classification = classification
		if classification:
			# it's a classification lable
			self.x_ids = [x for x in range(self.train_data.shape[1] - 2)]
			self.y_ids = [self.train_data.shape[1]-2, self.train_data.shape[1]-1]
			self.classification = True
		else:
			# this is a normalized montly return value
			self.x_ids = [x for x in range(self.train_data.shape[1]-1)]
			self.y_ids = [self.train_data.shape[1]-3]

	def getData(self, file_path):
		file_path = os.path.abspath(file_path)
		with open(file_path, 'rt') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
			data = []
			for row in reader:
				data.append([float(x) for x in row])
			# data_array = numpy.array(data)
			return data

	def readDataSet(self, data_dir, classification = True, test_precentage = 0.1, backtest_precentage = 0.1):
		file_path = os.path.join(data_dir,"stock.db")
		with open(file_path,'rb') as bfile:
			read_datas = numpy.load(bfile)
			print("Read the Database has shape",read_datas.shape)
		
		# read all data input a numpy array
		self.test_precentage = test_precentage
		self.backtest_precentage = backtest_precentage
		self.classification = classification
		#
		total_month = read_datas.shape[1]
		test_start_id = int(total_month*(1 - test_precentage - backtest_precentage))
		backtest_start_id = int(total_month*(1 - backtest_precentage))
		print("total month", total_month, "teast start id", test_start_id, "backtest start id", backtest_start_id)

		# first axis is depth which is each stock
		# second axis is row which is each month
		# third axis is col which is every input
		# [0...(train_data)...test_start_id...(test_data)...backtest_start_id...(backtest_data)...end]
		train_data = read_datas[:,0:test_start_id,:]
		test_data = read_datas[:,test_start_id:backtest_start_id,:]
		backtest_data = read_datas[:,backtest_start_id:total_month,:]
		# reshape the data 
		train_data = numpy.reshape(train_data, (train_data.shape[0]*train_data.shape[1],train_data.shape[2]))
		test_data = numpy.reshape(test_data, (test_data.shape[0]*test_data.shape[1],test_data.shape[2]))

		print("Get %d tranning data, get %d testing data, %d backtest data"%
			(train_data.shape[0],test_data.shape[0],backtest_data.shape[1]) )

		numpy.random.shuffle(train_data)
		self.train_data = train_data
		self.test_data = test_data
		self.backtest_data = backtest_data

		self.train_size = self.train_data.shape[0]
		if classification:
			# it's a classification lable
			self.x_ids = [x for x in range(self.train_data.shape[1] - 2)]
			self.y_ids = [self.train_data.shape[1]-2, self.train_data.shape[1]-1]
			self.classification = True
		else:
			# this is a normalized montly return value
			self.x_ids = [x for x in range(self.train_data.shape[1]-1)]
			self.y_ids = [self.train_data.shape[1]-3]
		self.found_data = True

	def nextBatch(self, num):
		if self.found_data == True:
			start_id = self.current_id
			end_id = start_id + num
			if end_id >= self.train_size:
				start_id = start_id - self.train_size
				end_id = end_id - self.train_size
			self.current_id = end_id
			return self.train_data[start_id:end_id, self.x_ids], self.train_data[start_id:end_id, self.y_ids]
		else:
			print("Can't find the data, You need to Read Data First! use readDataSet()")

	def getTestData(self):
		return self.test_data[:,self.x_ids], self.test_data[:,self.y_ids]

	def getBacktestData(self):
		return self.backtest_data

	def parseInputOutput(self,data):
		return data[:,self.x_ids], self.test_data[:,self.y_ids]

if __name__ == '__main__':
	# usecase example
	data_dir = "../pdata/"
	database = StockData()
	database.readDataSet(data_dir)
	database.setClassification(True)
	for i in range(1000):
		train_batch, label_batch = database.nextBatch(100)

	test_data, test_lable = database.getTestData()
	print(train_batch.shape)
	print(label_batch.shape)
	print(test_data.shape)
	print(test_lable.shape)
	# print(database.getBacktestData())

