import csv
import os.path
import numpy
import copy
import random

class StockData:

	def __init__(self):
		self.all_datas = []
		self.randidx = []
		self.train_data = []
		self.test_data = []
		self.current_id = 0
		self.found_data = False
		self.x_ids = 0
		self.y_ids = 0
		self.train_size = 0

	def getData(self, file_path):
		file_path = os.path.abspath(file_path)
		with open(file_path, 'rt') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
			data = []
			for row in reader:
				data.append([float(x) for x in row])
			# data_array = numpy.array(data)
			return data

	def readDataSet(self, data_dir, test_precentage = 0.1):
		# read all data input a numpy array

		# first axis is depth which is each stock
		# second axis is row which is each month
		# third axis is col which is every input
		data_dir = os.path.abspath(data_dir)
		data_files = os.listdir(data_dir)
		
		for file in data_files:
			file_path = os.path.join(data_dir, file)
			new_data = self.getData(file_path)
			total_length = len(new_data)
			test_length = int(total_length * 0.1)
			# from 0 to (total - test) is trainning data
			for id in range(0, total_length - test_length):
				self.train_data.append(new_data[id])
			# form (total -test) to total is test data
			for id in range(total_length - test_length, total_length):
				self.test_data.append(new_data[id])

		print("Get %d tranning data, get %d testing data"%(len(self.train_data),len(self.test_data)))
		random.shuffle(self.train_data)
		self.train_data = numpy.array(self.train_data)
		self.train_size = self.train_data.shape[0]
		self.x_ids = [x for x in range(self.train_data.shape[1]-1)] 
		self.y_ids = self.train_data.shape[1]-1
		self.found_data = True

	def nextBatch(self, num):
		if self.found_data == True:
			start_id = self.current_id
			end_id = start_id + num
			self.current_id = end_id
			return self.train_data[start_id:end_id,self.x_ids], self.train_data[start_id:end_id,self.y_ids]
		else:
			print("Can't find the data, You need to Read Data First! use readDataSet()")

	def getTestData(self, num):
		return self.test_data[:,self.x_ids], self.test_data[:,self.y_ids]

if __name__ == '__main__':
	# usecase example
	data_dir = "../pdata/"
	database = StockData()
	database.readDataSet(data_dir)
	for i in range(100):
		train_batch, label_batch = database.nextBatch(100)
