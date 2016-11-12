import cPickle as pickle
import os.path
import numpy
import random
import data_reader

DATABASE_NAME = data_reader.DATABASE_NAME
DATABASE_PATH = os.path.abspath("../pdata/%s.db"%DATABASE_NAME)
class MomentumReader():
	"""
	This class is ussed for reading the momentum database
	"""

	def data_type(self):
		return numpy.float32

	def __init__(self, dir=DATABASE_PATH, classification = True, test_precentage = 0.1, validation_precentage = 0.1, backtest_precentage = 0.1, hot_vector = False):
		print("Loading Database %s, Please Wait"%DATABASE_NAME)
		db_path = os.path.abspath(dir)
		# if not os.path.exists(db_path):
		# 	momentum_transformer.generate_database(db_path)
		file = open(db_path,'rb')
		db = pickle.load(file)
		self.current_id = 0
		self.classification = classification
		self.all_dates = []
		self.all_stocks = []
		for date in db:
			if date not in self.all_dates:
				self.all_dates.append(date)
			for stock in db[date]:
				if stock not in db[date][stock]:
					self.all_stocks.append(stock)

		self.input_token = "Input"
		if classification:
			self.output_token = "Class"
			self.num_classes = len(db[self.all_dates[0]][self.all_stocks[0]][self.output_token])
			if not hot_vector:
				self.output_size = 1
			else:
				self.output_size = self.num_classes
		else:
			self.output_token = "NNMR"
			self.num_classes = 1
			self.output_size = 1
		self.input_size = len(db[self.all_dates[0]][self.all_stocks[0]][self.input_token])

		# validation_start_id = int(len(self.all_dates)*(1 - test_precentage - backtest_precentage - validation_precentage))
		test_start_id = int(len(self.all_dates)*(1 - test_precentage - backtest_precentage)) 
		backtest_start_id = int(len(self.all_dates)*(1 - backtest_precentage))
		self.train_input = []
		self.train_output = []
		self.test_input = []
		self.test_output = []
		self.backtest_input = []
		self.backtest_output = []
		self.backtest_value = []
		for id,date in enumerate(db):
			test = False
			backtest = False
			if id >= test_start_id and id < backtest_start_id:
				test = True
			if id >= backtest_start_id:
				backtest = True
				self.backtest_input.append([])
				self.backtest_output.append([])
				self.backtest_value.append([])
			for stock in db[date]:
				ainput = db[date][stock][self.input_token]
				aoutput = db[date][stock][self.output_token]
				if not hot_vector and classification:
					if aoutput[0] == 1:
						newop = 0
					else:
						newop = 1
					aoutput = newop
				avalue = db[date][stock]["NMR"]
				if test:
					self.test_input.append(ainput)
					self.test_output.append(aoutput)
				if backtest:
					self.backtest_input[-1].append(ainput)
					self.backtest_output[-1].append(aoutput)
					self.backtest_value[-1].append(avalue)
				if (not test) and (not backtest):
					self.train_input.append(ainput)
					self.train_output.append(aoutput)
			if backtest:
				self.backtest_input[-1]  = numpy.array(self.backtest_input[-1],dtype=self.data_type())
				self.backtest_output[-1] = numpy.array(self.backtest_output[-1],dtype=self.data_type())
				self.backtest_value[-1]  = numpy.array(self.backtest_value[-1],dtype=self.data_type())

		self.shuffle()
		# get validataion data
		if validation_precentage > 0:
			validation_number = int(len(self.train_input)*validation_precentage)
			self.validataion_input = self.train_input[0:validation_number]
			self.validataion_output = self.train_output[0:validation_number]
			# cut off that part of validataion data
			self.train_input = self.train_input[validation_number:]
			self.train_output = self.train_output[validation_number:]
		else:
			self.validataion_input = None
			self.validataion_output = None
		# calculate final trainning data size
		self.train_size = len(self.train_input)
		print("Loadded data, total length is",self.train_size)

	def shuffle(self):
		ids = range(len(self.train_input))
		random.shuffle(ids)
		self.train_input = [self.train_input[i] for i in ids]
		self.train_output = [self.train_output[i] for i in ids]

	def get_input_size(self):
		return self.input_size
	
	def get_number_classes(self):
		return self.num_classes

	def get_output_size(self):
		return self.output_size

	def get_stock_num(self):
		return len(self.all_stocks)

	def get_train_data_size(self):
		return len(self.train_input)

	def get_test_data_size(self):
		return len(self.test_input)

	def get_backtest_data_size(self):
		return len(self.backtest_input)

	def get_all_train_data(self):
		input = numpy.array(self.train_input,self.data_type())
		output = numpy.array(self.train_output,dtype=self.data_type())
		return input,output

	def get_validation_data(self):
		if self.validataion_input is None:
			return None,None
		input = numpy.array(self.validataion_input,self.data_type())
		output = numpy.array(self.validataion_output,dtype=self.data_type())
		return input, output

	def get_test_data(self):
		input = numpy.array(self.test_input,self.data_type())
		output = numpy.array(self.test_output,dtype=self.data_type())
		return input, output

	def get_backtest_data(self):
		input = self.backtest_input
		output = self.backtest_output
		value = self.backtest_value
		return input,output,value
		
	def next_batch(self, num):

		# self.shuffle()
		start_id = self.current_id
		end_id = start_id + num
		if end_id > self.train_size:
			self.shuffle()
			start_id = 0
			end_id = num
		self.current_id = end_id
		assert num == (end_id - start_id)
		input = numpy.array(self.train_input[start_id:end_id],self.data_type())
		output = numpy.array(self.train_output[start_id:end_id],dtype=self.data_type())
		return input, output

if __name__ == '__main__':
	momr = MomentumReader("../pdata/%s.db"%DATABASE_NAME)
	print(momr.get_test_data_size())
	print(momr.get_test_data_size())
	for i in range(1000):
		input, output = momr.next_batch(100)
		print(input.shape, output.shape)
	print(momr.get_backtest_data())