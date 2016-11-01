# this monthly transformer will transform what ever data file in ../data/
# into ../fake_data/ as csv file with the specific designed input which can be directly
# feed into neural network

import csv
import os.path
import shutil
import warnings
import numpy
import copy
import matplotlib.pyplot as plt
import random

def parseDate(str):
	if len(str)!=8:
		warnings.warn('Wrong input for parseDate, input size does not match')
		return None
	else:
		try:
			year = int(str[0:4])
			month = int(str[4:6])
			date = int(str[6:8])
			return year, month, date
		except ValueError:
			warnings.warn('Wrong input for parseDate, can not convert to int')
			return None
	# pass

def isSameMonth(date1, date2):
	for x in range(2):
		if date1[x] != date2[x]:
			return False
	return True

def isSameDate(date1, date2):
	for x in range(3):
		if date1[x] != date2[x]:
			return False
	return True

def getAMRs(month_date, monthly_database, num_month, month_id = -1):

	current_month_id = -1
	if month_id < 0:
		for id, data in enumerate(monthly_database):
			if isSameMonth(month_date, data[0]):
				current_month_id = id
				break
		if current_month_id == -1:
			print("Can't find this month in database, check if you are passing the right database")
	else:
		current_month_id = month_id

	start_month_id = current_month_id - num_month # start month is the month you track back
	base_month_id = start_month_id - 1 # base month id is the last month before we start month
	if base_month_id < 0:
		# if we don't have enough month data to trace back return None
		return None

	base_price = monthly_database[base_month_id][1] # 1 is the closed value in month database
	# print("Current month", month_date)
	amr = []
	for x in range(num_month):
		eval_month_id = start_month_id + x
		eval_price = monthly_database[eval_month_id][1]
		amr.append(calAR(base_price, eval_price))
		# print(monthly_database[eval_month_id][0])
	return amr

def getADRs(day, daily_database, num_day):
	current_day_id = -1
	for id, data in enumerate(daily_database):
		if isSameDate(day, data[0]):
			current_day_id = id
			break
	
	if current_day_id == -1:
		print("Can't find this day in database, check if you are passing the right database")
		return None

	start_day_id = current_day_id - num_day
	base_day_id = start_day_id - 1
	if base_day_id < 0:
		return None

	base_price = daily_database[base_day_id][2] # 2 is the closed value
	adr = []
	for x in range(num_day):
		eval_day_id = start_day_id + x
		eval_price = daily_database[eval_day_id][2]
		adr.append( calAR(base_price, eval_price))
	return adr

def getNextMR(month_date, monthly_database, month_id = -1):
	if month_id < 0:
		for id, data in enumerate(monthly_database):
			if isSameMonth(month_date, data[0]):
				current_month_id = id
				break
	else:
		current_month_id = month_id

	if current_month_id == len(monthly_database)-1:
		return None

	next_month_id = current_month_id + 1
	next_month_return = monthly_database[next_month_id][2]
	return [next_month_return]

def calAR(start_price, close_price):
	return (close_price - start_price)/start_price

def getInputData(daily_database, monthly_database):
	input_datas = []
	for month_id, month_data in enumerate(monthly_database):
		date = month_data[0]
		# calculate AMRs
		amr = getAMRs(date, monthly_database, 12, month_id)
		if amr is None:
			continue
		# calculate ADRs
		adr = getADRs(date, daily_database, 20)
		if adr is None:
			continue
		# calculate Jan flag
		if date[1]==1:
			jan = [1]
		else:
			jan = [0]
		nmr = getNextMR(date, monthly_database)
		if nmr is None:
			continue

		input_data = amr + adr + jan + nmr
		input_datas.append(input_data)
	return input_datas

	# pass
def parseDataBase(file_path):
	'''
	parse a signel file into daily database and monthly database
	one assumption of using this function is that the data lenght is the same
	for every stock
	'''

	backtest_precentage = 0.1
	daily_database = []
	monthly_database = []
	# total_lines = row_count = sum(1 for line in open(file_path))
	# backtest_start_index = int(total_lines * backtest_precentage)
	# print(total_lines, backtest_start_index)
	with open(file_path, 'rt') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',', skipinitialspace=True)
		# calculate the backtest index
		# read the dates first
		last_month_date = None
		last_month_id = -1
		last_month_value = 0
		for id, row in enumerate(reader):
			date = parseDate(row['Date'])
			vclose = float(row['Close'])
			vopen = float(row['Open'])
			daily_return = (vclose - vopen)/vopen
			if last_month_date is None:
				last_month_date = date
				last_month_value = vclose
				last_month_id = 0
			else:
				if not isSameMonth(date, last_month_date):
					# new month value is equal to last day closed price
					# get the new month from last day in the daily database
					new_month = daily_database[-1]
					new_month_value = new_month[2] # 2 is the closed value
					monthly_return = calAR(last_month_value, new_month_value)
					monthly_database.append( [date, new_month_value, monthly_return] )
					last_month_date = date
					last_month_id = id
					last_month_value = new_month_value
			daily_database.append( [date, vopen, vclose, daily_return] )

	return daily_database, monthly_database

def writeInputData(file_path, input_data):
	with open(file_path, 'wt') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		for data in input_data:
			writer.writerow(["% .4f"%x for x in data])

def generateFakeData(input_data):
	input_array = numpy.array(input_data)
	std = numpy.std(input_array, axis=0)
	mu = numpy.std(input_array, axis=0)
	numpy.random.normal()
	fake_array = copy.deepcopy(input_array)
	for row in fake_array:
		for id, col in enumerate(row):
			rand_num = random.gauss(mu[id], std[id])
			row[id] = row[id] + rand_num
	# make the flag eaqual
	fake_array[:,-2] = input_array[:,-2]
	return fake_array.tolist()

def main():
	use_fakedata = True
	zscore = False

	write_address = os.path.abspath("../pdata/")  # pdata Stands for processed data
	data_address = os.path.abspath("../data/")
	data_files = os.listdir(data_address)

	if os.path.exists(write_address):
		shutil.rmtree(write_address)
	os.makedirs(write_address)

	all_datas = []
	if use_fakedata:
		# use first data file as sample data
		sample_data_path = os.path.join(data_address, data_files[0])
		[daily_database, monthly_database] = parseDataBase(sample_data_path)
		sample_data = getInputData(daily_database, monthly_database)
		# generate 100 fake data
		for x in range(100):
			fake_data = generateFakeData(sample_data)
			all_datas.append(fake_data)
		print("Generated %d Fake Data which all have size %d"%(len(all_datas),len(fake_data)))
	else:
		for file in data_files:
			file_path = os.path.join(data_address, file)
			[daily_database, monthly_database] = parseDataBase(file_path)
			input_data = getInputData(daily_database, monthly_database)
			all_datas.append(input_data)

	all_datas = numpy.array(all_datas, dtype = numpy.float64)
	# add another row to the data to store the normalized output
	all_datas = numpy.append(all_datas, numpy.zeros((all_datas.shape[0],all_datas.shape[1],1)),axis=2)
	# first axis is depth which is each stock
	# second axis is row which is each month
	# third axis is col which is every input, last col is label
	# [... flag, monthly_return, normalized_montly_return]
	for row in range(all_datas.shape[1]):
		for col in range(all_datas.shape[2]):
			# if col is the flag col don't nromalize, if col is the last col don't normalize 
			if col == all_datas.shape[2] - 3 or col == all_datas.shape[2] - 1:
				continue
			# if the col is the montly return col feed the data to tcol
			if col == all_datas.shape[2] - 2:
				tcol = col + 1
			else:
				tcol = col
 
			if zscore:
				# z-score implementation
				mu = numpy.mean(all_datas[:,row,col])
				sigma = numpy.std(all_datas[:,row,col])
				all_datas[:,row,tcol] = (all_datas[:,row,col] - mu)/sigma
			else:
				# normalize by the maximum and put the result to tcol
				# minus the minimal first to get all data positive
				minimum = numpy.amin(all_datas[:,row,col])
				all_datas[:,row,tcol] = (all_datas[:,row,col]-minimum+0.01)
				# divide by the new maximum to normalze the data from 0 to 1
				maximum = numpy.amax(all_datas[:,row,tcol])
				all_datas[:,row,tcol] = all_datas[:,row,tcol]/maximum

	# find the median value in the output list and classify them into two class
	# add three rows to store the normalized stock value, the 
	all_datas = numpy.append(all_datas, numpy.zeros((all_datas.shape[0],all_datas.shape[1],2)),axis=2)
	# normalize the data 
	# [... flag, monthly_return, normalized_montly_return, class1, class2]
	for row in range(all_datas.shape[1]):
		median = numpy.median(all_datas[:,row,-4])
		for depth in range(all_datas.shape[0]):
			if all_datas[depth,row,- 4] > median:
				all_datas[depth,row,-1] = 0
				all_datas[depth,row,-2] = 1
			else:
				all_datas[depth,row,-1] = 1
				all_datas[depth,row,-2] = 0

	# write into a numpy binary file
	write_path = os.path.join(write_address, "stock.db")
	with open(write_path, 'wb') as bfile:
		numpy.save(bfile, all_datas)

	# test if the file is really wroted
	with open(write_path,'rb') as bfile:
		read_datas = numpy.load(bfile)
		print("Saved as Database has shape",read_datas.shape)

	# write a sample of readable data 
	file_name = "sample.csv"
	write_path = os.path.join(write_address, file_name)
	write_data = all_datas[0,:,:]
	writeInputData(write_path, all_datas[0,:,:])

if __name__ == '__main__':
	main()