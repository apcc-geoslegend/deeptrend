# this data transformer will transform what ever data file in ../data/
# into a database file that using momenteum strategy

import csv
import os.path
import shutil
import warnings
import numpy
import copy
import matplotlib.pyplot as plt
import random
from pyexcel_xls import get_data
import json

def parse_date(str):
	v = str.split('/')
	if len(v)!=3:
		warnings.warn('Wrong input for parse_date, input size does not match')
		return None
	else:
		return v

def is_same_month(date1, date2):
	if date1.month != date2.month:
		return False
	else:
		return True


def is_same_date(date1, date2):
	if date1 != date2:
		return False
	else:
		return True	 

def cal_ar(start_price, close_price):
	return (close_price - start_price)/start_price

def get_amrs(month_date, monthly_database, num_month, month_id = -1):
	# get accumulative last num_month montly return
	current_month_id = -1
	if month_id < 0:
		for id, data in enumerate(monthly_database):
			if is_same_month(month_date, data[0]):
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
		amr.append(cal_ar(base_price, eval_price))
		# print(monthly_database[eval_month_id][0])
	return amr

def get_adrs(day, daily_database, num_day):
	# get accumulative last num_day daily retrun
	current_day_id = -1
	for id, data in enumerate(daily_database):
		if is_same_date(day, data[0]):
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
		adr.append( cal_ar(base_price, eval_price))
	return adr

def get_next_mr(month_date, monthly_database, month_id = -1):
	# get next month monthly return
	if month_id < 0:
		for id, data in enumerate(monthly_database):
			if is_same_month(month_date, data[0]):
				current_month_id = id
				break
	else:
		current_month_id = month_id

	# if it's the last month, we don't have the data
	if current_month_id == len(monthly_database)-1:
		return None

	next_month_id = current_month_id + 1
	next_month_return = monthly_database[next_month_id][2]
	return [next_month_return]

def get_input_data(daily_database, monthly_database):
	input_datas = []
	for month_id, month_data in enumerate(monthly_database):
		date = month_data[0]
		# calculate AMRs
		amr = get_amrs(date, monthly_database, 12, month_id)
		if amr is None:
			continue
		# calculate ADRs
		adr = get_adrs(date, daily_database, 20)
		if adr is None:
			continue
		# calculate Jan flag
		if date.month==1:
			jan = [1]
		else:
			jan = [0]
		nmr = get_next_mr(date, monthly_database)
		if nmr is None:
			continue
		input_data = amr + adr + jan + nmr
		input_datas.append(input_data)
	return input_datas

def parse_database(file_path):
	'''
	parse a signel file into daily database and monthly database
	one assumption of using this function is that the data lenght is the same
	for every stock
	'''
	data = get_data(file_path)
	data = data['Sheet1']
	daily_database = []
	monthly_database = []
	
	last_month_date = None
	last_month_value = 0
	for id, row in enumerate(data):
		# skip the header
		if id == 0:
			continue
		date = row[0]
		vclose = float(row[4])
		vopen = float(row[1])
		volume = float(row[5])
		daily_return = (vclose - vopen)/vopen
		if last_month_date is None:
			last_month_date = date
			last_month_value = vclose
		else:
			if not is_same_month(date, last_month_date):
				# new month value is equal to last day closed price
				# get the new month from last day in the daily database
				new_month = daily_database[-1]
				new_month_value = new_month[2] # 2 is the closed value
				new_month_date = new_month[0]
				monthly_return = cal_ar(last_month_value, new_month_value)
				monthly_database.append( [new_month_date, new_month_value, monthly_return] )
				last_month_date = date
				last_month_value = new_month_value
		daily_database.append( [date, vopen, vclose, daily_return] )
	return daily_database, monthly_database

def write_input_data(file_path, input_data):
	with open(file_path, 'wt') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
		for data in input_data:
			writer.writerow(["% .4f"%x for x in data])

def generate_fake_data(input_data):
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

DATA_VERSION = "0.1.2"
DATA_ADDRESS = "../data/CSI/"
WRITE_ADDRESS = "../pdata/"

def generate_database(data_address=DATA_ADDRESS, write_address=WRITE_ADDRESS, use_fakedata=False, zscore=True):

	write_address = os.path.abspath(write_address)  # pdata Stands for processed data
	data_address = os.path.abspath(data_address)
	data_files = os.listdir(data_address)

	if os.path.exists(write_address):
		shutil.rmtree(write_address)
	os.makedirs(write_address)
	
	# #######
	# # find all file length
	# all_num_lines = []
	# for file in data_files:
	# 	file_path = os.path.join(data_address, file)
	# 	num = 0
	# 	with open(file_path,'rt') as read_file:
	# 		for row in read_file:
	# 			num +=1
	# 	all_num_lines.append(num)
	# print(all_num_lines)
	# # count the majority length
	# all_length_dict = {}
	# for num_line in all_num_lines:
	# 	if num_line not in all_length_dict:
	# 		all_length_dict[num_line] = 1
	# 	else:
	# 		all_length_dict[num_line] +=1
	# maximum_length = max(all_length_dict.iteritems(),key=lambda x: x[1])
	# maximum_length = maximum_length[0]
	# print("Uisng the maximum length", maximum_length)

	# valid_files = []
	# for id, num_line in enumerate(all_num_lines):
	# 	if num_line == maximum_length:
	# 		valid_files.append(data_files[id])
	# data_files = valid_files
	# #######
	# found out the lenght of the data is actually very diverse, so filtering by the number of lines
	# in the file is actually not a good idea.

	# get all the input data
	all_datas = []
	count = 0
	for file in data_files:
		file_path = os.path.join(data_address, file)
		# print(file_path)
		[daily_database, monthly_database] = parse_database(file_path)
		input_data = get_input_data(daily_database, monthly_database)
		all_datas.append(input_data)
		count += 1
		# if count > 10:
		# 	break
	# for data in monthly_database:
	# 	print(data)

	# count the majority length
	all_length_dict = {}
	for data in all_datas:
		if len(data) not in all_length_dict:
			all_length_dict[len(data)] = 1
		else:
			all_length_dict[len(data)] +=1
	maximum_length = max(all_length_dict.iteritems(),key=lambda x: x[1])
	maximum_length = maximum_length[0]
	print("Uisng the maximum length",maximum_length)
	
	# filter out the data that doesn't have the same length
	useful_data = []
	for data in all_datas:
		if len(data) == maximum_length: # 107 is the total month
			useful_data.append(data)

	all_datas = useful_data
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
				all_datas[:,row,tcol] = (all_datas[:,row,tcol] / maximum) * 0.99

	# find the median value in the output list and classify them into two class
	# add three rows to store the normalized stock value, the 
	all_datas = numpy.append(all_datas, numpy.zeros((all_datas.shape[0],all_datas.shape[1],2)),axis=2) 
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
	write_path = os.path.join(write_address, DATA_VERSION+".db")
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
	write_input_data(write_path, all_datas[0,:,:])

def main():
	generate_database()

if __name__ == '__main__':
	main()