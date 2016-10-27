import csv
import os.path
import warnings
import numpy
import collections
import copy
import matplotlib.pyplot as plt

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

def getAMRs(month_date, monthly_database, num_month):

	current_month_id = -1
	
	for id, data in enumerate(monthly_database):
		if isSameMonth(month_date, data[0]):
			current_month_id = id
			break

	if current_month_id == -1:
		print("Can't find this month in database, check if you are passing the right database")

	start_month_id = id - num_month
	if start_month_id < 0:
		return None

	start_price = monthly_database[start_month_id][1]
	# print("Current month", month_date)
	amr = []
	for x in range(num_month):
		eval_month_id = start_month_id + x
		eval_price = monthly_database[eval_month_id][1]
		amr.append(calAR(start_price, eval_price))
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
	if start_day_id < 0:
		return None

	start_price = daily_database[start_day_id][1]
	adr = []
	for x in range(num_day):
		eval_day_id = start_day_id + x
		eval_price = daily_database[eval_day_id][1]
		adr.append( calAR(start_price, eval_price))
	return adr

def calAR(start_price, close_price):
	return (close_price - start_price)/start_price

def getInputData(daily_database, monthly_database):
	input_datas = []
	for month_data in monthly_database:
		# calculate AMRs
		amr = getAMRs(month_data[0], monthly_database, 12)
		if amr is None:
			continue
		# calculate ADRs
		adr = getADRs(month_data[0], daily_database, 20)
		if adr is None:
			continue
		# calculate Jan flag
		if month_data[0][1]==1:
			jan = [1]
		else:
			jan = [0]
		input_data = amr + adr + jan
		input_datas.append(input_data)
	return input_datas

	# pass
def parseDataBase(file_path):

	daily_database = []
	monthly_database = []
	with open(file_path, 'rt') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',', skipinitialspace=True)
		# read the dates first
		last_month = None
		last_month_value = 0
		for id, row in enumerate(reader):
			date = parseDate(row['Date'])
			vclose = float(row['Close'])
			vopen = float(row['Open'])
			daily_return = (vclose - vopen)/vopen
			daily_database.append( [date, vopen, vclose, daily_return] )
			if last_month is None:
				last_month = date
				last_month_value = vopen
			else:
				if date[0] != last_month[0] or date[1] != last_month[1]:
					last_month = date
					monthly_return = (vopen - last_month_value)/last_month_value
					last_month_value = vopen
					monthly_database.append( [date, vopen, monthly_return] )
					# print(monthly_database[-1])

	return daily_database, monthly_database

def main(DATA_ADDRESS):
	data_address = os.path.abspath(DATA_ADDRESS)
	data_files = os.listdir(data_address)
	for file in data_files:
		file_path = os.path.join(data_address,file)
		[daily_database, monthly_database] = parseDataBase(file_path)
		input_data = getInputData(daily_database, monthly_database)
		print("Found Input Data has dimention",len(input_data),len(input_data[0]))

if __name__ == '__main__':
	DATA_ADDRESS = "../data/"
	main(DATA_ADDRESS)
	# testParseDate()