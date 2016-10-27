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

def getAMR(month, data_base):
	# Acumulated monthly return from t - 2 to t - 13
	start_month = monthMinus(month, 12)
	for x in range(12):
		eval_month = monthMinus(month, x)
		first_day_value = findFirstDayOpenValue(eval_month,data_base)
		print(eval_month, first_day_value)

	# pass

def monthMinus(month, value):
	cal_month = list(month)
	cal_month[1] = cal_month[1] - value
	if cal_month[1] <= 0:
		cal_month[1] += 12
		cal_month[0] -= 1
	return cal_month

def monthPlus(month, value):
	cal_month = list(month)
	cal_month[1] = cal_month[1] + value
	if cal_month[1] > 12:
		cal_month[1] -= 12
		cal_month[0] += 1
	return cal_month

def findFirstDayOpenValue(month, data_base):
	for data in data_base:
		if isSameMonth(data, month):
			value = data[3]
			return value
	return None

def findLastDayClosedValue(month, data_base):
	target_month = monthPlus(month, 1)
	for id, data in enumerate(data_base):
		if isSameMonth(data, target_month):
			value = data_base[id][4]
			return value

def findAllDatesIndexByMonth(month, dates):
	find_dates = []
	for date in dates:
		if date[1] == month[1] and date[0] == month[0]:
			find_dates.append(date)
	return find_dates

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

def getAMRs(month_date, monthly_database):

	for id, data in enumerate(monthly_database):
		if isSameMonth(month_date, data[0]):
			current_month_id = id
			break

	start_month_id = id - 12
	if start_month_id < 0:
		return None

	start_price = monthly_database[eval_month_id][1]
	amr = []
	for x in range(12):
		eval_month_id = start_month_id + x
		eval_price = monthly_database[eval_month_id][1]
		amr.append(start_price, eval_price)
	return amr

def getADRs(date, daily_database):
	for id, daily_database in enumerate(daily_database):
		if isSameDate(date, data[0]):
			current_date_id = id
			break

	start_date_id = current_date_id - 20
	if start_date_id < 0
		return None

	start_price == daily_database

def calAR(start_price, close_price):
	return (close_price - start_price)/start_price

def getInputData(monthly_database, daily_database):
	input_data = []
	for month_data in monthly_database:
		amr = getAMRs(month_data[0],monthly_database)

	# pass

def main(DATA_ADDRESS):
	data_address = os.path.abspath(DATA_ADDRESS)
	data_files = os.listdir(data_address)
	for file in data_files:
		file_path = os.path.join(data_address,file)
		data_input = []
		stock_id = file
		dates =[]
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
				daily_database.append( [date, vopen, vclose] )
				if last_month is None:
					last_month = date
					last_month_value = vopen
				else:
					if date[0] != last_month[0] or date[1] != last_month[1]:
						last_month = date
						monthly_return = (vopen - last_month_value)/last_month_value
						last_month_value = vopen
						monthly_database.append( [date, vopen, monthly_return] )
						print(monthly_database[-1])
			getInputData(monthly_database, daily_database)

			# start_value = monthly_database[0][1]
			# for data in monthly_database:
			# 	amr.append((data[1] - start_value)/start_value)
			# plt.plot(amr)
			# plt.show()

if __name__ == '__main__':
	for x in range(2):
		print(x)
	# DATA_ADDRESS = "../data/"
	# main(DATA_ADDRESS)
	# testParseDate()