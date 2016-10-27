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
		if data[0] == month[0] and data[1] == month[1]:
			value = data[3]
			return value
	return None

def findLastDayClosedValue(month, data_base):
	target_month = monthPlus(month, 1)
	for id, data in enumerate(data_base):
		if data[0] == target_month[0] and data[1] == target_month[1]:
			value = data_base[id][4]
			return value

def findAllDatesIndexByMonth(month, dates):
	find_dates = []
	for date in dates:
		if date[1] == month[1] and date[0] == month[0]:
			find_dates.append(date)
	return find_dates

def main(DATA_ADDRESS):
	data_address = os.path.abspath(DATA_ADDRESS)
	data_files = os.listdir(data_address)
	for file in data_files:
		file_path = os.path.join(data_address,file)
		data_input = []
		stock_id = file
		dates =[]
		daily_data_base = []
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
				daily_data_base.append( [date, vopen, vclose] )
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

			start_value = monthly_database[0][1]
			for data in monthly_database:
				amr.append((data[1] - start_value)/start_value)
			plt.plot(amr)
			plt.show()

if __name__ == '__main__':
	DATA_ADDRESS = "../data/"
	main(DATA_ADDRESS)
	# testParseDate()