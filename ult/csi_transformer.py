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
import pickle
import datetime
import collections

def cal_ar(start_price, close_price):
	"""
	@brief      calculate the acumulative montly return
	
	@param      start_price  The start price
	@param      close_price  The close price
	
	@return     monthly return
	"""
	return (close_price - start_price)/start_price

def is_same_month(date1, date2):
	"""
	@brief      Determines if same month.
	
	@param      date1  The date 1
	@param      date2  The date 2
	
	@return     True if same month, False otherwise.
	"""
	if date1.month == date2.month and date1.year == date2.year:
	    return True
	else:
	    return False

def get_monthly_database(db):
	"""
	@brief      Gets the monthly database.
	
	@param      db    The database
	
	@return     The monthly database.
	"""
	
	# monthly database structs { Stock Name: {Montly Date: {Last Month Value, Current Month Value, Monthly Return} } }
	monthly_database = collections.OrderedDict()
	for stock, stock_value in db.stocks.items():
		last_day = None
		last_day_value = None
		last_month_date = None
		last_month_value = None
		for date, value in stock_value.items():
			vopen = value["Open"]
			vclose = value["Close"]
			if stock not in monthly_database:
				monthly_database.update({stock: collections.OrderedDict()})
			daily_return = (vclose - vopen)/vopen
			if last_month_date is None:
				last_month_date = date
				last_month_value = vclose
			else:
				if not is_same_month(date, last_month_date):
					# new month value is equal to last day closed price
					# get the new month from last day in the daily database
					new_month_value = last_day_value
					monthly_return = cal_ar(last_month_value, new_month_value)
					monthly_database[stock].update({last_day: {"Last":last_month_value, 
																"Current":new_month_value, 
																"Monthly Return":monthly_return}})
					last_month_date = date
					last_month_value = new_month_value
			last_day_value = vclose
			last_day = date
	return monthly_database

def transform(db, dir):
	"""
	@brief      transform a database manager object into a momentum database and
	            save it into dir
	
	@param      db    The database
	@param      dir   The directory
	
	@return     { description_of_the_return_value }
	"""

	mdb = get_monthly_database(db)
	# find all the end dates in the database
	all_month_dates = []
	for stock, value in mdb.items():
		for date in value:
			if date not in all_month_dates:
				all_month_dates.append(date)
	# find all the meadian in the database
	medians = {}
	for date in all_month_dates:
		# print(date)
		median = []
		for stock, value in mdb.items():
			if date in value:
				median.append(value[date]["Monthly Return"])
		median = numpy.array(median)
		median = numpy.median(median)
		medians[date] = median

	# amrs contains { stock_name: {monthly_date:[12 accumulative monthly return] } }
	amrs = {}
	# calculate the 12 month momentum
	mm_range = 12
	all_month_dates.sort()
	for stock, value in mdb.items(): # stock is the stock.name, value should be the dict of {date:value}
		# we need at least two extra month to process
		if len(value) < mm_range + 2:
			#if there is not enough data to process
			continue
		if stock not in amrs:
			amrs.update({stock:collections.OrderedDict()})
		# start with 1 because we need t - 13
		# finished with  - mm_rage -1 because we need next month value for monthly return
		for sid in range(1, len(value) - mm_range - 1):
			base_month = value.items()[sid - 1][0]
			base_month_value = value[base_month]["Current"]
			# calculate the acumulated monthly return
			amr = []
			for cid in range(mm_range):
				current_month = value.items()[sid + cid][0]
				current_month_value = value[current_month]["Current"]
				mr = cal_ar(base_month_value, current_month_value)
				amr.append(mr)
			amrs[stock].update({current_month:amr})
		if not amrs[stock]:
			print("THIS SHOULD NOT HAPPEN: No value found for this stock", stock, len(value))


	# for every amrs month calculate it's 20 days return
	# amrs contains { stock_name: {monthly_date:[12 accumulative monthly return] } }
	daily_return_length = 20
	adrs = {}
	for stock, value in amrs.items():
		if stock not in adrs: 
			adrs.update({stock:collections.OrderedDict()})
		for month, amr in value.items():
			data = db.get_last_N_days_data(stock, month, daily_return_length+1)
			if data is None:
				continue
			if (len(data) < daily_return_length + 1):
				print("THIS SHOULD NOT HAPPEN: Not enough day for this stock", stock, month, len(data))
				continue
			start_date = None
			start_value = None
			adr = []
			for xdate, xvalue in data.items():
				if start_date == None:
					start_date = xdate
					start_value = xvalue["Close"]
					continue
				current_value = xvalue["Close"]
				dr = cal_ar(start_value, current_value)
				adr.append(dr)
			adrs[stock].update({month:adr})

	for stock, value in adrs.items():
		# print(value)
		for date, xvalue in value.items():
			print(date)
			for yvalue in xvalue:
				print(yvalue)
		break

	# for stock, value in amrs.items():
		
	# for date in all_month_dates:
	# 	if date.month - 12
	# 	pass

	# print(all_month_dates)

	# for stock in stocks:
	# 	for value in stock.value:
	# 		pass

if __name__ == '__main__':

	# test open data base
	file = open('../pdata/csi.db','rb')
	db = pickle.load(file)
	transform(db,[])


	# test get montly database
	# monthly_database = get_monthly_database(db)
	# for k,v in monthly_database.items():
	# 	print(k)
	# 	for x,y in v.items():
	# 		print(x,y) 
	# 	break


	# # test get all stock by date function
	# date = datetime.date(2016,10,5)
	# xxx = db.get_all_stock_by_date(date)
	# for a in xxx:
	# 	print(a)
	# print(len(xxx))