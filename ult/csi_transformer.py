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
	return (close_price - start_price)/start_price

def is_same_month(date1, date2):
	if date1.month != date2.month:
		return False
	else:
		return True		

def get_monthly_database(db):
	monthly_database = collections.OrderedDict()
	for stock in db.stocks:
		last_day = None
		last_day_value = None
		last_month_date = None
		last_month_value = None
		for date, value in stock.values.items():
			vopen = value["Open"]
			vclose = value["Close"]
			if stock.name not in monthly_database:
				monthly_database.update({stock.name: collections.OrderedDict()})
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
					monthly_database[stock.name].update({last_day: {"Last":last_month_value, 
																	"Current":new_month_value, 
																	"Monthly Return":monthly_return}})
					last_month_date = date
					last_month_value = new_month_value
			last_day_value = vclose
			last_day = date
	return monthly_database

def transform(db, dir):
	"""
	@brief      transform a database manager object into a momentum database and save it into dir
	
	@param      db     The database
	@param      dir    The directory
	
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

	# calculate the 12 momentum
	for date in all_month_dates:
		pass

	# print(all_month_dates)

	# for stock in stocks:
	# 	for value in stock.value:
	# 		pass

if __name__ == '__main__':

	# test open data base
	file = open('../pdata/csi.db','rb')
	db = pickle.load(file)
	transform(db,[])


	# # test get montly database
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