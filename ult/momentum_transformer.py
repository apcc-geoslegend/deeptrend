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

import json
import cPickle as pickle
import datetime
from collections import OrderedDict

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
    """
    The way it finds the next month is ex:
    2006,1,30 Closed = 200
    2006,2,1 Closed = 100
    Once the month changed to a new one, add the last day into a new month
    {(2006,1,30),"Closed":200}
    """
    # monthly database structs { Stock Name: {Montly Date: {Last Month Value, Current Month Value, Monthly Return} } }
    monthly_database = OrderedDict()
    for stock, stock_value in db.stocks.items():
        last_day = None
        last_day_value = None
        last_month_date = None
        last_month_value = None
        for date, value in stock_value.items():
            vclose = value["Close"]
            if stock not in monthly_database:
                monthly_database.update({stock: OrderedDict()})
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

def transform(db):
    """
    @brief      transform a database manager object into a momentum database and
                save it into dir

    @param      db    The database
    @param      dir   The directory

    @return     { description_of_the_return_value }
    """
    print("Processing Data, Please wait")
    mdb = get_monthly_database(db)

    print("Calculating Acumulative Montly Return")
    # idb contains { stock_name: {monthly_date: {"AMR:[12 accumulative monthly return], "ADR":[20 days dialy return]} } }
    idb = {}
    # calculate the 12 month momentum
    monthly_return_range = 12
    for stock, value in mdb.items(): # stock is the stock.name, value should be the dict of {date:value}
        # we need at least two extra month to process
        if len(value) < monthly_return_range + 2:
            #if there is not enough data to process
            continue
        if stock not in idb:
            idb.update({stock:OrderedDict()})
        # start with 1 because we need t - 13
        # finished with  - mm_rage -1 because we need next month value for monthly return
        srange = range(1, len(value) - monthly_return_range)
        for sid in srange:
            base_month = value.items()[sid - 1][0]
            base_month_value = value[base_month]["Current"]
            # calculate the acumulated monthly return
            amr = []
            for cid in range(monthly_return_range):
                current_month = value.items()[sid + cid][0]
                current_month_value = value[current_month]["Current"]
                mr = cal_ar(base_month_value, current_month_value)
                amr.append(mr)
            if len(amr)!= monthly_return_range:
                print("THIS SHOULD NOT HAPPEN: AMR length not match")
            idb[stock].update({current_month:{"AMR":amr}})
            next_month_id = sid + monthly_return_range
            if next_month_id > len(value)-1:
                print("THIS SHOULD NOT HAPPEN: Next Month ID greater than it's maximum length")
            next_month = value.items()[next_month_id][0]
            next_month_return = value[next_month]["Monthly Return"]
            # print(base_month, current_month, next_month)
            
            # NMR stands for Next Month Return
            idb[stock][current_month]["NMR"] = next_month_return
        if not idb[stock]:
            print("THIS SHOULD NOT HAPPEN: No value found for this stock", stock, len(value))

    print("Calculating Acumulative Daily Return")
    # for every idb month calculate it's 20 days return
    # idb contains { stock_name: {monthly_date:[12 accumulative monthly return] } }
    daily_return_length = 20
    for stock, value in idb.items():
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
            if len(adr) != daily_return_length:
                print("ADR length not match Should not happen")
                idb[stock][month] = {}
            idb[stock][month].update({"ADR":adr})

    for stock, value in idb.items():
        for date, xvalue in value.items():
            if date.month == 1:
                idb[stock][date]["Jan"] = 1
            else:
                idb[stock][date]["Jan"] = 0

    # find all the end dates in the database
    all_month_dates = []
    all_stock_name = []
    for stock, value in idb.items():
        if stock not in all_stock_name:
            all_stock_name.append(stock)
        for date in value:
            if date not in all_month_dates:
                all_month_dates.append(date)

    print("Calculating the Median")
    # find all the meadian in the database
    medians = {}
    means = {}
    for date in all_month_dates:
        # print(date)
        nmrs = []
        for stock in idb:
            if date in idb[stock]:
                nmrs.append(idb[stock][date]["NMR"])
        nmrs = numpy.array(nmrs)
        median = numpy.median(nmrs)
        mean = numpy.mean(nmrs)
        medians[date] = median
        means[date] = mean

    print("Noramlizing Input")
    # zero score normalization
    # calculate mean
    all_means = {}
    # odb stands for output database
    odb = OrderedDict()
    for date in all_month_dates:
        count = 0
        # in all_value row is stock, col is value
        all_value = []
        astocks = []
        for stock in idb:
            if date in idb[stock]:
                astocks.append(stock)
                value = numpy.array(idb[stock][date]["AMR"] + idb[stock][date]["ADR"] +[idb[stock][date]["Jan"]])
                all_value.append(value)
                count += 1
        if len(all_value) == 1:
            print("Found a date that only have one value",date,stock)
            continue
        all_value = numpy.array(all_value)
        # z-score is happening here
        for n, col in enumerate(all_value.T):
            if n == all_value.shape[1]-1:
                continue
            mu = numpy.mean(col)
            sigma = numpy.std(col)
            all_value[:,n] = (col - mu)/sigma

        if date not in odb:
            odb.update({date:{}})
        # constrcut the final output
        for m, row in enumerate(all_value):
            stock = astocks[m]
            median = medians[date]
            nmr = idb[stock][date]["NMR"]
            if nmr > median:
                oc = [1,0]
            else:
                oc = [0,1]
            oc = numpy.array(oc)
            odb[date].update({stock:{
                "Input": all_value[m,:],
                "Class": oc,
                "NMR": nmr
                }})
    return odb

INPUT_ADDRESS = os.path.abspath("../pdata/nyse.rdb")
OUTPUT_ADDRESS = os.path.abspath("../pdata/momentum.db")
import nyse_reader as data_reader
from database_manager import DatabaseManager 

def generate_database(input_address = INPUT_ADDRESS, output_address = OUTPUT_ADDRESS):
    
    save_dir = os.path.abspath("../pdata/momentum.db")
    if data_reader.database_exsists():
        db = data_reader.load()
    else:
        db = data_reader.generate_database()

    odb = transform(db)
    output_file = open(save_dir,'wb')
    print("Saving Database, Please Wait")
    pickle.dump(odb,output_file)

if __name__ == '__main__':
    generate_database()
    # a = []
    # print(len(a))
    # 
    # a = numpy.array([1,2,3,4])
    # b = numpy.array([1,2,3,4])
    # c = numpy.array([a,b])
    # print(c)