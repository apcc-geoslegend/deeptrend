# this data transformer will transform what ever data file in ../data/
# into a database file that using momenteum strategy

import os.path
import warnings
import numpy
import copy
import cPickle as pickle
from collections import OrderedDict
import time

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

    @return     mdb   The monthly database.
    """
    """
    The way it finds the next month is ex:
    2006,1,30 Closed:200
    2006,2,1  Closed:100
    Once the month changed to a new one, add the last day into a new month
    {(2006,1,30),"Closed":200}
    """
    # monthly database structs { Stock Name: {Montly Date: {Last Month Value, Current Month Value, Monthly Return} } }
    mdb = OrderedDict()
    for stock, stock_value in db.stocks.items():
        last_day = None
        last_day_value = None
        last_month_date = None
        last_month_value = None
        for date, value in stock_value.items():
            vclose = value["Close"]
            if stock not in mdb:
                mdb.update({stock: OrderedDict()})
            if last_month_date is None:
                last_month_date = date
                last_month_value = vclose
            else:
                if not is_same_month(date, last_month_date):
                    # new month value is equal to last day closed price
                    # get the new month from last day in the daily database
                    new_month_value = last_day_value
                    monthly_return = cal_ar(last_month_value, new_month_value)
                    mdb[stock].update({last_day: {"Last":last_month_value,
                                                    "Current":new_month_value,
                                                    "Monthly Return":monthly_return}})
                    last_month_date = date
                    last_month_value = new_month_value
            last_day_value = vclose
            last_day = date
    return mdb

def cal_amr(db, idb, month_range):
    """
    @brief      Calculate the Acumulative Monthly Return
    
    @param      db           The database
    @param      idb          The idb
    @param      month_range  The month range
    
    @return     the output database
    """

    print("Calculating Monthly Database")
    start_time = time.time()
    mdb = get_monthly_database(db)
    print("Calculating Acumulative Montly Return")
    for stock, value in mdb.items(): # stock is the stock.name, value should be the dict of {date:value}
        # we need at least two extra month to process
        if len(value) < month_range + 2:
            #if there is not enough data to process
            continue
        if stock not in idb:
            idb.update({stock:OrderedDict()})
        # start with 1 because we need t - 13
        # finished with  - mm_rage -1 because we need next month value for monthly return
        srange = range(1, len(value) - month_range)
        for sid in srange:
            base_month = value.items()[sid - 1][0]
            base_month_value = value[base_month]["Current"]
            # calculate the acumulated monthly return
            amr = []
            for cid in range(month_range):
                current_month = value.items()[sid + cid][0]
                current_month_value = value[current_month]["Current"]
                mr = cal_ar(base_month_value, current_month_value)
                amr.append(mr)
            if len(amr)!= month_range:
                print("THIS SHOULD NOT HAPPEN: AMR length not match")
            next_month_id = sid + month_range
            if next_month_id > len(value)-1:
                print("THIS SHOULD NOT HAPPEN: Next Month ID greater than it's maximum length")
            next_month = value.items()[next_month_id][0]
            next_month_return = value[next_month]["Monthly Return"]
            # print(base_month, current_month, next_month)
            # NMR stands for Next Month Return
            idb[stock].update({current_month:{"AMR":amr}})
            idb[stock][current_month]["NMR"] = next_month_return
            # idb[stock][current_month]["Base Value"] = base_month_value
            # idb[stock][current_month]["Current Value"] = current_month_value
        if not idb[stock]:
            print("THIS SHOULD NOT HAPPEN: No value found for this stock", stock, len(value))
    print("AMR tiem used:", time.time() - start_time)
    return idb

def filter(db, idb):
    """
    @brief      filter idb by the designed filter
    
    @param      db    The raw database
    @param      idb   The intermediate database
    
    @return     the intermediate database
    """
    # filter idb with designed filter
    print("Applying filters on the data")
    filter_by_close_price = True
    filter_by_total_dollar = False
    minimum_total_value = 2.5e4
    minimum_close_price = 5 
    # apply filter:
    for stock in idb:
        for date in idb[stock]:
            if filter_by_close_price:
                if db.stocks[stock][date]["Close"] < minimum_close_price:
                    idb[stock].pop(date)
                    print("Pop stock %s at date %s that closed value is less than %f"%(stock, str(date), minimum_close_price))
                    continue
            if filter_by_total_dollar:
                total_dollar_volume = db.stocks[stock][date]["Close"] * db.stocks[stock][date]["Volume"]
                if total_dollar_volume < minimum_total_value:
                    idb[stock].pop(date)
                    print("Pop stock %s at date %s that totoal value is %f which is less than %f"%(stock, str(date), total_dollar_volume, minimum_total_value))
                    continue
    return idb

def cal_adr(db, idb, day_range):
    """
    @brief      calculate acumulative dialy return
    
    @param      db         The raw database
    @param      idb        The idb
    @param      day_range  The day range
    
    @return     the intermediate database
    """
    print("Calculating Acumulative Daily Return")
    start_time = time.time()
    for stock in idb:
        for month in idb[stock]:
            data = db.get_last_N_days_data(stock, month, day_range+1)
            if data is None:
                idb[stock].pop(month)
                continue
            if (len(data) < day_range + 1):
                print("THIS SHOULD NOT HAPPEN: Not enough day for this stock", stock, month, len(data))
                continue
            start_date = None
            start_value = None
            adr = []
            for xdate, xvalue in data:
                if start_date == None:
                    start_date = xdate
                    start_value = xvalue["Close"]
                    continue
                current_value = xvalue["Close"]
                dr = cal_ar(start_value, current_value)
                adr.append(dr)
            if len(adr) != day_range:
                print("ADR length not match Should not happen")
                idb[stock][month] = {}
            idb[stock][month].update({"ADR":adr})
    print("ADR tiem used:", time.time() - start_time)
    return idb

def cal_jan(idb):
    # calculate the Jan flag
    for stock in idb:
        for date in idb[stock]:
            if date.month == 1:
                idb[stock][date]["Jan"] = 1
            else:
                idb[stock][date]["Jan"] = 0
    return idb

def idb_to_odb(idb, z_score=False):
    """
    @brief      convert intermediate database to output database
    
    @param      idb   The idb
    
    @return     output database, which constcuted as {date:{stock:{"Input":xx,"Class":xx,"NMR":xx}}}
    """
    # find all the end dates in the database
    all_month_dates = []
    all_stock_name = []
    for stock, value in idb.items():
        if stock not in all_stock_name:
            all_stock_name.append(stock)
        for date in value:
            if date not in all_month_dates:
                all_month_dates.append(date)

    if len(all_month_dates) == 0:
        print("NO valid month found, Something wrong please check")

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
    count = 0
    # stock_num = 0
    for date in all_month_dates:
        # in values row is stock, col is value
        vinputs = []
        nnmrs = []
        astocks = list()
        for stock in idb:
            if date in idb[stock]:
                astocks.append(stock)
                vinput = numpy.array(idb[stock][date]["AMR"] + idb[stock][date]["ADR"] +[idb[stock][date]["Jan"]])
                nnmrs.append(idb[stock][date]["NMR"])
                vinputs.append(vinput)
                count += 1
        if len(vinputs) == 1:
            print("Found a date that only have one value",date,stock)
            continue
        nnmrs = numpy.array(nnmrs)
        maxima = numpy.max(nnmrs)
        minima = numpy.min(nnmrs)
        nnmrs = (nnmrs - minima + 1e-3)/(maxima - minima + 1e-3)
        # normalization is happening here
        vinputs = numpy.array(vinputs)
        for n, col in enumerate(vinputs.T):
            if n == vinputs.shape[1]-1:
                continue
            # z-score is happening here
            if z_score:
                mu = numpy.mean(col)
                sigma = numpy.std(col)
                vinputs[:,n] = (col - mu)/sigma
            else:
                # min max normalization
                minima = numpy.min(col)
                maxima = numpy.max(col)
                vinputs[:,n] = (col - minima + 1e-3)/(maxima - minima + 1e-3)

        if date not in odb:
            odb.update({date:{}})

        # constrcut the final output
        for m, row in enumerate(vinputs):
            stock = astocks[m]
            median = medians[date]
            nmr = idb[stock][date]["NMR"]
            if nmr > median:
                oc = [1,0]
            else:
                oc = [0,1]
            oc = numpy.array(oc)
            odb[date].update({stock:{"Input": vinputs[m,:],
                                    "Class": oc,
                                    "NMR": nmr,
                                    "NNMR": nnmrs[m]
                                    }})
    print("Toltal Data Point is %d, Total valid stocks is around %d"%(count, int(count/len(all_month_dates))))
    return odb

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
    # idb is intermediate database
    # idb contains { stock_name: {monthly_date: {"AMR:[12 accumulative monthly return], "ADR":[20 days dialy return]} } }
    idb = {}
    # calculate the 12 month momentum
    month_rage = 12
    cal_amr(db, idb, month_rage)
    # Just in case of it break the continunity of the month, it's better to
    # apply the filter after it calculate the month return
    filter(db, idb)

    # for every idb month calculate it's 20 days return
    # idb contains { stock_name: {monthly_date:[12 accumulative monthly return] } }
    day_range = 20
    cal_adr(db, idb, day_range)

    # clear the raw database
    del db
    # calculate january flag
    cal_jan(idb)
    # intermediate database to output database
    # odb constcuted as {date:{stock:{"Input":xx,"Class":xx,"NMR":xx}}}
    odb = idb_to_odb(idb)

    return odb

import data_reader
from database_manager import DatabaseManager 
DATABASE_NAME = data_reader.DATABASE_NAME
OUTPUT_ADDRESS = os.path.abspath("../pdata/%s.db"%data_reader.DATABASE_NAME)

def generate_database(output_address = OUTPUT_ADDRESS):
    """
    @brief      A higher level wrapper for generating the database
    
    @param      input_address   The input address
    @param      output_address  The output address
    
    @return     nothing
    """
    
    save_dir = os.path.abspath(output_address)
    db = data_reader.load()
    odb = transform(db)
    output_file = open(save_dir,'wb')
    print("Saving Database, Please Wait")
    pickle.dump(odb,output_file)

if __name__ == '__main__':
    generate_database()
    
    # a = {}
    # def fun(a):
    #     a["a"] = 1
    # fun(a)
    # print(a)