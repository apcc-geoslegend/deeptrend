import momentum_transformer as mt
import data_reader
import datetime
from  database_manager import DatabaseManager
import warnings

def print_database(db):
    for x in db:
        for y in db[x]:
            print(x,y,db[x][y])

def test_cal_adr():
    db = DatabaseManager()
    stocks = ["A"]
    year = [2016,2017]
    for x, stock in enumerate(stocks):
        for y, year in enumerate([2016,2017]):
            for month in range(1,13):
                for day in range(1,27):
                    date = datetime.date(year,month,day)
                    feed_dict = {}
                    # feed_dict["Close"] = float(day * 100 + x * 100 + y * 100 + month * 100)
                    feed_dict["Close"] = float(day)
                    # feed_dict["Volume"] = float(10000)
                    # dbm.feed_current_stock(date,feed_dict)
                    db.feed_stock(stock, date, feed_dict)
    db.sort()
    idb = {}
    mt.cal_amr(db, idb, 12)
    mt.cal_adr(db, idb, 25)
    # print_database(idb)
    for x in idb:
        for y in idb[x]:
            for i in range(25):
                if idb[x][y]["ADR"][i] != i+1:
                    assert ValueError("test_cal_adr failed")
                    return False
    return True

def test_cal_amr():
    db = DatabaseManager()
    stocks = ["A"]
    year = [2016,2017]
    for x, stock in enumerate(stocks):
        count = 0
        for y, year in enumerate([2016,2017,2018]):
            for month in range(1,13):
                for day in range(1,27):
                    date = datetime.date(year,month,day)
                    feed_dict = {}
                    # feed_dict["Close"] = float(day * 100 + x * 100 + y * 100 + month * 100)
                    count = count+1
                    feed_dict["Close"] = float(count)
                    # feed_dict["Volume"] = float(10000)
                    # dbm.feed_current_stock(date,feed_dict)
                    db.feed_stock(stock, date, feed_dict)
    db.sort()
    # print_database(db.stocks)
    idb = {}
    mt.cal_amr(db,idb,12)
    # print_database(idb)

    v = idb.itervalues().next().itervalues().next()["AMR"]
    for x in range(1,12):
        if v[x-1] != x:
            assert ValueError("test cal_amr failed")
            return False
    print("test cal_amr success")
    return True

def test_get_monthly_database():
    db = DatabaseManager()
    stocks = ["A"]
    year = [2016,2017]
    for x, stock in enumerate(stocks):
        for y, year in enumerate([2016,2017]):
            for month in range(1,13):
                for day in range(1,27):
                    date = datetime.date(year,month,day)
                    feed_dict = {}
                    feed_dict["Close"] = float(month)
                    db.feed_stock(stock, date, feed_dict)
    db.sort()
    mdb = mt.get_monthly_database(db)
    # print_database(mdb)

def test_cal_jan():
    db = DatabaseManager()
    stocks = ["A"]
    year = [2016,2017]
    for x, stock in enumerate(stocks):
        for y, year in enumerate([2016,2017]):
            for month in range(1,13):
                for day in range(1,27):
                    date = datetime.date(year,month,day)
                    feed_dict = {}
                    feed_dict["Close"] = float(month)
                    db.feed_stock(stock, date, feed_dict)
    db.sort()
    idb = {}
    idb = mt.cal_amr(db,idb,12)
    idb = mt.cal_jan(idb)
    # print_database(idb)
    if idb.itervalues().next().itervalues().next()["Jan"]!=1:
        assert ValueError("test cal_jan failed")
        return False
    else:
        print("test cal_jan success")
        return True

def test_transform():
    db = DatabaseManager()
    stocks = ["A","B","C"]
    year = [2016,2017]
    for x, stock in enumerate(stocks):
        count = 0
        for y, year in enumerate([2016,2017]):
            for month in range(1,13):
                for day in range(1,27):
                    date = datetime.date(year,month,day)
                    feed_dict = {}
                    count += 1
                    feed_dict["Close"] = float(count - x + 3)*100
                    db.feed_stock(stock, date, feed_dict)
    idb = mt.transform(db)
    # print_database(idb)
    date = idb.keys()[0]
    if idb[date]["C"]["Input"][0] < idb[date]["B"]["Input"][0] or idb[date]["B"]["Input"][0] < idb[date]["A"]["Input"][0]:
        assert ValueError("test_transform failed")
        return
    else:
        print("test_transform success")

if __name__ == '__main__':
    # db = data_reader.load()
    test_cal_adr()
    test_cal_amr()
    test_get_monthly_database()
    test_cal_jan()
    test_transform()

