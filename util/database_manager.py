import csv
import os.path
import copy
from collections import OrderedDict
from operator import itemgetter

# class DailyValue:
#   def __init__(self,value):

#       if "Close" in value:
#           vclose = value["Close"]
#       else:
#           return None

#       if "Open" in value:
#           vopen = value["Open"]

#       if "Volume" in value:
#           volume = value["Volume"]

#       self.value = value["Date"]

# class Stock:
#     def __init__(self, stock_name, stock_id, db_id):
#         self.name = stock_name
#         self.id = stock_id
#         self.db_id = db_id
#         self.datas = []
#         self.values = OrderedDict()

#     def add_value(self, value):
#         if "Date" not in value:
#             print("stock.add_value(): Need to at least have date in the value")
#             return None
#         date = value["Date"]
#         self.values.update( {date: {x:value[x] for x in value if x is not "Date"}})

#     def get_all_dates(self):
#         dates = [] 
#         for date in values:
#             dates.append(date)
#         return dates

class DatabaseManager:
    """
    @brief      Data Base Manager provide some basic function that might be used for other
                Classes
    """
    
    def __init__(self):
        self.current_stock = None
        self.stocks = OrderedDict()
        self.dates = []

    def create_new_stock(self, stock_name):
        print("Added new stock %s"%stock_name)
        # st = Stock(stock_name, stock_id, len(self.stocks))
        st = {stock_name:OrderedDict()}
        self.stocks.update(st)
        self.current_stock = self.stocks[stock_name]

    def feed_current_stock(self, date, value):
        if self.current_stock is None:
            print("Please create stock first")
            return None
        self.current_stock.update({date:value})

    @staticmethod
    def is_same_month(date1, date2):
        if date1.month == date2.month and date1.year == date2.year:
            return True
        else:
            return False

    @staticmethod
    def is_same_date(date1,date2):
        if date1 == date2:
            return True
        else:
            return False

    def sort(self):
        print("Sorting database, plase wait")
        for stock in self.stocks:
            self.stocks[stock] = OrderedDict(sorted(self.stocks[stock].items(), key=itemgetter(0)))

    def get_last_N_days_data(self,stock,date,N):
        if stock not in self.stocks:
            return None
        
        values = self.stocks[stock]
        if date not in values:
            return None

        id = values.keys().index(date)
        if (id - N)<0: # not enough data
            return None

        all_datas = OrderedDict()
        for xdate,value in values.items()[id-N:id]:
            all_datas.update({xdate:value})

        # all_datas = OrderedDict(sorted(all_datas.items(), key=itemgetter(0)))
        return all_datas

    def get_current_month_all_data(self,stock,date):
        if stock not in self.stocks:
            return None
        values = self.stocks[stock]
        if date not in values:
            return None
        all_datas = OrderedDict()
        for xdate,value in values.items():
            if self.is_same_month(xdate, date):
                all_datas.update({xdate:value})

        # all_datas = OrderedDict(sorted(all_datas.items(), key=itemgetter(0)))
        return all_datas

    def get_all_dates(self):
        self.dates.sort()
        return dates

    def get_all_stock_by_date(self,date):
        values = []
        for stock in self.stocks:
            if date in stock.values:
                value = copy.copy(stock.values[date])
                name = {"Name": stock}
                value.update(name)
                values.append(value)
        return values

    def feed_stock(self, stock_name, date, value):
        if stock_name not in self.stocks:
            self.stocks.update({stock_name:OrderedDict()})
        self.stocks[stock_name].update({date: value})

if __name__ == '__main__':
    pass
    # db = DatabaseManager("../data/")
    # print(db.getAllStocksId())
