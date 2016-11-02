import csv
import os.path
from collections import namedtuple

# class DailyValue:
# 	def __init__(self,value):

# 		if "Close" in value:
# 			vclose = value["Close"]
# 		else:
# 		 	return None

# 		if "Open" in value:
# 			vopen = value["Open"]

# 		if "Volume" in value:
# 			volume = value["Volume"]

# 		self.value = value["Date"]

class Stock:
	def __init__(self, stock_name, stock_id, db_id):
		self.name = stock_name
		self.stock_id = stock_id
		self.db_id = db_id
		self.datas = []
		self.values = []

class DatabaseManager:
	
	def __init__(self):
		self.current_stock = None
		self.stocks = []
		self.dates = []

	def create_new_stock(self, stock_name, stock_id):
		print("Added new stock",stock_name)
		st = Stock(stock_name, stock_id, len(self.stocks))
		self.stocks.append(st)
		self.current_stock = self.stocks[-1]

	def feed_current_stock(self,value):
		if self.current_stock is None:
			print("Please create stock first")
			return None
		self.current_stock.values.append(value)

		if value["Date"] not in self.dates:
			self.dates.append(dates)

	@staticmethod:
	def is_same_month(date1, date2):
		if date1.month != date2.month:
			return False
		else:
			return True	

	@staticmethod:
	def cal_ar(start_price, close_price):
		return (close_price - start_price)/start_price

	def get_montly_database(self):
		monthly_database = []
		for stock in self..stocks:
			lase_day_value = None
			db = []
			for value in stock.value:
				date = value["Date"]
				vopen = value["Open"]
				vclose = value["Close"]
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
						monthly_return = cal_ar(last_month_value, new_month_value)
						db.append( {"Date": date, "Close": vclose, "Monthly Return": monthly_return})
						last_month_date = date
						last_month_value = new_month_value
				lase_day_value = vclose
			monthly_database.append(db)

	def get_all_dates(self):
		self.dates.sort()


	def get_all_stock_by_date(self,date):
		values = []
		for stock in self.stocks:
			for value in stock.values:
				if value["Date"] == date:
					values.append({"Stock Name":	stock.name, 
									"Stock ID":		stock_id, 
									"Database ID":	db_id, 
									"Open":			value["Open"]
									"Close":		value["Close"]
									})
				break
		return values

	def feed_stock(self, stock_name, value,stock_id = -1):
		pass

if __name__ == '__main__':
	pass
	# db = DatabaseManager("../data/")
	# print(db.getAllStocksId())
