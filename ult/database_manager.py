import csv
import os.path
from collections import namedtuple

Stock = namedtuple("Stock","Name ID FileName")
class DatabaseManager:
	
	def __init__(self,dir):
		self.dir = os.path.abspath(dir)
		self.files = os.listdir(self.dir)
		self.num_stocks = len(self.files)
		if self.num_stocks == 0:
			print('No File Detected')
			return None
		else:
			print('Successfully load database, found stocks number',self.num_stocks)
		self.stocks = []
		for id, file in enumerate(self.files):
			stock = Stock(self.parseNameFromFile(file), id, file)
			print(stock)
			self.stocks.append(stock)

		for stock in self.stocks:
			pass
		
		# TODO: develop a database manager that can get data by date, month, year
	@staticmethod
	def parseNameFromFile(file):
		id = file.find('.')
		return file[0:id]

	@staticmethod
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
	
	def getDataByDate(date):
		pass

	def getAllDaysByMonth(date):
		pass

	def getLastDayCloseValue(date):
		pass

	def getFirstDayOpenValue(date):
		pass

	def getPriviousDaysClosedValue(date, days):
		pass

	def getAllStocksId(self):
		return self.files


if __name__ == '__main__':
	db = DatabaseManager("../data/")
	print(db.getAllStocksId())
