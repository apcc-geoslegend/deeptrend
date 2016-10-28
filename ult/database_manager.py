import csv
import os.path

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
		
		# TODO: develop a database manager that can get data by date, month, year 
	
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
