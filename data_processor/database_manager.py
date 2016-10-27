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
		

	def getAllStocksId(self):
		return self.files


if __name__ == '__main__':
	db = DatabaseManager("../data/")
	print(db.getAllStocksId())
