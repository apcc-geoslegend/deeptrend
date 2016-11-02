# this is the interface for CSI data to feed to Database Manager

import os.path
from pyexcel_xls import get_data
from database_manager import DatabaseManager

# class CSIReader:
# 	def __init__(self,db):
# 		self.db = db

def feed_data(db, dir):
	dir = os.path.abspath(dir)
	data_files = os.listdir(dir)
	if len(data_files) == 0:
		print("Can't find any data file, check your directory")
		return None
	for file in data_files:
		file_path = os.path.join(dir, file)
		stock_name = file[0:file.find('.')]
		db.create_new_stock(stock_name, 0)
		data = get_data(file_path)
		data = data['Sheet1']
		for id, row in enumerate(data):
			if id == 0:
				continue
			date = row[0]
			vclose = float(row[4])
			vopen = float(row[1])
			volume = float(row[5])
			db.feed_current_stock({'Date':date, 'Open':vopen, 'Close':vclose, 'Volume':volume})

if __name__ == '__main__':
	db = DatabaseManager()
	feed_data(db,"../data/CSI/")