import csv
import os.path
from database_manager import DatabaseManager
import cPickle as pickle
import datetime

READ_ADDRESS = os.path.abspath("../data/NYSE/")
WRITE_ADDRESS = os.path.abspath("../pdata/nyse.rdb")

def database_exsists():
	if os.path.exists(WRITE_ADDRESS):
		return True
	else:
		return False

def feed_data(db, dir):
	"""
	@brief      feed the data to the database
	
	@param      db    The database
	@param      dir   The directory of the saved database
	
	@return     nothing
	"""
	dir = os.path.abspath(dir)
	data_files = os.listdir(dir)
	if len(data_files) == 0:
		print("Can't find any data file, check your directory")
		return None
	count = 0
	for file in data_files:
		if file == "0.HEADER.csv":
			continue
		file_path = os.path.join(dir, file)
		stock_name = file[0:file.find('.')]
		db.create_new_stock(stock_name)
		with open(file_path,'rt') as file:
			reader = csv.reader(file)
			for row in reader:
				date = row[0]
				ids = [i for i in range(len(date)) if date[i] == "/"]
				year = int(date[0:ids[0]])
				month = int(date[ids[0]+1:ids[1]])
				day = int(date[ids[1]+1:])
				# print(year,month,day)
				date = datetime.date(year,month,day)
				vclose = float(row[3])
				db.feed_current_stock(date,{'Close':vclose})
			count += 1
			# if count > 1000:
			# 	break
	# db.sort()
	print("added row database for NYSE, total %d stocks"%count)
	return db

def load():
	if database_exsists():
		file = open(WRITE_ADDRESS, 'rb')
		print("Loading Database: NYSE, Please Wait")
		db = pickle.load(file)
		return db
	else:
		return None

def generate_database(read_address = READ_ADDRESS, write_address = WRITE_ADDRESS):
	db = DatabaseManager()
	feed_data(db,read_address)
	return db

if __name__ == '__main__':
	db = generate_database()
	output = open(WRITE_ADDRESS, 'wb')
	print("Saving data base please wait")
	pickle.dump(db, output)