import csv
import os.path
import sys
from database_manager import DatabaseManager
import cPickle as pickle
import datetime

# DATABASE_NAME = "merged"
DATABASE_NAME = "NASDAQ30"
# DATABASE_NAME = "MERGED30"
# DATABASE_NAME = "NYSE30"
READ_ADDRESS = os.path.abspath("../data/"+DATABASE_NAME)
WRITE_ADDRESS = os.path.abspath("../pdata/%s.rdb"%DATABASE_NAME)

def database_exsits():
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
            file_path = os.path.join(dir, file)
            with open(file_path,'rt') as file:
                reader = csv.reader(file,skipinitialspace=True)
                for row in reader:
                    for id, col in enumerate(row):
                        if col == 'close_price':
                            col_close = id
                        if col == 'unadjusted_close_price':
                            col_unadjusted_close = id
                        if col == 'total_volume':
                            col_total_volume = id
    # print(col_close,col_unadjusted_close,col_total_volume)
    for file in data_files:
        if file == "0.HEADER.csv":
            continue
        file_path = os.path.join(dir, file)
        stock_name = file[0:file.find('.')]
        db.create_new_stock(stock_name)
        with open(file_path,'rt') as file:
            reader = csv.reader(file)
            for row in reader:
                feed_dict = {}
                date = parse_date(row[0])
                vclose = float(row[col_close])
                feed_dict['Close'] = vclose
                # vuclose = float(row[col_unadjusted_close])
                # feed_dict['Unadjusted Close'] = vuclose
                volume = float(row[col_total_volume])
                feed_dict['Volume'] = volume
                db.feed_current_stock(date,feed_dict)
            count += 1
            # if count > 100:
            #   break
    # db.sort()
    print("added row database, total %d stocks"%count)
    return db

def parse_date(string):
    """
    @brief      parse a string into date, the format of the string should be year/month/day
    
    @param      string  The string
    
    @return     { description_of_the_return_value }
    """
    ids = [i for i in range(len(string)) if string[i] == "/"]
    if len(ids) < 2:
        print("Date format of this data is wrong, plase use year/month/day")
        sys.exit(0)
    year = int(string[0:ids[0]])
    month = int(string[ids[0]+1:ids[1]])
    day = int(string[ids[1]+1:])
    # print(year,month,day)
    date = datetime.date(year,month,day)
    return date

def load():
    if database_exsits():
        file = open(WRITE_ADDRESS, 'rb')
        print("Loading Database, Please Wait")
        db = pickle.load(file)
        return db
    else:
        db = generate_database()
        return db

def generate_database(read_address = READ_ADDRESS, write_address = WRITE_ADDRESS):
    db = DatabaseManager()
    feed_data(db,read_address)
    return db

if __name__ == '__main__':
    db = generate_database()
    output = open(WRITE_ADDRESS, 'wb')
    print("Saving data base please wait")
    pickle.dump(db, output)