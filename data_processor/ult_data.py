import csv
import os.path
import numpy

def getData(file_path):
	with open(file_path, 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		data = []
		for row in reader:
			data.append([float(x) for x in row])
		data_array = numpy.array(data)
		return data_array

# if __name__ == '__main__':
# 	file_path = "../fake_data/0000.txt"
# 	file_path = os.path.abspath(file_path)
# 	