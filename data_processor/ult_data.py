import csv
import os.path
import numpy
import copy

def getData(file_path):
	file_path = os.path.abspath(file_path)
	with open(file_path, 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		data = []
		for row in reader:
			data.append([float(x) for x in row])
		data_array = numpy.array(data)
		return data_array

def getAllData(data_dir):
	data_dir = os.path.abspath(data_dir)
	data_files = os.listdir(data_dir)
	all_datas = []
	for file in data_files:
		file_path = os.path.join(data_dir, file)
		new_data = getData(file_path)
		all_datas.append(new_data)
	all_datas = numpy.array(all_datas)
	print("Get all datas, database has shape:", all_datas.shape)

if __name__ == '__main__':
	data_dir = "../fake_data/"
	all_datas = getAllData(data_dir)