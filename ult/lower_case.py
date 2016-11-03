import os

def main():
	path = os.path.abspath('../data/CSI')
	files = os.listdir(path)
	for file in files:
		id = file.find('.')
		new_file = file[0:id] + ".xls"
		file_path = os.path.join(path,file)
		new_file_path = os.path.join(path,new_file)
		print(new_file_path)
		os.rename(file_path, new_file_path)

if __name__ == '__main__':
	main()