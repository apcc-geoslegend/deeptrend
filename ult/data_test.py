import datetime

if __name__ == '__main__':
	date1 = datetime.date(1992,10,30)
	date2 = datetime.date(1992,10,29)
	
	if date1 == date2:
		print("Same")

	if date1 in [date2]:
		print("In")

	l = [date2,date1]
	print(sorted(l))