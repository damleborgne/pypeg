import numpy as np

def myreadtest(file, array):
	""" Reads an array from file at the current position. The
	Array can span over several lines. If the array is defined, """

	n=len(array)
	dtype=array.dtype
		
	#array=np.array([]) # DO NOT DO THIS OTHERWISE ARRAY WILL NOT BE CHANGED !
	# as explained in http://www.python-course.eu/passing_arguments.php
	
	iarr=0
	while (iarr < n):
		line=file.readline()
		arrline=[np.float(i) for i in line.rstrip().split()]
		array[iarr:min(iarr+len(arrline),n)] = arrline
		iarr+=len(arrline)
		
	array=array.astype(dtype) # convert back to input type

def myread(file, n, dtype=None):
	""" Reads an array from file at the current position. The
	Array can span over several lines. If the array is defined, """

	if dtype is None:
		dtype=float
				
	array=np.zeros(n)
	
	iarr=0
	while (iarr < n):
		line=file.readline()
		arrline=[np.float(i) for i in line.rstrip().split()]
		array[iarr:min(iarr+len(arrline),n)] = arrline
		iarr+=len(arrline)		
	array=array.astype(dtype) # convert back to input type
	return array

