import os
import numpy as np
def read_text_file(file_path): 
	fp = open(file_path, "r")
	lines = fp.readlines()
	lines = [i for i in lines  if i != "\n"]


	fp.close()
	return  lines
def addToProcessedResults(fp1,fp2,lines): 
	line = " ".join(lines)  #make lines of string into a line of strings
	if line.strip() != "":
		line = line.replace("\n", "|")
		line = line.strip("|")	 
		fp1.write(line + "\n") 
		fp2.write("<tr>")   
		for e in line.split("|"):
			fp2.write("<td>" +  e  +  "</td>")  
		fp2.write("</tr>")  
	return True
path = "Results\ProcessAll"
os.chdir(path)  # changed directory
files = os.listdir()
rows, cols = (len(files), 2)
arr2sort = np.full( (rows, cols),  0)  
iRowIndex = -1 
for file in  files:   
	iRowIndex = iRowIndex +  1  # index number of this file
	lines = read_text_file(file) 
	print(lines)
	arr2sort[iRowIndex , 0 ] = lines[0] # index number of this simulation is stored here...used for sorting array
	arr2sort[iRowIndex , 1 ] =  iRowIndex  # used to retrieve simulation from lines/list of simulations 
	 
arr2sort = arr2sort[arr2sort[:, 0].argsort()] 
os.chdir("..")  # changed directory to Results, i.e. the parent of Results\ProcessAll  
fp1 = open("outputOfProcessedResults.txt", "w")
fp2 = open("outputOfProcessedResults.html", "w")
path = "ProcessAll"
os.chdir(path)  # changed director to "Results\ProcessAll"  
for e  in arr2sort:
	file = files[e[1]]
	lines = read_text_file(file)
	addToProcessedResults(fp1,fp2,lines)

fp1.close()
fp2.close()
