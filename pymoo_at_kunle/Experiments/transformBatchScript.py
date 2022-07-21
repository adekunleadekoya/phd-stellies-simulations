import fileinput

def preprocessLine(s,iIndexNumberOfSimulation):
	nS = s.split()
	nS[3] = "30"  # number of decision variables is initialized here 
	if nS[4] == "4":
		nS[4] = "5"  #change number of objectives to 5 if it's set to 4

	if nS[4] == "5":
		nS[6] = "128"  # pop-size:128 for 5-objective problems
	elif nS[4] == "8":
		nS[6] = "120"  # pop-size: 120 used for 8-objective problems
	elif nS[4] == "10":	 
		nS[6] = "220"   # pop-size:220 for 10-objective problems

	if  "dtlz1" in nS[2]:  #this applies to dtlz1
		k = 5
		nS[3] = int(nS[4]) + k - 1  #sets the number of decision variables
		nS[3] = str(nS[3])
	elif "dtlz" in nS[2]:   #other dtlz problems except dtlz1
		k = 10
		nS[3] = int(nS[4]) + k - 1    #sets the number of decision variables
		nS[3] = str(nS[3])

	elif "wfg" in nS[2]:
		if int(nS[4]) == 2:
			n_var = 4  # 2-objective problems forces k = 4, as defined in pymoo source code and  l = 0 where n_var = k + l
		else:
			n_var =  (int(nS[4]) - 1) * int(nS[4]) # n_var = (M - 1) * M,  where M is the number of objectives
		nS[3] = str(n_var)

	nS[7] = str(1000)   # number of iterations/generations per simulation
	nS[8]  = str(30)  # number of runs
	nS.insert(2, str(iIndexNumberOfSimulation))  #inserts index number of simulation into appropriate position in list
	return (" ".join(nS))

 
fout = open("runBatchInWindowsAfterPreprocess.bat", "w")  #output of preprocessing batch file is stored here 
fout_CHPCSCRIPT = open("Preprocess/script.qsub", "w")  #output of preprocessing batch file is stored here 

countNSGA2 = 0 
countOMOPSO = 0
countMGPSO = 0
count = 0
for line in fileinput.input(['runBatchInWindows.bat']):
    strippedLine = line.strip()
    #print(strippedLine + "\n")
    if "echo"  not in strippedLine  and strippedLine != "":
    	if "mgpso" in strippedLine:
    		countMGPSO = countMGPSO + 1
    		count = countMGPSO + 800
    	elif "nsga2" in strippedLine:
    		countNSGA2 = countNSGA2 + 1
    		count = countNSGA2 
    	elif "omopso" in strippedLine:
    		countOMOPSO = countOMOPSO +  1
    		count = countOMOPSO + 400
    	else:  #used by any other algorithm that may be defined later
    		count = count + 1

 
    	strippedLine = preprocessLine(strippedLine, count)
    	iIndexNumberOfSimulation = strippedLine.split()[2]    
    	name = str(iIndexNumberOfSimulation) +  ".sh" 
    	fout_CHPCSCRIPT.write( "chmod u+x " +  name  + "\n")
    	fout_CHPC = open("Preprocess/"  +  name, "w")  # creates a shell script for CHPC server
    	fout_CHPC.write("module load chpc/python/3.7.0  \n" + strippedLine + "\n") 
    	fout_CHPC.close()
    	fout.write(strippedLine + "\n") 

fout.close() #closes output file
fout_CHPCSCRIPT.close() #closes output file


