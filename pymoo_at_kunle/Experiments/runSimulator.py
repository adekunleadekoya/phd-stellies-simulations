
import numpy as np

import sys

from pymoo.optimize import minimize

from pymoo.factory import get_problem, get_termination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from pymoo.problems.multi import ZDT1

from pymoo.problems.many import DTLZ1
from pymoo.problems.many import DTLZ2
from pymoo.problems.many import DTLZ3
from pymoo.problems.many import DTLZ4
from pymoo.problems.many import DTLZ5
from pymoo.problems.many import DTLZ6
from pymoo.problems.many import DTLZ7
from pymoo_at_kunle.problems.many.dtlz import DTLZ2a

from pymoo.problems.many import WFG1
from pymoo.problems.many import WFG2
from pymoo.problems.many import WFG3
from pymoo.problems.many import WFG4
from pymoo.problems.many import WFG5
from pymoo.problems.many import WFG6
from pymoo.problems.many import WFG7
from pymoo.problems.many import WFG8
from pymoo.problems.many import WFG9


 

from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.visualization.scatter import Scatter



from pymoo_at_kunle.algorithms.moo.omopso import OMOPSO 
from pymoo_at_kunle.algorithms.moo.mgpso import MGPSO 

from pymoo_at_kunle.algorithms.maoo.rnsga2 import rNSGA2
from pymoo_at_kunle.algorithms.maoo.romopso import rOMOPSO
from pymoo_at_kunle.algorithms.maoo.rmgpso import  rMGPSO

from pymoo_at_kunle.algorithms.maoo.promopso import prOMOPSO
from pymoo_at_kunle.algorithms.maoo.prnsga2 import prNSGA2
from pymoo_at_kunle.algorithms.maoo.prmgpso import prMGPSO

from pymoo_at_kunle.factory  import get_performance_indicator

from pymoo_at_kunle.tests.test_sum  import  *

#from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
 
import pymoo_at_kunle.util.nds.non_dominated_sorting
from pymoo_at_kunle.util.nds.non_dominated_sorting import NonDominatedSorting

import multiprocessing
from pymoo.config import Config
Config.show_compile_hint = False

# python runSimulator.py 100  dtlz1 8 4 prnsga2  100 10 15 5 IGD    =>  an example of how to invoke the simulation from command prompt 
iIndexOfArgumentInList = 1 
iIndexNumberOfThisSimulation = sys.argv[iIndexOfArgumentInList]   # first index is 1 
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
prob_name = sys.argv[iIndexOfArgumentInList]    
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
iNumberOfDecisionVariables = sys.argv[iIndexOfArgumentInList]
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
iNumberOfObjectives = sys.argv[iIndexOfArgumentInList] 
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
optimizer =  sys.argv[iIndexOfArgumentInList]
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
pop_size =  sys.argv[iIndexOfArgumentInList]
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
iNumberOfIterations =  sys.argv[iIndexOfArgumentInList]
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
iNumberOfRuns = sys.argv[iIndexOfArgumentInList]
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
iFreq = sys.argv[++iIndexOfArgumentInList] 
iIndexOfArgumentInList = iIndexOfArgumentInList + 1
          # used by partial dominance driven optimizers to determine when to choose prefered objectives for partial dominance comparison
iTypeOfMeasureUsedByProposedImprovedOptimizer =  sys.argv[iIndexOfArgumentInList]  
parmsUsedInSimulation =  optimizer + " "  +  prob_name + " "   +   iFreq  + " " +  iTypeOfMeasureUsedByProposedImprovedOptimizer
'''
fo = open("foo.txt", "w")
fo.write(res)
fo.close()
'''

'''
def testEpsilonDominance(pop,  epsilon = 0.0):  
    fronts, ranks = NonDominatedSorting(epsilon = epsilon).do(pop.get("F"), only_non_dominated_front=False , return_rank=True)
    print(f"begin: epsilon = {epsilon}")
    rank = 0 
    for front in fronts:
     for ind in front:
        #print(f" indv({ind}):  X = {pop[ind].X},   F =  {pop[ind].F}, rank = {rank}")
        pass
         
     rank = rank + 1
    print(f"End: epsilon = {epsilon}")

'''
def testNonDomination():
    iPopSize = 20
    iNumberOfDecisionVariables = 5
    arrRandomSolutions =  np.random.random((iPopSize, iNumberOfDecisionVariables)) 
    problem = get_problem("dtlz1") 
    problem = DTLZ1(n_var=iNumberOfDecisionVariables, n_obj = 3) 
    pop = Population(iPopSize)
    pop.set("X", arrRandomSolutions)
    Evaluator().eval(problem, pop) 
    print(calc_crowding_distance(pop.get("F")))    
    #testEpsilonDominance(pop, 0.075) 
    #testEpsilonDominance(pop)
 
#testNonDomination()     ####  done testing with nonDomination operators and functions

'''

 ===================
iPopSize = 5
iNumberOfDecisionVariables = 5
arrRandomSolutions =  np.random.random((1, iNumberOfDecisionVariables))  
print(arrRandomSolutions)
print("============")
m = np.repeat(arrRandomSolutions, 5, axis = 0)
print(m)
pop = Population(iPopSize)
pop.set("X", m)

'''

 


def savePOFperRun(fp,  iIdOfRUn = 1,  POF = None): 
	fp.write("Run:" + str(iIdOfRUn) + "\n" +   np.array2string(POF) + "\n") 


def saveAggregatestats(algo, iNumberOfRuns = 1, iFreq = None, iTypeOfMeasure = None, prob = None, 
	iNumberOfDecisionVariables = None, iNumberOfObjectives = None, pop_size = 100,  iNumberOfIterations = 10, _stats = None): 
	#appends aggregate stats for each simulation to a file....file content is html based
 

	global iIndexNumberOfThisSimulation
	params = algo + "\t" + \
			    str(pop_size) +  "  "  + \
				str(iNumberOfIterations) +  "  "  + \
	            str(iNumberOfRuns) +  "  "  + \
	            str(iFreq)  +  "  "  + \
	            str(iTypeOfMeasure)  +  "  "  + \
	            str(prob)  +  "  "  + \
	            str(iNumberOfDecisionVariables)  +  "  "  + \
	            str(iNumberOfObjectives)  +  "  "  
	
	avg_spread = np.mean(_stats[0,:])
	std_spread =  np.std(_stats[0,:])

	avg_igd = np.mean(_stats[1,:])
	std_igd =  np.std(_stats[1,:])


	avg_igdplus = np.mean(_stats[2,:])
	std_igdplus =  np.std(_stats[2,:])

	'''

	filename = "Results/aggrStats.html"

	fo = open(filename, "a")
	output = "<tr>" +  \
	"<td>" + str(iIndexNumberOfThisSimulation) + "</td>" +  \
	"<td>" + str(params) + "</td>" +  \
	"<td>" + str(avg_spread) + "</td>" +  \
	"<td>" + str(std_spread) + "</td>"  +  \
	"<td>" + str(avg_igd) + "</td>"  +  \
	"<td>" + str(std_igd) + "</td>"  +  \
	"<td>" + str(avg_igdplus) + "</td>"  +  \
	"<td>" + str(std_igdplus) + "</td>"  +  \
	"</tr>"
	fo.write(output + "\n")
	fo.close() 

	'''

	if iFreq == -1 or iFreq == "-1":
		iFreq = "minusOne"
	#print(f"Spread: ({avg_spread}, {std_spread}), Igd: ({avg_igd}, {std_igd}), igdplus: ({avg_igdplus}, {std_igdplus})")
	filename = "Results/" +  algo +  "_" +   str(iNumberOfRuns) + "_"  + str(iFreq) + "_"  + str(iTypeOfMeasure) + "_" +  prob +  "_" +  str(iNumberOfDecisionVariables) +  "_"   +  str(iNumberOfObjectives)  + ".txt"
	#print(f"Store (aggstats):  {filename}")
	fo = open(filename, "w")
	output = str(iIndexNumberOfThisSimulation) + "\n" + \
	  str(params) + "\n" +  \
	  str(avg_spread) + "\n" +  \
	  str(std_spread) + "\n"  +  \
	  str(avg_igd) + "\n"  +  \
	  str(std_igd) + "\n"  +  \
	  str(avg_igdplus) + "\n"  +  \
	  str(std_igdplus) + "\n"  
	fo.write(output + "\n")
	

def  run(optimizer = None, pop_size = 50,  iNumberOfIterations = 100, problem  =None, iNumberOfVariables = 8, iNumberOfObjectives = 3, 
	iFreq = None,  iTypeOfMeasureUsedByProposedImprovedOptimizer = None): 	 
	optimizer = optimizer.lower()
	problem = problem.lower()
	iNumberOfObjectives = int(iNumberOfObjectives)
	iNumberOfVariables = int(iNumberOfVariables) 
	pop_size  = int(pop_size)
	iNumberOfIterations = int(iNumberOfIterations)
	iTypeOfMeasureUsedByProposedImprovedOptimizer = iTypeOfMeasureUsedByProposedImprovedOptimizer.upper()  

	if problem == "dtlz1":
		problem = DTLZ1(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem == "dtlz2":
		problem = DTLZ2(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem == "dtlz2a":
		problem = DTLZ2a(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem == "dtlz3":
		problem = DTLZ3(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem == "dtlz4":
		problem = DTLZ4(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem == "dtlz5":
		problem = DTLZ5(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)  # not yet fully implemented in pymoo ... skip for now
	elif problem == "dtlz6":
		problem = DTLZ6(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)  # not yet fully implemented in pymoo ... skip for now
	elif problem == "dtlz7":
		problem = DTLZ7(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)  # not yet fully implemented in pymoo ... skip for now

	elif problem ==  "wfg1":
		problem = WFG1(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg2":
		problem = WFG2(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg3":
		problem = WFG3(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg4":
		problem = WFG4(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg5":
		problem = WFG5(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg6":
		problem = WFG6(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem == "wfg7":
		problem = WFG7(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg8":
		problem = WFG8(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	elif problem ==  "wfg9":
		problem = WFG9(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)
	else: ### add more later
		#print(f"Unknwon problem: {problem}")
		problem = DTLZ1(n_var=iNumberOfVariables, n_obj = iNumberOfObjectives)  # default benchmark function  
	

	if optimizer == "prnsga2"  or  optimizer == "promopso" or  optimizer == "prmgpso":
		#print("Improved partial dominance-based optimizer called!")
		if (optimizer == "prnsga2"):
			algorithm = prNSGA2(pop_size=pop_size, iFrequency = iFreq,iTypeOfObjectiveMeasure =iTypeOfMeasureUsedByProposedImprovedOptimizer, pf = problem.pareto_front() )  
		elif optimizer == "promopso": 		 
			algorithm =  prOMOPSO(pop_size=pop_size, iFrequency = iFreq, iTypeOfObjectiveMeasure =iTypeOfMeasureUsedByProposedImprovedOptimizer, pf = problem.pareto_front()) 
		else:
			algorithm =  prMGPSO(pop_size=pop_size, iFrequency = iFreq, iTypeOfObjectiveMeasure =iTypeOfMeasureUsedByProposedImprovedOptimizer, pf = problem.pareto_front()) 
	elif optimizer == "rnsga2"  or  optimizer == "romopso" or optimizer == "rmgpso":
		 if optimizer == "rnsga2":
		 	algorithm = rNSGA2(pop_size=pop_size, iFrequency = iFreq)   
		 elif optimizer == "romopso":
		 	algorithm = rOMOPSO(pop_size=pop_size, iFrequency = iFreq)   
		 else:
		 	algorithm = rMGPSO(pop_size=pop_size, iFrequency = iFreq) 		 
	else:
		if optimizer == "nsga2":
			algorithm = NSGA2( pop_size=pop_size)   
		elif optimizer == "omopso":
			algorithm = OMOPSO(pop_size=pop_size)   
		else:
			algorithm = MGPSO(pop_size=pop_size)   
		 
	termination = get_termination("n_gen", iNumberOfIterations) 
	l  = [0,1,2]
	l[0] = algorithm 
	l[1] = problem 
	l[2] = termination
	return l


 
 
def  runInParallel(paramsForThisRun):  		 
		algorithm = paramsForThisRun[0]
		problem = paramsForThisRun[1]		
		termination = paramsForThisRun[2]
		idOfRun = paramsForThisRun[3]  
		#print(f"{algorithm.id_1}---- {idOfRun}")
		algorithm.id_1 = idOfRun
		#print (f"idOfRun:  {idOfRun}")  
		return minimize(problem,
                   algorithm,
                   termination,
                   seed=np.random.randint(1,1000),
                   verbose=False)
iNumberOfRuns = int(iNumberOfRuns) 
if __name__ == '__main__': 	 
	pool = multiprocessing.Pool(processes= iNumberOfRuns) 
	inputs=[]
	for e in  np.arange(iNumberOfRuns) + 1:
		l = run(optimizer = optimizer, pop_size = pop_size, iNumberOfIterations = iNumberOfIterations, problem = prob_name,
 	iNumberOfVariables = iNumberOfDecisionVariables, iNumberOfObjectives = iNumberOfObjectives, iFreq = iFreq, 
 iTypeOfMeasureUsedByProposedImprovedOptimizer = iTypeOfMeasureUsedByProposedImprovedOptimizer)

		algorithm = l[0]
		problem = l[1]
		termination = l[2]  

		inputs.append([algorithm, problem, termination, e])
	outputs = pool.map(runInParallel, inputs)
	pf = problem.pareto_front()
	_iNumberOfPerformanceMesuresUsedInStudy =  3
	_stats = np.full( ( _iNumberOfPerformanceMesuresUsedInStudy, iNumberOfRuns), 0.0)  #
	i = 0 # used to index runs in variable _stats
	o_igdplus = get_performance_indicator("igd+", pf) 
	o_igd = get_performance_indicator("igd", pf) 

	t_iFreq = iFreq
	t_iTypeOfMeasureUsedByProposedImprovedOptimizer = iTypeOfMeasureUsedByProposedImprovedOptimizer
	if (iFreq == "-1" or iFreq == -1):
		iFreq = 0
	if iTypeOfMeasureUsedByProposedImprovedOptimizer == "N/A":
		print("NA is true")
		iTypeOfMeasureUsedByProposedImprovedOptimizer = "NA"
	else:
		print("NA is false")

	filename = "POFs/" +  optimizer +  "_" +   str(iNumberOfRuns) + "_"  + str(iFreq) + "_"  + str(iTypeOfMeasureUsedByProposedImprovedOptimizer) + "_" +  prob_name +  "_" +  str(iNumberOfDecisionVariables) +  "_"   +  str(iNumberOfObjectives)  + ".txt"
	#print(f"filename: {filename}")
	fp = open(filename, "w")
	for  res in outputs:	
		s = get_performance_indicator("s", res.F) 
		s=  s.do()	 
		igdplus = o_igdplus.do(res.F)		
		igd = o_igd.do(res.F)
		_stats[0, i] = s
		_stats[1, i] = igd
		_stats[2, i] = igdplus
		savePOFperRun(fp, iIdOfRUn = i+1, POF = res.F)
		i = i + 1 # index of next run
	fp.close()  # close file used to store per run POFs

	saveAggregatestats(algo = optimizer, iNumberOfRuns = iNumberOfRuns, iFreq = t_iFreq, iTypeOfMeasure = iTypeOfMeasureUsedByProposedImprovedOptimizer,
	 prob=prob_name, iNumberOfDecisionVariables = iNumberOfDecisionVariables,
		 iNumberOfObjectives = iNumberOfObjectives,  pop_size =pop_size,  iNumberOfIterations = iNumberOfIterations,  _stats =  _stats)
 

 
'''

def testOMOPSOImplementation(optimizer = None, problem  =None, iFreq = None,  iTypeOfMeasureUsedByProposedImprovedOptimizer = None):
    iNumberOfVariables = 8
    #iNumberOfVariables = 30 
 
    #problem = DTLZ1(n_var=iNumberOfVariables, n_obj = 3)
    problem = WFG7(n_var=iNumberOfVariables, n_obj = 3)
    #problem = ZDT1(n_var=iNumberOfVariables)
    termination = get_termination("n_gen", 1000) 
    #algorithm = OMOPSO(pop_size=500, epsilon = 0.0075, pert = 0.5, p_m = 1.0 / iNumberOfVariables)    
    #algorithm = MGPSO(pop_size=100, epsilon = 0.01, pert = 0.5, p_m = 1.0 / iNumberOfVariables)   
    algorithm = NSGA2(n_var=iNumberOfVariables, pop_size=500)
    algorithm = rNSGA2(pop_size=100, iFrequency = 5)  
    #algorithm =  rOMOPSO(pop_size=100, iFrequency = 5) 
    #algorithm =  rMGPSO(pop_size=100, iFrequency = 5) 

    #algorithm =  prOMOPSO(pop_size=100, iFrequency = 5, iTypeOfObjectiveMeasure ="igd") 
    algorithm = prNSGA2(pop_size=100, iFrequency = 5)  
    algorithm =  prMGPSO(pop_size=200, iFrequency = 5)  
 
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=np.random.randint(1,1000),
                   verbose=False)  
    print("Result:")
    #print(res.F)
    pf = problem.pareto_front()
    #print(f"pf:  {pf}")       
    igdplus = get_performance_indicator("igdplus", pf)	
    igd = get_performance_indicator("igd", pf)
    s = get_performance_indicator("s", res.F) 
    igd = igd.do(res.F)
    igdplus = igdplus.do(res.F)
    print(f"igd = {igd},\nigdplus={igdplus} ")
    print("s =  ", s.do())
    fo= open("pof.txt", "w")
    fo.write( np.array2string(pf))
    fo.close()

    Scatter(legend=True).add(pf, label="True POF").add(res.F, label="Approximated POF").show() 
    pf = get_problem("zdt1").pareto_front()
    #print(pf)
    s = get_performance_indicator("s", pf) 
    print("test: ", s.do())
	 

'''


 
'''
pop = Population(5)
pop.set("X", np.random.random((5, 7)))

 
problem = ZDT1(n_var=30)
problem = get_problem("dtlz1") 
problem = DTLZ1(n_var=7, n_obj = 4)
resPop = Evaluator().eval(problem, pop)

for ind in resPop:
    print(ind.F)

print("===================\n")    

for ind in xyz:
    print(ind.F)    
''' 

'''

from pymoo.factory import get_problem
from pymoo.util.plotting import plot 
from pymoo.problems.many.wfg import * 

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")
problem = get_problem("dtlz1") 
problem =  WFG7(n_var=5, n_obj=3) 
algorithm = NSGA2(pop_size=8) 
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()


 
problem = get_problem("dtlz7")
problem =  WFG4(n_var=10, n_obj=3)
plot(problem.pareto_front(), no_fill=True)

 

''' 

'''

from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
from pymoo.factory import Ackley
from pymoo.optimize import minimize

problem = Ackley()

algorithm = PSO(max_velocity_rate=0.025)
 


res = minimize(problem,
               algorithm,
               callback=PSOAnimation(fname="pso.mp4", nth_gen=5),
               seed=1,
               save_history=True,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
 
 '''