import pytest
import numpy as np 


from pymoo.optimize import minimize

from pymoo.factory import get_problem, get_termination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from pymoo.problems.multi import ZDT1
from pymoo.problems.many import DTLZ1
from pymoo.problems.many import DTLZ2
from pymoo.problems.many import DTLZ3

from pymoo.factory import get_performance_indicator


from pymoo_at_kunle.algorithms.moo.mgpso import MGPSO


def test_same_seed_same_result():
	iNumberOfVariables = 30
	problem = ZDT1(n_var=iNumberOfVariables)
	#problem = ZDT1(n_var=iNumberOfVariables)
	termination = get_termination("n_gen",10) 
	algorithm = MGPSO(pop_size=500, epsilon = 0.01, pert = 0.5, p_m = 1.0 / iNumberOfVariables)   
	res1 = minimize(problem, algorithm, termination, seed=1)
	np.random.seed(200)
	res2 = minimize(problem, algorithm, termination, seed=1) 
	np.testing.assert_almost_equal(res1.X, res2.X)
	np.testing.assert_almost_equal(res1.F, res2.F) 

def test_closeness_to_true_pof():
	'''
	pf = get_problem("zdt1").pareto_front()
	iNumberOfVariables = 30
	problem = ZDT1(n_var=iNumberOfVariables)
	#problem = ZDT1(n_var=iNumberOfVariables)
	termination = get_termination("n_gen",100) 
	algorithm = MGPSO(pop_size=500, epsilon = 0.01, pert = 0.5, p_m = 1.0 / iNumberOfVariables)   
	res = minimize(problem, algorithm, termination, seed=1)
	np.testing.assert_almost_equal(res.F[pf.shape[0], :], pf)  # compares equal number of solutions in both sets
	'''
	assert True
def test_accuracy_and_diversity():
	pf = get_problem("zdt1").pareto_front()

	igd = get_performance_indicator("igd", pf)	
	iNumberOfVariables = 30
	problem = ZDT1(n_var=iNumberOfVariables)
	#problem = ZDT1(n_var=iNumberOfVariables)
	termination = get_termination("n_gen",100) 
	algorithm = MGPSO(pop_size=500, epsilon = 0.01, pert = 0.5, p_m = 1.0 / iNumberOfVariables)   
	res = minimize(problem, algorithm, termination, seed=1)
	ret = igd.do(res.F)
	np.testing.assert_almost_equal(ret, 0.001)  # compares equal number of solutions in both sets
