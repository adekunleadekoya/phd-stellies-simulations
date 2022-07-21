from pymoo.problems.many.dtlz import *  
class DTLZ2a(ScaledProblem):
    def __init__(self, n_var=10, n_obj=3, scale_factor=10, **kwargs):
        problem = DTLZ2(n_var=n_var, n_obj=n_obj, **kwargs)
        problem.type_var = float
        super().__init__(problem, scale_factor=scale_factor)
 