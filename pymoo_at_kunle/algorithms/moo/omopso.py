import scipy 
import  numpy as np   
import math  

from pymoo.algorithms.moo.nsga2 import calc_crowding_distance 
from pymoo.algorithms.soo.nonconvex.pso import pso_equation
from pymoo.core.evaluator import Evaluator
from  pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.core.initialization import Initialization 
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.docs import parse_doc_string 
from pymoo.problems.many.dtlz import DTLZ1 
from pymoo.operators.sampling.lhs import LHS 
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem 
from pymoo.util.termination.default import MultiObjectiveDefaultTermination  
from pymoo.util.misc import termination_from_tuple   
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
#from pymoo.util.termination.max_gen import MaximumGenerationTermination 
from  pymoo_at_kunle.algorithms.base.pso import *  
from pymoo_at_kunle.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo_at_kunle.util.nds.fast_non_dominated_sort import paretoDominanceComparePopulations

 

class Mutation(): 
    def __init__(self, 
                T = 10,  # max number of generations
                t = 1 ,  # current generation
                pert = 0.01,  #perturbation factor, as called in Marde's referenced paper
                p_m = 0.1,     # probability of mutation
                problem = None
                ):
        self.T = T
        self.t = t
        self.pert = pert 
        self.p_m = p_m
        self.problem = problem
        #print(f"pert = {self.pert}, p_m = {self.p_m}")        
    def repair(self, position):
        position[position - self.problem.xl < 0 ] =  self.problem.xl[position - self.problem.xl < 0]
        position[position - self.problem.xu > 0 ] =  self.problem.xu [position - self.problem.xu > 0] 
        return position 
    def uniform_mutation(self, Xp):  
        #print(f"Uniform mutation - max gen: {self.T}, curr-gen= {self.t}")  
        def uniformMutateThis(indv): 
            randumStuffs = np.random.random(indv.shape[0])
            ind = np.where(randumStuffs < self.p_m) 
                  # indices of positions to be mutated
            randBits = (np.random.random(indv[ind].shape[0] ) - 0.5 ) * self.pert
            indv[ind] = indv[ind] + randBits
            return self.repair(indv)  # returns mutated individual      
        return np.apply_along_axis(uniformMutateThis , 1 , Xp) # returns mutated individuals     
        
    def non_uniform_mutation(self, Xp):  
        def delta(t, y, mutationrate = self.pert):
            d = math.pow(1 - t/ self.T,mutationrate)
            return y * (1 - np.power(np.random.random(y.shape[0]),d))              
        def nonUniformMutateThis(indv):             
            randumStuffs = np.random.random(indv.shape[0])             
            ind = np.where(randumStuffs < self.p_m) 
                    # indices of positions to be mutated  
            lbs = self.problem.xl[ind]  # lower bounds of dimensions/positions to mutate   
            ubs = self.problem.xu[ind]  # upper bounds of dimensions/positions to mutate  
            randomBits = np.random.random(ind[0].shape[0])
                 #ind[0] extracts an array from a tuple named ind
            ind_1 = ind[0][np.where(randomBits <= 0.5)]  
                    # indices of group 1 of positions to mutate
            ind_2 = ind[0][np.where(randomBits >  0.5)]
                    # indices of group 2 of positions to mutate             
            indv[ind_1] =  indv[ind_1] + delta(self.t,  self.problem.xu[ind_1] - indv[ind_1])
            indv[ind_2] = indv[ind_2]  -  delta(self.t,  indv[ind_2]  - self.problem.xl[ind_2])                          
            return self.repair(indv) # returns mutated individual         
        return np.apply_along_axis(nonUniformMutateThis, 1 , Xp) # returns mutated individuals

class repairExtended(Repair):
        def do(self, problem, positions, velocities, **kwargs): 
            self.problem = problem 
            self.positions = positions
            self.velocities = velocities          
            def repairPosition(position, velocity):               
                position[position - self.problem.xl < 0 ] =  self.problem.xl[position - self.problem.xl < 0]             
                velocity[position - self.problem.xl < 0] =  -1 *   velocity[position - self.problem.xl < 0] 
                position[position - self.problem.xu > 0 ] =  self.problem.xu[position - self.problem.xu > 0]                 
                velocity[position - self.problem.xu > 0] = -1 *    velocity[position - self.problem.xu > 0]              
                #velocity[np.abs(velocity) < 0.000001] =  np.random.uniform(0,1)  # adjusts for stationary particles
                return (position, velocity)            
            for  i  in  range(self.positions.shape[0]):
                    self.positions[i], self.velocities[i] = repairPosition(self.positions[i].copy(), self.velocities[i].copy())         
            return (self.positions, self.velocities)
             

class ImprovementReplacementMultiObjective(ImprovementReplacement):
    def _do(self, problem, pop, off, **kwargs):
        ret = np.full((len(pop), 1), False)
        pop_F, pop_CV = pop.get("F", "CV")
        off_F, off_CV = off.get("F", "CV")
        eps = 0.0
        # eps = calc_adapt_eps(pop)
        pop_feasible, off_feasible = pop_CV <= eps, off_CV <= eps 
        if problem.n_constr > 0:
            # 1) Both infeasible and constraints have been improved
            ret[(~pop_feasible & ~off_feasible) & (off_CV < pop_CV)] = True
            # 2) A solution became feasible
            ret[~pop_feasible & off_feasible] = True
             # 3) Both feasible but objective space value has improved
            ret[(pop_feasible & off_feasible) & paretoDominanceComparePopulations(off_F, pop_F) ] = True          
        else:            
            ret[paretoDominanceComparePopulations(off_F, pop_F) ] = True     
        # never allow duplicates to become part of the population when replacement is used
        _, _, is_duplicate = DefaultDuplicateElimination(epsilon=0.0).do(off, pop, return_indices=True)
        ret[is_duplicate] = False       
        return ret[:, 0]

class OMOPSO(Algorithm):     
        def __init__ (self, 
                archive_size =  20,           
                pop_size=25,
                epsilon = 0.01,    #used to insert leaders in epsilon archive                
                sampling=LHS(),
                w=0.1,
                c1=1.5,
                c2=2.0,
                adaptive=True,
                initial_velocity="zero",
                max_velocity_rate=1,
                pert = 0.5, # perturbation factor, so-called in Marde's referenced paper
                p_m = 0.8,  # probability of mutation, used by mutation operators
                repair=repairExtended(),               
                 **kwargs): 

                super().__init__(**kwargs)   
                self.__archive = [] # 2-D array used by OMOPSO     
                self.archive_size = archive_size           
                self.callback = None
                self.crowdingFactorsOfLeaders = []
                self.default_termination = MultiObjectiveDefaultTermination(n_last=10)
                self.epsilon = epsilon
                self.has_terminated = None
                self.history = None
                self.initialization = Initialization(sampling)                 
                self.initial_velocity = initial_velocity
                self.is_initialized = False
                self.__leaders = Population() # 2-D array used by OMOPSO      
                self.max_velocity_rate = max_velocity_rate
                self.n_gen = None 
                self.pert = pert # perturbation rate, mutation rate
                self.p_m = p_m
                self.pop_size = pop_size
                self.pop = None
                self.problem = None  
                self.repair = repair  
                self.termination =  None                        
                self.V_max = None     
                self.w = w
                self.c1 = c1
                self.c2 = c2          
     
        def getMasksOfObjectives(self,g):
            return np.random.choice(g, len(g) - 3, False)
        def getarchive(self):
            return self.__archive
        def computeCrowdingFactorsOfLeaders(self,leaders):
                self.crowdingFactorsOfLeaders = self.computeCrowdingFactors(leaders)                  
        def computeCrowdingFactors(self, pop):
                return calc_crowding_distance(pop.get("F"))
        def createleaders(self):          
            indicesOfLeaders = NonDominatedSorting(epsilon = 0.0).do(self.pop.get("F"), only_non_dominated_front=True , return_rank=False)
            self.__leaders =   self.pop[(indicesOfLeaders)] 
        def updateleaders(self, pop):
            pool = Population().merge(self.__leaders, pop)
            indicesOfUniqueLeaders = np.unique(pool.get("X"), True, axis = 0)[1]                 
            pool = pool[indicesOfUniqueLeaders]
            indicesOfLeaders= NonDominatedSorting(epsilon = 0.0).do(pool.get("F"), only_non_dominated_front=True , return_rank=False)
            if len(indicesOfLeaders) <=  self.pop_size:   
                self.__leaders =  pool[(indicesOfLeaders)] 
            else:
                crowdingFactorsOfLeaders =self.computeCrowdingFactors(pool[(indicesOfLeaders)])                 
                f= np.column_stack( (indicesOfLeaders, crowdingFactorsOfLeaders) )                             
                f = f[np.argsort(f[:,1])[::-1]]  # sorts array in decreasing order of second column, i.e. crowding factor
                f = f[:, 0]  # gets indices of the sorted leaders
                f = f[0: self.pop_size]  # gets a max of [pop-size] number of leaders               
                self.__leaders =  pool[(f.astype(int))] 
        def getIndicesOfSubSwarmToMutate(self, 
                             f, 
                             iPartIndentifier = 0 #  range of values: 0,1,2
                             ):
            d = np.arange(f.shape[0]) # computes row indices of f
            b =np.where(np.mod(d, 3) ==  iPartIndentifier) # computes indices of required subswarm
            return  b   # returns indices of particles          
        def sendleadersToEpsilonArchive(self, leaders): 
            if ( len(self.__archive)) > 0 :
                leaders = Population().merge(leaders, self.__archive)
            indicesOfUniqueLeaders = np.unique(leaders.get("F"), True, axis = 0)[1]                 
            leaders = leaders[indicesOfUniqueLeaders]
            indicesOfEpsilonNonDominatedLeaders = NonDominatedSorting(epsilon = self.epsilon).do(leaders.get("F"), only_non_dominated_front=True , return_rank=False)
            EpsilonNonDominatedLeaders = leaders[(indicesOfEpsilonNonDominatedLeaders)]             
            if (indicesOfEpsilonNonDominatedLeaders.shape[0] <= self.archive_size):                 
                self.__archive = EpsilonNonDominatedLeaders
                           # this will be true during initialization 
            else:               
                crowdingFactorsOfLeaders = calc_crowding_distance(leaders[(indicesOfEpsilonNonDominatedLeaders)].get("F"))   
                 #use above function to ensure that rOMOPSO does not use overriden method          
                f= np.column_stack( (indicesOfEpsilonNonDominatedLeaders, crowdingFactorsOfLeaders) )                             
                f = f[np.argsort(f[:,1])[::-1]]  # sorts array in decreasing order of second column, i.e. crowding factor
                f = f[:, 0]  # gets indices of the sorted leaders
                f = f[0: self.archive_size]  # gets a max of [archive_size]  leaders
                self.__archive =  leaders[(f.astype(int))]    
        def selectRandomLeaders(self):
            # randomly select  leaders and return the leaders to the 
            iPopSize = self.pop_size         
            leaders = Population(iPopSize)   # initializes to an array of empty population of leaders
            iNumberOfLeaders = len(self.__leaders)
            for  i in range(iPopSize):
                 g = np.array(range(iNumberOfLeaders)) # sets indices of particles in archive
                 if ( len(g) <= 1):  # only one particle exists as a potential leader
                    indx =  np.random.choice(g, 2, True) # gets indices of 2 randomly selected leaders, repetation allowed  
                 elif  len(g) >= 2:  # two or more potential leaders exist
                    indx =  np.random.choice(g, 2, False) # gets indices of 2 randomly selected leaders, repetation not allowed      

                 a = indx[0] #np.random.randint(0, iNumberOfLeaders)
                 b = indx[1] # np.random.randint(0, iNumberOfLeaders)   
                 if (self.crowdingFactorsOfLeaders[a] > self.crowdingFactorsOfLeaders[b]):
                    leaders[i] = self.__leaders[a] 
                 else:
                      leaders[i] = self.__leaders[b]
            return leaders
        def processImprovements(self, prob, pop, off):
            return ImprovementReplacementMultiObjective().do(prob, pop, off, return_indices=True)
        def _setup(self, problem, **kwargs):            
            self.V_max =  self.max_velocity_rate * (self.problem.xu - self.problem.xl)   
              #max velocity is defined per dimension in decision variable space 
            if isinstance(self.termination, MaximumFunctionCallTermination):
                n_gen = np.ceil((self.termination.n_max_evals - self.pop_size) / self.n_offsprings)
                self.termination = MaximumGenerationTermination(n_gen)
            #print(f"self.termination = {self.termination.n_max_gen}")    
        def _initialize_infill(self):             
              self.pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
              #print(f"prob (nvar):  {self.problem.n_var}, prob(nobj): {self.problem.n_obj}")        
              self.pop.set("n_gen", self.n_gen)
              Evaluator().eval(self.problem, self.pop) 
              #print(self.pop.get("F"))
              self.createleaders()
              self.sendleadersToEpsilonArchive(self.__leaders)     
              self.computeCrowdingFactorsOfLeaders(self.__leaders)                   
              return self.pop      
        def _initialize_advance(self, infills=None, **kwargs):                  
            pbest = self.pop #self.pop stores personal best positions of particles in swarm
            particles = pbest.copy()             
            if self.initial_velocity == "random":
                init_V = np.random.random((particles.shape[0], self.problem.n_var)) * self.V_max[None, :]    
            elif self.initial_velocity == "zero":
                init_V = np.zeros((particles.shape[0], self.problem.n_var)) 
            particles.set("V", init_V)
            self.particles = particles   
             # self.particles store current positions and velocities of particles in the swarm
            super()._initialize_advance(infills=infills, **kwargs)
        def _infill(self):
            oMutation = Mutation(T = self.termination.n_max_gen, t = self.n_gen, 
                  pert = self.pert, p_m = self.p_m, problem =  self.problem)
            problem, particles, pbest = self.problem, self.particles, self.pop 
            (X, V) = particles.get("X", "V")
            #if self.n_gen % 2000 == 0:
                #V1 =     np.random.random((particles.shape[0], self.problem.n_var)) * self.V_max[None, :]
            P_X = pbest.get("X")      
            sbest = self.selectRandomLeaders()           
            S_X = sbest.get("X") 
            n_s = len(P_X)  # number of particles or population size
            self.c1 =  1.5 +  (2.0 - 1.5) * np.random.random(  (n_s, 1))     
            self.c2 =  1.5 +  (2.0 - 1.5) * np.random.random( (n_s, 1))   
            self.w =   0.1 +  (0.5 - 0.1) * np.random.random( (n_s, 1))  
            Xp, Vp = pso_equation(X, P_X, S_X, V, self.V_max, self.w, self.c1, self.c2)    
            if ( self.problem.has_bounds()):
                Xp, Vp = self.repair.do(problem, Xp, Vp)  
            #print(Vp)
            i = self.getIndicesOfSubSwarmToMutate(Xp, 1)  # apply uniform mutation
            Xp[(i)] = oMutation.uniform_mutation(Xp[(i)])
            i = self.getIndicesOfSubSwarmToMutate(Xp, 2)  #  apply non uniform mutatation 
            Xp[(i)] = oMutation.non_uniform_mutation(Xp[(i)])           
            self.particles = Population.new(X=Xp, V=  Vp )
            Evaluator().eval(self.problem, self.particles)  # evaluates particles
            return self.particles # returns updated particles
        def _advance(self, infills=None, **kwargs):    
            # self.particles  = infills           
            has_improved = self.processImprovements(self.problem, self.pop, infills)
            # set the personal best which have been improved        
            self.pop[has_improved] = infills[has_improved].copy()    
            self.updateleaders(infills[has_improved].copy())
            self.sendleadersToEpsilonArchive(self.__leaders)            
            self.computeCrowdingFactorsOfLeaders(self.__leaders)    

        def _finalize(self):                 
            self.opt = self.__archive             
            Evaluator().eval(self.problem, self.opt)  # evaluates optimal solutions stored in archive  
                

parse_doc_string(OMOPSO.__init__) 
 


 
'''

a = 5
b = 40
print (f"add({a}, {b}) = {OMOPSO().add(a,b)} ")
print(np.random.random(size=(2,1)))

pop = Population(5)
for ind in pop:
	print(ind.X)

print( f" pop size = {len(pop)}")


class C:
    def __new__(cls, *args):
        print('Cls in __new__:', cls)
        print('Args in __new__:', args)
        # The `object` type __new__ method takes a single argument.
        return object.__new__(cls)

    def __init__(self, *args):
        print('type(self) in __init__:', type(self))
        print('Args in __init__:', args)

 
 

# os.environ['PYTHONPATH']  -> keep code and use after restarting o/s
 

import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

 '''

