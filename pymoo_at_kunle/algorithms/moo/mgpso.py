import scipy 
import  numpy as np   
import math  


from pymoo.factory import get_sampling
from pymoo.util import plotting
from pymoo.interface import sample

from pymoo.algorithms.moo.nsga2 import calc_crowding_distance 
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside

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
 
def pso_equation(X, P_X, S_X, V, A_X, L, V_max, w, c1, c2, c3,  r1=None, r2=None, r3 =None): 

    n_particles, n_var = X.shape


    if r1 is None:
        r1 = np.random.random((n_particles, n_var))

    if r2 is None:
        r2 = np.random.random((n_particles, n_var))

    if r3 is None:
        r3 = np.random.random((n_particles, n_var))

    inerta = w * V
    cognitive = c1 * r1 * (P_X - X)
    social = L * c2 * r2 * (S_X - X) 
    archive = (1-L) * c3 * r3 * (A_X - X)

    # calculate the velocity vector
    Vp = inerta + cognitive + social  + archive
    #Vp = set_to_bounds_if_outside(Vp, - V_max, V_max)  
   
    Xp = X + Vp

    return Xp, Vp

class repairExtended(Repair):
		def do(self, problem, positions, velocities, **kwargs): 
			self.problem = problem 
			self.positions = positions
			self.velocities = velocities          
			def repairPosition(position, velocity):               
				position[position - self.problem.xl < 0 ] =   self.problem.xl[position - self.problem.xl < 0]             
				velocity[position - self.problem.xl < 0] =  -1 *  velocity[position - self.problem.xl < 0] 
				position[position - self.problem.xu > 0 ] =   self.problem.xu[position - self.problem.xu > 0]                 
				velocity[position - self.problem.xu > 0] = -1 *  velocity[position - self.problem.xu > 0]              
				velocity[np.abs(velocity) < 0.00001] =  np.random.uniform(0,1)  # adjusts for stationary particles
				return (position, velocity)            
			for  i  in  range(self.positions.shape[0]):
					self.positions[i], self.velocities[i] = repairPosition(self.positions[i].copy(), self.velocities[i].copy())         
			return (self.positions, self.velocities)


class ImprovementReplacementMGPSO(ImprovementReplacement):
	def compareForBetterObjective(self, off_F, pop_F,indexOfObjective):
		f1 = off_F[:,indexOfObjective] #selects required column from 2d array
		f2 = pop_F[:,indexOfObjective] #selects required column from 2d array
		return  f1 <= f2 
		  # returns indices of particles where offspring is better for the [indexOfObjective] objective 
	def _do(self, problem, pop, off, indexOfObjective,  **kwargs):
		ret = np.full((len(pop), 1), False)
		pop_F, pop_CV = pop.get("F", "CV")
		off_F, off_CV = off.get("F", "CV")
		eps = 0.0
		# eps = calc_adapt_eps(pop)
		pop_feasible, off_feasible = pop_CV <= eps, off_CV <= eps
		if problem.n_constr > 0:
			# 1) Both infeasible and constraints have been 
			ret[(~pop_feasible & ~off_feasible) & (off_CV < pop_CV)] = True 
			# 2) A solution became feasible
			ret[~pop_feasible & off_feasible] = True
			# 3) Both feasible but objective space value has improved
			ret[(pop_feasible & off_feasible) & self.compareForBetterObjective(off_F, pop_F,indexOfObjective) ] = True
		else:
			ret[self.compareForBetterObjective(off_F, pop_F,indexOfObjective) ] = True
		# never allow duplicates to become part of the population when replacement is used
		_, _, is_duplicate = DefaultDuplicateElimination(epsilon=0.0).do(off, pop, return_indices=True)
		ret[is_duplicate] = False 		 
		return ret[:, 0]



class MGPSO(Algorithm):     
		def __init__ (self, 
				archive_size =  20,           
				pop_size=25,
				epsilon = 0.01,    #used to insert leaders in epsilon archive                
				sampling=FloatRandomSampling(),
				w=0.1,
				c1=1.5,
				c2=2.0,
				adaptive=True,
				initial_velocity="zero",
				max_velocity_rate=1.0,
				pert = 0.001, # perturbation factor, so-called in Marde's referenced paper
				p_m = 0.8,  # probability of mutation, used by mutation operators
				repair=repairExtended(),  
				improvementreplace = ImprovementReplacementMGPSO(),             
				 **kwargs): 

				super().__init__(**kwargs)   
				self.__archive = None # 2-D array used by OMOPSO     
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
				self.name ="mgpso"
				self.n_gen = None 
				self.pert = pert # perturbation rate, mutation rate
				self.p_m = p_m
				self.pop_size = pop_size
				 
				self.problem = None  
				self.repair = repair  
				self.termination =  None                        
				self.V_max = None     
				self.w = w
				self.c1 = c1
				self.c2 = c2    

				self.subswarms = None  
				self.subswarmsBest = None 
				self.sampling =  get_sampling('real_random')    
 
		def getBestInThisSubSwarm(self, F): 
			return np.argmin(F)  
			  # returns index of the least value in array F 		 
		def isDominatedByArchive(self, particle):
			f= np.row_stack( (self.__archive, particle.get("X")) ) #adds particle to the last row
			pop = Population(f.shape[0])
			pop.set("X", f)
			Evaluator().eval(self.problem, pop)
			indx = NonDominatedSorting(epsilon = 0.0).do(pop.get("F"), only_non_dominated_front=True , return_rank=False)
			return np.any(indx ==  f.shape[0] - 1) == False,  pop[indx].get("X")  
		def returnNonDominated(self, archive, particle):
			f= np.row_stack( (archive, particle.get("X")) )
			pop = Population(f.shape[0])
			pop.set("X", f)
			Evaluator().eval(self.problem, pop)
			indx = NonDominatedSorting(epsilon = 0.0).do(pop.get("F"), only_non_dominated_front=True , return_rank=False)
			return pop[indx].get("X")
		def insertIntoArchive(self, particle): 			 
			if self.__archive is None:			 	 
				self.__archive =  particle.get("X")  			    				 
				return 			
			bisDominated, nonDominatatedParticles = self.isDominatedByArchive(particle) 		 
			if (bisDominated):
				return # leaves archive intact bcos particle cannot enter the archive 
			if (len(self.__archive) < self.archive_size): 			 
				 self.__archive = nonDominatatedParticles # already included particle
			else:
				# archive is full already...crowd and remove a particle from the most crowded region 			 
				f=  self.__archive 			 
				pop = Population(f.shape[0])
				pop.set("X", f) 
				Evaluator().eval(self.problem, pop)			 			
				crowdingFactors = self.computeCrowdingFactors(pop)
				#calc_crowding_distance(pop.get("F"))  				
				indicesOfParticles = np.array(range(self.__archive.shape[0])) # indices of particles in all swarms  
				f= np.column_stack((indicesOfParticles, crowdingFactors) )  				
				f = f[np.argsort(f[:,1])[::-1]]  # sorts array in decreasing order of second column, i.e. crowding factor
				f = f[0: f.shape[0] -1,  :] # removes the last row, which is the particle with lowest crowding factor
				indx = f[:, 0]  # gets indices of qualified particles in archive
				self.__archive = self.returnNonDominated(pop[indx.astype(int)].get("X"), particle)	 			 			 
		def computeCrowdingFactors(self,pop): 	
			Evaluator().eval(self.problem, pop) 
			return calc_crowding_distance(pop.get("F"))  			
		def sendParticlesToArchive(self, particles): 		 
			f = particles.reshape(1, len(particles)) 	#converts 1-D to 2-D		 
			np.apply_along_axis(self.insertIntoArchive, 0 , f) 	
		def determineArchiveGuidesOfParticles(self, N):
			# N =>   number of particles
			def determineArchiveGuideOfParticle(i):					 
				archiveSize = self.__archive.shape[0] 			 
				g = np.array(range(archiveSize)) # sets indices of particles in archive
				if len(g) <= 2 :
					indx =  np.random.choice(g, 3, True) # gets indices of 3 randomly selected particles
				else:
					indx =  np.random.choice(g, 3, False) # gets indices of 3 randomly selected particles
				crowdingFactorsOfSelectedParticles = self.crowdingFactorsOfArchive[indx]
				maxCrowdingFactor = np.max(crowdingFactorsOfSelectedParticles)
				if  self.crowdingFactorsOfArchive[indx[0]] == maxCrowdingFactor:
					indxRequired = indx[0]
				elif self.crowdingFactorsOfArchive[indx[1]] == maxCrowdingFactor:
					indxRequired = indx[1]
				else:
					indxRequired = indx[2] 
				archiveGuide = self.__archive[indxRequired] 
				return archiveGuide
			arr =  np.array(range(N)) # sets indices of particles in a subswarm 
			f= arr.reshape(1, N)
			return np.apply_along_axis(determineArchiveGuideOfParticle, 1 , f)	   			 
			
		def _setup(self, problem, **kwargs): 
			self.V_max =  self.max_velocity_rate * (self.problem.xu - self.problem.xl) 	
		   
		def _initialize_infill(self):
			self.pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
			self.pop.set("n_gen", self.n_gen)
			Evaluator().eval(self.problem, self.pop)
			return self.pop 		
		def _initialize_advance(self, infills=None, **kwargs):
			pbest = self.pop #self.pop stores personal best positions of particles in swarm
			particles = pbest.copy()
			if self.initial_velocity == "random":
				init_V = np.random.random((particles.shape[0], self.problem.n_var)) * self.V_max[None, :]
			elif self.initial_velocity == "zero":
				init_V = np.zeros((particles.shape[0], self.problem.n_var)) 
			init_L =  X =  0.01 * sample(self.sampling, particles.shape[0], 1) 
			#0.01 *  np.random.random( (particles.shape[0], 1) )  # lambda
			arr = np.array(range(self.pop_size)) # indices of particles in all swarms
			self.subswarmsBest = np.full(self.problem.n_obj, None) 		 
			for i in range(self.problem.n_obj):
				indx = np.where(np.mod(arr, self.problem.n_obj) == i )
				   # indices of particles in current subswarm
				objValues = particles[indx].get("F") 
				indxBest = self.getBestInThisSubSwarm(objValues[:,i]) 				 
				subSwarm = particles[indx] 
				self.subswarmsBest[i] = subSwarm[int(indxBest)]  		 
			particles.set("V", init_V)
			particles.set("L", init_L)
			self.particles = particles 

		def _infill(self): 			 
			particles, pbest =  self.particles, self.pop
			(X, V) = particles.get("X", "V") 
			arr = np.array(range(self.pop_size)) # array stores indices of particles in all swarms 
			for i in range(self.problem.n_obj):
				indx = np.where(np.mod(arr, self.problem.n_obj) == i )
				# indices of particles in current subswarm  			 		 
				has_improved = ImprovementReplacementMGPSO().do(self.problem, pbest[indx], particles[indx] , indexOfObjective = i, return_indices=True )
				has_improved = indx[0][has_improved] #gets indices of particles that improved in subswarm
				pbest[has_improved]= particles[has_improved] 
				objValues = pbest[indx].get("F")
				indxBest = self.getBestInThisSubSwarm(objValues[:,i]) 				 
				indxBest = indx[0][int(indxBest)] 
				if pbest[indxBest].get("F")[i] <= self.subswarmsBest[i].get("F")[i]:
					self.subswarmsBest[i] = pbest[indxBest] 
				self.pop[has_improved] = pbest[has_improved] #updates self.pop with improved particles in subswarm
				self.sendParticlesToArchive(particles[indx]) 	
			f=  self.__archive  
			pop = Population(f.shape[0])
			pop.set("X", f)
			Evaluator().eval(self.problem, pop)			 		 
			#print (f"gen: {self.n_gen} ")
			return self.pop			
		def _advance(self, infills=None, **kwargs):
			particles, pbest = self.particles, infills	   
			(X, V, L) = particles.get("X", "V", "L") 
			if self.n_gen % 2000 == 0: 			 
				V1 =  np.random.random((particles.shape[0], self.problem.n_var)) * self.V_max[None, :] 				
			P_X = pbest.get("X") 			 
			f=  self.__archive  
			pop = Population(f.shape[0])
			pop.set("X", f) 
			self.crowdingFactorsOfArchive = self.computeCrowdingFactors(pop) 			
			arr = np.array(range(self.pop_size)) # arr stores indices of particles in all swarms  
			for i in range(self.problem.n_obj):
				indx = np.where(np.mod(arr, self.problem.n_obj) == i ) 			 
				n_s = len(indx[0])  #  size of subswarm
				self.c1 =  1.5 +   (2.0 - 1.5) * sample(self.sampling, n_s, 1)				 
				self.c2 =  1.5 +  (2.0 - 1.5) * sample(self.sampling, n_s, 1)
				self.c3 =  1.5 +  (2.0 - 1.5) * sample(self.sampling, n_s, 1) 
				self.w =   0.1 +  (0.5 - 0.1) * sample(self.sampling, n_s, 1) 
				XX = self.subswarmsBest[i].get("X") 		 
				XX = (XX.reshape(1, XX.shape[0])) 			 
				SX = np.repeat(XX, n_s, axis = 0 ) 			 
				      # above uses same social best for all particles in same subswarm			 
				A_X = self.determineArchiveGuidesOfParticles(n_s) 
				X[indx], V[indx] = pso_equation(X[indx], P_X[indx], SX, V[indx], A_X, L[indx], self.V_max, self.w, self.c1, self.c2, self.c3 )  
					#indices of particles in this subswarm 	
			if  self.problem.has_bounds():
				X, V = self.repair.do(self.problem, X, V)  
			particles.set("X", X)
			particles.set("V", V) 
			Evaluator().eval(self.problem, particles) #updates objective values of particles based on updated X
			self.particles = particles 				
		def _finalize(self):
			return self.getarchive()  # returns archive, where optimal solutions are stored 		 
			 
		def getarchive(self):
			f=  self.__archive  
			pop = Population(f.shape[0])
			pop.set("X", f)
			Evaluator().eval(self.problem, pop)	# evaluates optimal solutions stored in archive	 
			return  pop # returns optimal solutions stored in archive
		
		  

parse_doc_string(MGPSO.__init__) 






 
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

