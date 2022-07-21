import scipy 
import  numpy as np   
import math  
from pymoo.core.population import Population
from pymoo_at_kunle.factory  import get_performance_indicator
from pymoo.core.evaluator import Evaluator
from pymoo_at_kunle.algorithms.maoo.rmgpso import  rMGPSO
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance 
from pymoo_at_kunle.algorithms.maoo.promopso import getMasksOfObjectives

class prMGPSO(rMGPSO):
	def getMasksOfObjectives(self, g, F = None, iTypeOfObjectiveMeasure = None, _3selectedObjectives = None):    
		return getMasksOfObjectives(g, self.getarchive().get("F"), self.iTypeOfObjectiveMeasure, self._3selectedObjectives ) 

	def _initialize_infill(self):          
		self.pop = super()._initialize_infill() 	 
		if self.problem.n_obj  >  3:              
			if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD":           
			    self.p_s = np.full(self.problem.n_obj ,  1.0)  # potential of each objective set to 1  
			    self._igd = self.oIGD.do(self.pop.get("F"))              
		return self.pop


	def _setup(self, problem, **kwargs):    
	    super()._setup(self.problem, **kwargs)
	    if self.problem.n_obj  > 3: 
	     	if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD":                 
	            self.oIGD = get_performance_indicator("igd", self.pof)  
	            self.p_s = np.full(self.problem.n_obj ,  1)  # potential of each objective set to 1   
	            self.probs =  self.p_s / np.sum(self.p_s)
	            self._3selectedObjectives =   np.random.choice(self.problem.n_obj, 3, p =  self.probs, replace = False)  # roulette wheel selection of 3 preferred objectives
	
	def _advance(self, infills=None, **kwargs):
		super()._advance(infills, **kwargs)
		if self.problem.n_obj  >  3:			  
			if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD" :
				if  (self.iFrequency == 5 and self.n_gen % 5 == 1) or  self.iFrequency ==  1:
					n_igd = self.oIGD.do(self.getarchive().get("F"))
					if  n_igd - self._igd < 0:
						self.p_s[ self._3selectedObjectives] =    self.p_s[self._3selectedObjectives] +  abs(n_igd - self._igd ) / 3
					elif n_igd - self._igd >  0:
						_indx_non_selectedObjectives = np.delete(np.arange(self.problem.n_obj) , self._3selectedObjectives) 
						self.p_s[_indx_non_selectedObjectives] =    self.p_s[_indx_non_selectedObjectives] +  abs(n_igd - self._igd ) / 3
					probs = self.p_s / np.sum(self.p_s) # computes selection probabilities of objectives using their selection potentials
					self._3selectedObjectives = np.random.choice(self.problem.n_obj, 3, p =  probs, replace = False)
					self._igd = n_igd  # stores neww igd value to be used in next iteration       
         

	def __init__ (self, pop_size =25, iFrequency = 5, iTypeOfObjectiveMeasure = "IGD", pf  =None, **kwargs):
		self.iFrequency = int(iFrequency)		
		super().__init__(pop_size = pop_size, iFrequency = iFrequency,  **kwargs) 
		self.indxOfObjectivesToTurnOff = None 	
		self.iTypeOfObjectiveMeasure = iTypeOfObjectiveMeasure
		self._3selectedObjectives = None
		self._igd = None   #  applied if IGD performance measure is chosen by user/caller
		self.p_s = None    #  potentials of objectives,  first set inside _initialize_infill() 
		self.pof =pf
		self.n_genLastUsedForPartialDominance = -1
		self.name = "prmgpso"
	def computeCrowdingFactors(self, pop): 
		n_gen = self.n_gen
		iFrequency = self.iFrequency
		Evaluator().eval(self.problem, pop)	 
		F = pop.get("F")
		g = np.array(range(F.shape[1])) # sets indices of  objective functions	 
		if n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
			#this is the first generation/iteration, randomly select indices of 3 objective functions
			self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g) 
		if  iFrequency ==  1:
			#randomly select   objectives to mask off at every iteration/generation			 
			if  self.n_genLastUsedForPartialDominance !=  n_gen:
					self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)   
					self.n_genLastUsedForPartialDominance = n_gen 
		elif iFrequency == 5:
			#randomly select   objectives to mask off after every 5 iterations/generations
			if n_gen % 5 == 1:				 
				if  self.n_genLastUsedForPartialDominance !=  n_gen:
					self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)   
					self.n_genLastUsedForPartialDominance = n_gen  
		else:
			#use all objective functions
			self.indxOfObjectivesToTurnOff = () # do not mask off any objective
		#print(f"n_gen: {n_gen},  self.n_genLastUsedForPartialDominance: {self.n_genLastUsedForPartialDominance}, self.indxOfObjectivesToTurnOff:  {self.indxOfObjectivesToTurnOff}")
		F[:, self.indxOfObjectivesToTurnOff] = 0.0 # turn off selected objective
		return calc_crowding_distance(F)