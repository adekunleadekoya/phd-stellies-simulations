import scipy 
import  numpy as np   
import math  

from pymoo.core.evaluator import Evaluator
from pymoo_at_kunle.algorithms.moo.mgpso import MGPSO
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance 
class rMGPSO(MGPSO):
	def __init__ (self, pop_size =25, iFrequency = 5,  **kwargs):
		self.iFrequency = int(iFrequency)		
		super().__init__(pop_size = pop_size, **kwargs) 
		self.indxOfObjectivesToTurnOff = None
		self.n_genLastUsedForPartialDominance = -1
		self.name = "rmgpso" 
	def computeCrowdingFactors(self, pop): 
		n_gen = self.n_gen
		iFrequency = self.iFrequency
		Evaluator().eval(self.problem, pop)	 
		F = pop.get("F")
		g = np.array(range(F.shape[1])) # sets indices of  objective functions
		def getMasksOfObjectives(g):
			indx = np.random.choice(g, len(g) - 3, False)
			return indx 
		if n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
			#this is the first generation/iteration, randomly select indices of 3 objective functions
			self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g) 
		 
		if  iFrequency ==  1:
			#randomly select   objectives to mask off at every iteration/generation
			if  self.n_genLastUsedForPartialDominance !=  n_gen:
					self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g)   
					self.n_genLastUsedForPartialDominance = n_gen 
		elif iFrequency == 5:
			#randomly select   objectives to mask off after every 5 iterations/generations
			if n_gen % 5 == 1:
				#select()  
				if  self.n_genLastUsedForPartialDominance !=  n_gen:
					self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g)   
					self.n_genLastUsedForPartialDominance = n_gen     
			 
		else:
			#use all objective functions
			self.indxOfObjectivesToTurnOff = () # do not mask off any objective
		#print(f" n_gen: {n_gen}, iFrequency: {iFrequency}, self.indxOfObjectivesToTurnOff: {self.indxOfObjectivesToTurnOff}")
		F[:, self.indxOfObjectivesToTurnOff] = 0.0 # turn off selected objective
		return calc_crowding_distance(F)