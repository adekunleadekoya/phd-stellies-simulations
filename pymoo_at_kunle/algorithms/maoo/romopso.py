import scipy 
import  numpy as np   
import math  

from pymoo_at_kunle.algorithms.moo.omopso import OMOPSO
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.duplicate import DefaultDuplicateElimination

from pymoo_at_kunle.util.nds.fast_non_dominated_sort import paretoDominanceComparePopulations
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance  

from pymoo.docs import parse_doc_string 
class ImprovementReplacementMultiObjective(ImprovementReplacement): 
    def __init__(self):
        self.indxOfObjectivesToTurnOff = None 
        self.n_genLastUsedForPartialDominance = -1
    def do(self, problem, pop, off, n_gen, iFrequency,  **kwargs):         
        ret = np.full((len(pop), 1), False)
        pop_F, pop_CV = pop.get("F", "CV")
        off_F, off_CV = off.get("F", "CV")
        eps = 0.0   
        # eps = calc_adapt_eps(pop)         
        g = np.array(range(pop_F.shape[1])) # sets indices of  objective functions      
        def getMasksOfObjectives(g):
            return np.random.choice(g, len(g) - 3, False)     
        if n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
          #this is the first generation/iteration, randomly select indices of 3 objective functions           
           self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g)  
        if  iFrequency ==  1:
           #randomly select   objectives to mask off at every iteration/generation           
           if  self.n_genLastUsedForPartialDominance != n_gen: 
                    self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g)
                    self.n_genLastUsedForPartialDominance = n_gen
        elif iFrequency == 5:
           #randomly select   objectives to mask off after every 5 iterations/generations
           if n_gen % 5 == 1:        
             if  self.n_genLastUsedForPartialDominance != n_gen: 
                    self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g)
                    self.n_genLastUsedForPartialDominance = n_gen                        
        else:
              #use all objective functions
              self.indxOfObjectivesToTurnOff = () # do not mask off any objective  

        #print(f" n_gen: {n_gen}, self.n_genLastUsedForPartialDominance: {self.n_genLastUsedForPartialDominance},  self.indxOfObjectivesToTurnOff:  {self.indxOfObjectivesToTurnOff }")   
  
        #pop_F = np.delete(pop_F, self.indxOfObjectivesToTurnOff , 1) # deletes objectives that are not selected
        #off_F = np.delete(off_F, self.indxOfObjectivesToTurnOff , 1) # deletes objectives that are not selected 

        pop_F[:, self.indxOfObjectivesToTurnOff] = 0.0
        off_F[:, self.indxOfObjectivesToTurnOff] = 0.0


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

class rOMOPSO(OMOPSO):   # this class implements partial dominance as per Marde's referenced paper    
            
    def computeCrowdingFactors(self, pop):
            n_gen = self.n_gen
            iFrequency = int(self.iFrequency)             
            F = pop.get("F")
            g = np.array(range(F.shape[1])) # sets indices of  objective functions          
            if n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
                #this is the first generation/iteration, randomly select indices of 3 objective functions           
                self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)  

            if  iFrequency ==  1:
                #randomly select   objectives to mask off at every iteration/generation   
                if  self.n_genLastUsedForPartialDominance != n_gen: 
                    self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)
                    self.n_genLastUsedForPartialDominance = n_gen 
            elif iFrequency == 5:
                #randomly select   objectives to mask off after every 5 iterations/generations
                if n_gen  % 5 == 1:             
                    if  self.n_genLastUsedForPartialDominance != n_gen: 
                        self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)   
                        self.n_genLastUsedForPartialDominance = n_gen
            else:
                 #use all objective functions
                self.indxOfObjectivesToTurnOff = () # do not mask off any objective   

            
        
            return calc_crowding_distance(np.delete(F,self.indxOfObjectivesToTurnOff, 1 ))

    def __init__ (self, pop_size =25,
	  	            iFrequency = 5,               
                    **kwargs):
            self.iFrequency = int(iFrequency)
            self.improvementreplacement = ImprovementReplacementMultiObjective()
            super().__init__(pop_size = pop_size, **kwargs) 
            self.indxOfObjectivesToTurnOff = None 
            self.n_genLastUsedForPartialDominance = -1  #used to ensure correctness in masking off of objectives
            self.name = "rOMOPSO"  
    def processImprovements(self, prob, pop, off): 
            return  self.improvementreplacement.do(prob, pop, off, self.n_gen, self.iFrequency, return_indices=True)