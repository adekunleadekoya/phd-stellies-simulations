import numpy as np
from pymoo_at_kunle.factory  import get_performance_indicator
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance 
from pymoo.core.evaluator import Evaluator
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.dominator import Dominator

from pymoo.core.population import Population
from pymoo_at_kunle.algorithms.maoo.rnsga2 import rNSGA2
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting 
from pymoo_at_kunle.algorithms.maoo.promopso import getMasksOfObjectives

def getMasksOfObjectives_2(g, pop_F,iTypeOfObjectiveMeasure, _3selectedObjectives):
    return   getMasksOfObjectives(g,  pop_F, iTypeOfObjectiveMeasure, _3selectedObjectives)   

class RankAndCrowdingSurvival(Survival):

    def __init__(self, iFrequency, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.iFrequency = iFrequency
        self.indx = None
        self._3selectedObjectives = None 
        self.iTypeOfObjectiveMeasure = None
        self.n_genLastUsedForPartialDominance = -1  # used in partial dominance when selecting preferred objectives  

    def _do(self, problem, pop,  n_survive=None, algorithm = None,   **kwargs):
        # get the objective space values and objects    
        #print(f"hey:2 - {self.iTypeOfObjectiveMeasure}")   
        F = pop.get("F").astype(float, copy=False)
        g = np.array(range(F.shape[1])) # sets indices of  objective functions        
       
        n_gen = 1
        iFrequency = int(self.iFrequency)    

        if algorithm is not None:
        	n_gen = int(algorithm.n_gen) 
        if n_gen == 1 and self.indx is None:
        	#this is the first generation/iteration, randomly select indices of 3 objective functions         	
        	self.indx =  getMasksOfObjectives(g, F,self.iTypeOfObjectiveMeasure, self._3selectedObjectives)          	      	
        if  iFrequency ==  1:          
        	#randomly select   objectives to mask off at every iteration/generation 
            if  self.n_genLastUsedForPartialDominance !=  n_gen: 
                        self.indx =  getMasksOfObjectives(g,F,self.iTypeOfObjectiveMeasure, self._3selectedObjectives)    
                        self.n_genLastUsedForPartialDominance = n_gen             	
        elif iFrequency == 5:
            #self.indx = ()
            if n_gen % 5 == 1:
                if  self.n_genLastUsedForPartialDominance !=  n_gen: 
                        self.indx =  getMasksOfObjectives(g,F,self.iTypeOfObjectiveMeasure, self._3selectedObjectives)  
                        self.n_genLastUsedForPartialDominance = n_gen                    
        else:
        	#use all objective functions
        	self.indx = () # do not mask off any objective     

        #print(f"n_gen: {n_gen}, self.n_genLastUsedForPartialDominance: {self.n_genLastUsedForPartialDominance}, self.indx: {self.indx}")   
        
        # the final indices of surviving individuals
        survivors = [] 
        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive) 
        for k, front in enumerate(fronts): 
            # calculate the crowding distance of the front
            t = F.copy() 
            F = np.delete(F, self.indx , 1 )
            #print(f"cut (start): {F} \n cut(end) \n")        
            #F[:, self.indx] = 0.0  #turn-off columns in self.indx 
            crowding_of_front = calc_crowding_distance(F[front, :])
            F = t #restores columns that were turned off       

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j]) 
            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))] 
            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front)) 
            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]

class prNSGA2(rNSGA2): 
    def getMasksOfObjectives(self, g, F = None, iTypeOfObjectiveMeasure = None, _3selectedObjectives = None):  
        #print(f"here 1:   {self.iTypeOfObjectiveMeasure}, {self._3selectedObjectives} ")  
        return getMasksOfObjectives(g, self.pop.get("F"), self.iTypeOfObjectiveMeasure, self._3selectedObjectives ) 
    def _initialize_infill(self):  
        #print("1")        
        self.pop = super()._initialize_infill()   
        Evaluator().eval(self.problem, self.pop)       
        if self.problem.n_obj  >  3:              
            if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD":                     
                self.p_s = np.full(self.problem.n_obj ,  1.0)  # potential of each objective set to 1  
                self._igd =   self.oIGD.do(self.pop.get("F"))                                
        return self.pop
    def _setup(self, problem, **kwargs):              
        super()._setup(self.problem, **kwargs)
        if self.problem.n_obj  > 3: 
            if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD" :               
                    self.oIGD = get_performance_indicator("igd", self.pof)
                    self.p_s = np.full(self.problem.n_obj ,  1)  # potential of each objective set to 1   
                    self.probs =  self.p_s / np.sum(self.p_s)
                    self._3selectedObjectives =   np.random.choice(self.problem.n_obj, 3, p =  self.probs, replace = False)  # roulette wheel selection of 3 preferred objectives
    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.pop = Population.merge(self.pop, infills)  
        if self.problem.n_obj  >  3:   
            Evaluator().eval(self.problem, self.pop)    
            if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD" :             
                if  (self.iFrequency == 5 and self.n_gen % 5 == 1) or  self.iFrequency ==  1:                 
                    n_igd = self.oIGD.do(self.pop.get("F"))
                    if   n_igd - self._igd  < 0:
                        # improvement in igd                    
                        self.p_s[ self._3selectedObjectives] =    self.p_s[self._3selectedObjectives] +  abs(n_igd - self._igd ) / 3                   
                    elif n_igd - self._igd >  0:
                        #deterioration in igd
                        _indx_non_selectedObjectives = np.delete(np.arange(self.problem.n_obj) , self._3selectedObjectives) 
                        self.p_s[_indx_non_selectedObjectives] =    self.p_s[_indx_non_selectedObjectives] +  abs(n_igd - self._igd ) / 3
                    probs = self.p_s / np.sum(self.p_s) # computes selection probabilities of objectives using their selection potentials
                    self._3selectedObjectives = np.random.choice(self.problem.n_obj, 3, p =  probs, replace = False) 
                                 # above line uses roulette wheel to select 3 preferred objectives
                    self._igd = n_igd  # stores neww igd value to be used in next iteration   

        self.survival._3selectedObjectives = self._3selectedObjectives     
        self.survival.iTypeOfObjectiveMeasure = self.iTypeOfObjectiveMeasure 
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)
      

    def  _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:            
            self.survival._3selectedObjectives = self._3selectedObjectives     
            self.survival.iTypeOfObjectiveMeasure = self.iTypeOfObjectiveMeasure 
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills))
          
            
    def __init__(self, pop_size=100, iFrequency = 1, survival=  None,  iTypeOfObjectiveMeasure = "IGD", pf = None, **kwargs):
            iFrequency = int(iFrequency)
            self.survival = RankAndCrowdingSurvival(iFrequency = iFrequency)
            self.iTypeOfObjectiveMeasure = iTypeOfObjectiveMeasure           
            super().__init__(pop_size = pop_size,  survival = self.survival, iFrequency = iFrequency,  **kwargs)
            self.name = "prnsga2"
            self._3selectedObjectives = None
            self._igd = None   #  applied if IGD performance measure is chosen by user/caller
            self.p_s = None    #  potentials of objectives,  first set inside _initialize_infill()     
            self.pof = pf      #  stores the true POF of the problem being optimized
            self.n_genLastUsedForPartialDominance = -1  # used in partial dominance when selecting preferred objectives            
