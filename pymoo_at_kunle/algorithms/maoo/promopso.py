import scipy 
import  numpy as np   
import math  
from pymoo_at_kunle.algorithms.maoo.romopso import rOMOPSO
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.duplicate import DefaultDuplicateElimination 
from pymoo_at_kunle.util.nds.fast_non_dominated_sort import paretoDominanceComparePopulations
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance  
from pymoo_at_kunle.factory  import get_performance_indicator
from pymoo.core.evaluator import Evaluator 
def getObjectivesToTurnOffBasedOnSpread(F):
            res = np.full(F.shape[1], np.nan)    
            for i in range(F.shape[1]):
                nF = F.copy()
                nF[:, np.arange(F.shape[1]) !=  i] = 0.0               
                s = get_performance_indicator("s", nF) # computes SPREAD for given objective
                res[i] = s.do()              
            probs = 1 - np.divide(res,max(res)) # the lower the spread the preferred the objective function
            probs = np.divide(probs, np.sum(probs))  # normalizes spread measures and converts to probability measures
            rt = np.random.choice(len(res), 3, p =  probs, replace = False)  # roulette wheel selection of 3 preferred objectives
            rt = np.delete( np.arange(len(res)) , rt)  #computes the objectives that are not preferred, i.e. to be masked off by the caller of this method
            return rt.astype(int) #  returns the indices of the objectives  to mask off 
def getMasksOfObjectives(g,archive_F, iTypeOfObjectiveMeasure, _3selectedObjectives):              
            # g =>  indices of objective functions 
            n_obj =  len(g) # number of objectives
            rt = () 
            if ( n_obj  <= 3):
                return   rt # non of the objectives will be masked bcos the problem is not a MaOP                   
            if ( np.char.upper(iTypeOfObjectiveMeasure) ==  "SPREAD" ):                  
                rt = getObjectivesToTurnOffBasedOnSpread(archive_F)             
            elif (np.char.upper(iTypeOfObjectiveMeasure) ==  "IGD"):                                
                # _3selectedObjectives is used here to compute the objectives to masks off
                rt =    np.delete( np.arange(n_obj) , _3selectedObjectives).astype(int) # returns indices of non-selected objectives
            return rt            
class ImprovementReplacementMultiObjective(ImprovementReplacement):
    def __init__(self):
        self.indxOfObjectivesToTurnOff = None      
        self.iTypeOfObjectiveMeasure = None 
        self.n_genLastUsedForPartialDominance = -1 # used to ensure correctness in masking-off of objectives
    def do(self, archive, _3selectedObjectives, p_s, problem, pop, off, n_gen, iFrequency, iTypeOfObjectiveMeasure, pf,  **kwargs):  
        # p_s =>  potential of each objective is stored in this array/list         
        ret = np.full((len(pop), 1), False)
        pop_F, pop_CV = pop.get("F", "CV")
        off_F, off_CV = off.get("F", "CV")
        Evaluator().eval(problem, archive)  # evaluates self.pop    
        archive_F = archive.get("F")         
        eps = 0.0   
        # eps = calc_adapt_eps(pop)   
        self.iTypeOfObjectiveMeasure = iTypeOfObjectiveMeasure      
        g = np.array(range(pop_F.shape[1])) # sets indices of  objective functions    
        if n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
          #this is the first generation/iteration, randomly select indices of 3 objective functions           
            self.indxOfObjectivesToTurnOff = getMasksOfObjectives(g,archive_F,iTypeOfObjectiveMeasure,_3selectedObjectives)  
        if  iFrequency ==  1:
           #randomly select   objectives to mask off at every iteration/generation    
           if  self.n_genLastUsedForPartialDominance != n_gen:
                        self.indxOfObjectivesToTurnOff = getMasksOfObjectives(g,archive_F,iTypeOfObjectiveMeasure,_3selectedObjectives)
                        self.n_genLastUsedForPartialDominance = n_gen       
          
        elif iFrequency == 5:
           #randomly select   objectives to mask off after every 5 iterations/generations
           if n_gen % 5 == 1:
                if  self.n_genLastUsedForPartialDominance != n_gen:
                        self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g,archive_F,iTypeOfObjectiveMeasure,_3selectedObjectives)  
                        self.n_genLastUsedForPartialDominance = n_gen         
        else:
            #use all objective functions
            self.indxOfObjectivesToTurnOff = ()

        #print(f" n_gen: {n_gen}, self.n_genLastUsedForPartialDominance: {self.n_genLastUsedForPartialDominance}, self.indxOfObjectivesToTurnOff: {self.indxOfObjectivesToTurnOff}")

        # pop_F = np.delete(pop_F, self.indxOfObjectivesToTurnOff ,1 )  #deletes objectives that are not selected
        #off_F = np.delete(off_F, self.indxOfObjectivesToTurnOff ,1 )  #same as above line    

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
class prOMOPSO(rOMOPSO): 
    # this class implements my proposed improvements to partial dominance     
    def __init__ (self, pop_size =25, iFrequency = 5, iTypeOfObjectiveMeasure = "SPREAD",  pf  =None,  **kwargs): 		  
        super().__init__(pop_size = pop_size, iFrequency = iFrequency, **kwargs)
        self.name = "prOMOPSO"
        self.iFrequency = int(iFrequency)
        self.iTypeOfObjectiveMeasure = iTypeOfObjectiveMeasure
        self.improvementreplacement = ImprovementReplacementMultiObjective()
        self.indxOfObjectivesToTurnOff = None 
        self._3selectedObjectives =   None
        #print(f"\n========= {pf}")
        self.pof = pf #  employed  when IGD measure is used to select preferred objectives
        self._igd = None   #  applied if IGD performance measure is chosen by user/caller
        self.p_s = None    #  potentials of objectives,  first set inside _initialize_infill()
        self.n_genLastUsedForPartialDominance = -1 # used to ensure correctness in masking off of objectives
    def processImprovements(self, prob, pop, off):	 
        return  self.improvementreplacement.do(self.getarchive(),self._3selectedObjectives, self.p_s, prob, pop, off, self.n_gen, self.iFrequency,  self.iTypeOfObjectiveMeasure, self.problem.pareto_front(), return_indices=True)
    def computeCrowdingFactors(self, pop):         
        archive = self.getarchive()
        Evaluator().eval(self.problem, archive)  # evaluates self.pop    
        archive_F = archive.get("F")
        iTypeOfObjectiveMeasure = self.iTypeOfObjectiveMeasure         
        pop_F = pop.get("F")       
        iFrequency = self.iFrequency             
        F = pop.get("F")
        g = np.array(range(F.shape[1])) # sets indices of  objective functions     
        if self.n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
            #this is the first generation/iteration, randomly select indices of 3 objective functions           
            self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g,archive_F,iTypeOfObjectiveMeasure, self._3selectedObjectives)  
        if  iFrequency ==  1:
            #randomly select   objectives to mask off at every iteration/generation   
            if  self.n_genLastUsedForPartialDominance != self.n_gen: 
                        self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g,archive_F,iTypeOfObjectiveMeasure, self._3selectedObjectives)
                        self.n_genLastUsedForPartialDominance = self.n_gen     
            
        elif iFrequency == 5:
            #randomly select   objectives to mask off after every 5 iterations/generations
            if self.n_gen % 5 == 1:
                #select()  
                if  self.n_genLastUsedForPartialDominance != self.n_gen: 
                        self.indxOfObjectivesToTurnOff =  getMasksOfObjectives(g,archive_F,iTypeOfObjectiveMeasure, self._3selectedObjectives)  
                        self.n_genLastUsedForPartialDominance = self.n_gen         
        else:
             #use all objective functions
            self.indxOfObjectivesToTurnOff = () # do not mask off any objective      

        F[:, self.indxOfObjectivesToTurnOff] = 0.0       
        return calc_crowding_distance(F)       
    def _setup(self, problem, **kwargs):        
        super()._setup(self.problem, **kwargs)
        if self.problem.n_obj  > 3:  
            if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD":                 
                self.oIGD = get_performance_indicator("igd", self.pof)  
                self.p_s = np.full(self.problem.n_obj ,  1)  # potential of each objective set to 1   
                self.probs =  self.p_s / np.sum(self.p_s)
                self._3selectedObjectives =   np.random.choice(self.problem.n_obj, 3, p =  self.probs, replace = False)  # roulette wheel selection of 3 preferred objectives
    def _initialize_infill(self):          
        self.pop = super()._initialize_infill()         
        if self.problem.n_obj  >  3:           
            if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD":           
                self.p_s = np.full(self.problem.n_obj ,  1.0)  # potential of each objective set to 1  
                self._igd = self.oIGD.do(self.getarchive().get("F"))                   
        return self.pop
    def _advance(self, infills=None, **kwargs):          
        super()._advance(infills=infills, **kwargs)       
        if self.problem.n_obj  >  3:   
            if  np.char.upper(self.iTypeOfObjectiveMeasure) ==  "IGD" :
                if  (self.iFrequency == 5 and self.n_gen % 5 == 1) or  self.iFrequency ==  1:
                    n_igd = self.oIGD.do(self.getarchive().get("F"))                            
                    if  n_igd - self._igd < 0:
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