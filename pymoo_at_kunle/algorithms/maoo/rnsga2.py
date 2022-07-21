import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.dominator import Dominator

from pymoo.algorithms.moo.nsga2 import calc_crowding_distance 
from pymoo.core.evaluator import Evaluator

def getMasksOfObjectives(g):     
    return np.random.choice(g, len(g) - 3, False)      

class RankAndCrowdingSurvival(Survival):

    def __init__(self, iFrequency, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.iFrequency = int(iFrequency)
        self.indx = None
        self.n_genLastUsedForPartialDominance = -1

    def _do(self, problem, pop,  n_survive=None, algorithm = None,   **kwargs):
        # get the objective space values and objects                
        F = pop.get("F").astype(float, copy=False)
        g = np.array(range(F.shape[1])) # sets indices of  objective functions         
       
        n_gen = 1
        iFrequency = int(self.iFrequency)          

        if algorithm is not None:
        	n_gen = int(algorithm.n_gen) 

        if n_gen == 1 and self.indx is None:
        	#this is the first generation/iteration, randomly select indices of 3 objective functions         	
        	self.indx =  getMasksOfObjectives(g)
         	      	
        if  iFrequency ==  1:
        	#randomly select   objectives to mask off at every iteration/generation   
            if  self.n_genLastUsedForPartialDominance !=  n_gen: 
                        self.indx =  getMasksOfObjectives(g)   
                        self.n_genLastUsedForPartialDominance = n_gen      	 
         
                             
        elif iFrequency == 5:
        	#randomly select   objectives to mask off after every 5 iterations/generations
            #self.indx = ()
            if n_gen % 5 == 1:
                 if  self.n_genLastUsedForPartialDominance !=  n_gen: 
                        self.indx =  getMasksOfObjectives(g)   
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

class rNSGA2(NSGA2):
    def __init__(self,
                 pop_size=100,   
                 iFrequency = 1 ,  # 1 =>  change set of objective functions at every iteration/generation
                              # 5 =>  change set of objective functions at every 5th iteraion/generation                  
                 survival=  None,               
                 **kwargs):                
                self.iFrequency = int(iFrequency)
                if survival == None:
                    self.survival = RankAndCrowdingSurvival(iFrequency = iFrequency)
                else:
                    self.survival = survival
                super().__init__(pop_size=pop_size,  \
                    selection=TournamentSelection(func_comp=self.binary_tournament),  \
                    crossover=SimulatedBinaryCrossover(eta=20, prob=0.9),  \
                    mutation=PolynomialMutation(prob=None, eta=20), survival=self.survival,  **kwargs)
                self.n_genLastUsedForPartialDominance = -1
                self.indxOfObjectivesToTurnOff = None 
                self.name = "rnsga2" 

    def getMasksOfObjectives(self, g, F = None, a = None, b = None):                               
                return getMasksOfObjectives(g)   
    def binary_tournament(self,pop, P, algorithm, **kwargs): 
                #print(f"sup:  {self.n_gen}")
                iFrequency =self.iFrequency
                n_tournaments, n_parents = P.shape 
                if n_parents != 2:
                    raise ValueError("Only implemented for binary tournament!")
                tournament_type = algorithm.tournament_type
                S = np.full(n_tournaments, np.nan)                
                F = pop.get("F").astype(float, copy=False)
                g = np.array(range(F.shape[1])) # sets indices of  objective functions                                
                n_gen = int(self.n_gen)                 
                if n_gen == 1 and self.indxOfObjectivesToTurnOff is None:
                    #this is the first generation/iteration, randomly select indices of 3 objective functions           
                    self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)  

                if  iFrequency ==  1:
                    #randomly select   objectives to mask off at every iteration/generation     
                    if  self.n_genLastUsedForPartialDominance !=  n_gen: 
                        self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)  #; print ("1 cool ")
                        self.n_genLastUsedForPartialDominance = n_gen         
                   
                elif iFrequency == 5:                   
                     #randomly select   objectives to mask off after every 5 iterations/generations
                    if n_gen % 5 == 1:
                        if self.n_genLastUsedForPartialDominance !=  n_gen: 
                            self.indxOfObjectivesToTurnOff =  self.getMasksOfObjectives(g)  #; print ("1 cool ")
                            self.n_genLastUsedForPartialDominance = n_gen              
                else:
                    #use all objective functions
                    self.indxOfObjectivesToTurnOff = () # do not mask off any objective            
               
                for i in range(n_tournaments):
                    a, b = P[i, 0], P[i, 1]
                    a_cv, a_f, b_cv, b_f, = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F 
                    rank_a, cd_a  = pop[a].get("rank", "crowding")
                    rank_b, cd_b = pop[b].get("rank", "crowding")

                    # if at least one solution is infeasible
                    if a_cv > 0.0 or b_cv > 0.0:
                        S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)

                    # both solutions are feasible
                    else:

                        if tournament_type == 'comp_by_dom_and_crowding':
                            a_t = a_f.copy() 
                            b_t = b_f.copy()             
                            a_t[self.indxOfObjectivesToTurnOff] = 0.0
                            b_t[self.indxOfObjectivesToTurnOff] = 0.0     
                            rel = Dominator.get_relation(a_t, b_t)
                            if rel == 1:
                                S[i] = a
                            elif rel == -1:
                                S[i] = b                    
                        elif tournament_type == 'comp_by_rank_and_crowding':
                            S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

                        else:
                            raise Exception("Unknown tournament type.")

                        # if rank or domination relation didn't make a decision compare by crowding
                        if np.isnan(S[i]):
                            S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

                return S[:, None].astype(int, copy=False)