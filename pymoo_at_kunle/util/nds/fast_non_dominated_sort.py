import numpy as np 
from pymoo.util.dominator import Dominator 
def paretoDominanceComparePopulations(offspring, parent, eps = 0.0): 
    res = np.full((len(offspring), 1), False)
    for i in range(offspring.shape[0]):
        res[i, 0] =   not dom(parent[i],offspring[i])  
           # offspring is better if it's not dominated by parent
    return res    
def dom(a, b, eps = 0.0):
        return np.all(a <= (1 + eps) * b) and np.any(a <  (1 + eps) * b) 
def get_relation(a, b, eps = 0.0):
        a_dom_b = dom(a , b, eps)
        b_dom_a = dom(b , a, eps)
        if a_dom_b and not b_dom_a:
            return 1
        elif b_dom_a and not a_dom_b:
            return -1
        else:
            return 0  
def calc_domination_matrix_Hack(F, eps = 0.0):
        n = F.shape[0]
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = get_relation(F[i, :], F[j, :], eps)
                M[j, i] = -M[i, j] 
        return M 
def fast_non_dominated_sort(F, **kwargs):
  
    epsilon = kwargs["epsilon"]     
    #M = Dominator.calc_domination_matrix(F,  epsilon = epsilon)
    M = calc_domination_matrix_Hack(F, eps = epsilon)  #
    #M = Dominator.calc_domination_matrix(F, epsilon =epsilon)
    n = M.shape[0]
   # print(f"here nau: M = {M}") 
    fronts = [] 
    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=int)

    # for each individual a list of all individuals that are dominated by this one
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts

