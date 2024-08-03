from sage.all import *
import numpy as np
from itertools import chain,combinations
import pandas as pd

#python generat permutations
def generateRandomPermutations(numberOf, order):
    """
        creates numberOf permutations with given order
    """
    permutations = np.zeros((numberOf,order))
    for i in range(0,numberOf):
        indexlist = [i for i in range(order)]
        len_list = order
        for j in range(0, order):
            k = np.random.randint(0,len_list)
            permutations[i,j]=indexlist[k]
            len_list = len_list - 1
            indexlist.remove(indexlist[k])
    ppermutations = np.unique(permutations, axis=0)
    return ppermutations

#  list(PermutationGroup([['b','c','a']], domain=['a','b','c']))
# [(), ('a','b','c'), ('a','c','b')]
def convertPermutation(permutation):
    """
        Input e.g. permutation = np.array([2,1,0,4,3])
        converts a permutationrepresentation to the one that is accepted by sageMath 
        e.g. [2 3 1], which sends 1 --> 2, 2 --> 3 , 3 --> 1 to the representation (1 2 3)

    """
    order = len(permutation)
    generator = []
    index = 0 
    cycle = [index]
    first = 0 # remembers the beginning of a cycle
    for i in range(order):
        if permutation[index] == first:
            generator.append(cycle)
            new_start = np.random.randint(order)
            rest_list = list(chain.from_iterable(generator))
            while new_start in rest_list and len(rest_list)!=order :
                new_start = np.random.randint(order)
            first = new_start
            index = int(new_start)
            cycle = [index]
        else:
            cycle.append(permutation[index])
            index = int(permutation[index])
    for i in range(len(generator)):
        generator[i] = tuple(generator[i]) 
    return generator

def createPermutationMatrix(permutation):
    order = len(permutation)
    p_matrix =  np.zeros((order,order))
    for i in range(order):
        p_matrix[i,permutation[i]]=1
    return p_matrix
#g1 = PermutationGroup([[3,1,2,0],[0,2,1,3]], domain=[0,1,2,3])
#g1 = PermutationGroup([array([0., 2., 3., 1.]), array([0., 3., 2., 1.])], domain=[0,1,2,3])

def simpletest(cperms):
	return list((map (lambda pi: PermutationGroup(pi,domain = list(range(len(pi[0])))).is_simple(), cperms)))



# preperms = list((map
#               	    (lambda ro: convertPermutation(ro),
#               	    generateRandomPermutations(6,4))
#                 ))
single_generators = generateRandomPermutations(6,4)
combined_generators = list(combinations(single_generators,2))
combined_generators = [[list(elem[0]),list(elem[1])] for elem in combined_generators]

combined_generators_df = pd.DataFrame({'X':combined_generators})
print(len(combined_generators))
print(combined_generators)
mathpermssimple =  simpletest(combined_generators)

combined_generators_df['y'] = mathpermssimple
combined_generators_df.to_csv('data_1.csv')

