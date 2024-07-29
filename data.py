#from sage.all import *
import numpy as np
from itertools import chain



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
def convertPermutation(order,permutation):
    """
        Input e.g. permutation = np.array([2,1,0,4,3])
        converts a permutationrepresentation to the one that is accepted by sageMath 
        e.g. [2 3 1], which sends 1 --> 2, 2 --> 3 , 3 --> 1 to the representation (1 2 3)

    """
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
            index = new_start
            cycle = [index]
        else:
            cycle.append(permutation[index])
            index = permutation[index]
    for i in range(len(generator)):
        generator[i] = tuple(generator[i]) 
    return generator

def simpletest(perms):
	return (map (lambda pi: PermutationGroup(pi).is_simple(), perms)) 

list(simpletest(
	list((map
		(lambda ro: convertPermutation(10,ro),
	 	generateRandomPermutations(50,10))))))
