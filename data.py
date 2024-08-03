#from sage.all import *
import numpy as np
from itertools import chain,combinations
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader

#python generat permutations
def generateRandomPermutations(numberOf, order):
    """
        creates numberOf permutations with given order
    """
    permutations = np.zeros((numberOf,order),dtype=int)
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
def flattenPermMatrix(pMatrix):
    return pMatrix.flatten()
def simpletest(cperms):
	return list((map (lambda pi: PermutationGroup(pi,domain = list(range(len(pi[0])))).is_simple(), cperms)))
def getDataLoader(dataframe):
    X_raw = np.array(dataframe['X'].values)
    X_raw = [elem[1:-1] for elem in X_raw]
    X_raw = np.stack([np.array(ast.literal_eval(f"[{elem}]")) for elem in X_raw])
    X_Matrix = np.apply_along_axis(createPermutationMatrix,axis=2,arr=X_raw)
    #X_flat = np.apply_along_axis(flattenPermMatrix,axis=3,arr=X_Matrix)
    orig_shape = X_Matrix.shape
    new_shape = (orig_shape[0],orig_shape[1]*orig_shape[3]*orig_shape[2])
    X_polished = X_Matrix.reshape(new_shape)
    
    y = np.array(dataframe['y'].values)

    X = torch.tensor(X_polished,dtype=torch.int64)
    y = torch.tensor(y,dtype=torch.int64)
    print(X)
    print(y)
    loader = DataLoader(list(zip(X,y)),shuffle=True,batch_size=16)
    return loader 



# preperms = list((map
#               	    (lambda ro: convertPermutation(ro),
#               	    generateRandomPermutations(6,4))
#                 ))
# single_generators = generateRandomPermutations(6,4)
# combined_generators = list(combinations(single_generators,2))
# combined_generators = [[list(elem[0]),list(elem[1])] for elem in combined_generators]

# combined_generators_df = pd.DataFrame({'X':combined_generators})
# mathpermssimple =  simpletest(combined_generators)

# combined_generators_df['y'] = mathpermssimple
# combined_generators_df.to_csv('data_1.csv')
combined_generators_df = pd.read_csv('data_1.csv')
getDataLoader(combined_generators_df)


