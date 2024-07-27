from sage.all import *
import numpy as np
print("hello")

G = PermutationGroup(['(1,2,3)(4,5)', '(3,4)'])

n=6
s=5
permutations = np.zeros((s,n))
for i in range(0,s):
    #permutation = np.array([0 for i in range(n)])
    indexlist = [i for i in range(n)]
    len_list = n
    for j in range(0, n):
       k = np.random.randint(0,len_list)
       permutations[i,j]=indexlist[k]
       len_list = len_list - 1
       indexlist.remove(indexlist[k])

ppermutations = np.unique(permutations, axis=0)
    

