import numpy as np


aa_prop = [5,2,4,1,8] ## descriptor of categorical variable

def output_property(x):
    ''''
    x: input sequences
    aa_prop: define the descriptor of every 
    categorical variable'''
    
    ## assigning values at which pos
    N = x.shape[0]
    L = x.shape[1]
    prop = np.zeros((N,L))
    for i in range(N):
        j = 1
        while j < (L-1):     
            if 5 not in x[i,j-1:j+2]:
                local = [ aa_prop[x[i,k]] for k in range(j-1, j+2)]
                prop[i, j] = (local[0] + local[1])*local[2]
            j += 1
            
    output_y = np.sum(prop, axis=1)
    return output_y, prop