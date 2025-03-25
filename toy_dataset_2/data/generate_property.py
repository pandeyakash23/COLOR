import numpy as np

## different amino acids
amino_acid = ['A','B','C','D','E','F','G','H','I','X'] # X is the uncommon amino acid, so total length is 6
aa_prop = [5,2,4,1,8,10,7,6,3,0]
neig = [10,12,14,16,18,20,22,24,26,0]

def output_property(x):
    ## assigning values at which pos
    print('yes')
    N = x.shape[0]
    L = x.shape[1]
    
    in_prop = np.zeros((N,L))
    for i in range(N):
        for j in range(L):
            temp = aa_prop[int(x[i,j])]
            in_prop[i,j] =  temp
        
    prop = np.zeros((N,L))
    for i in range(N):
        for j in range(L): 
            if x[i,j] != 9:
                ne = int(neig[int(x[i,j])])
                if (j >= (ne/2)) and ( j <= (L- (ne/2))  ):
                    idx_temp = int(ne/2)
                    # print(range(j-idx_temp,j+idx_temp))
                    prop[i,j] = np.sum(in_prop[i,j-idx_temp:j+idx_temp])
                elif j < (ne/2):
                    prop[i,j] = np.sum(in_prop[i,0:ne])
                elif j > (L-(ne/2)):
                    prop[i,j] = np.sum(in_prop[i, L-ne:])
                
            # local = [ aa_prop[x[i,k]] for k in range(j-2, j+3)]
            # prop[i, j] = (local[0] + local[1])*((local[3] + local[4]))*local[2]

            
    output_y = np.mean(prop, axis=1)
    return output_y, prop