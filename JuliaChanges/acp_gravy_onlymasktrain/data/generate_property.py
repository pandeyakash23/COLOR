import numpy as np

## different amino acids
amino_acid = ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G','X']# X is the uncommon amino acid, so total length is 6
hydro = [1.8, 4.2, 2.8, 4.5, 3.8, -3.5, -3.5, -3.9, -0.8, -0.7, -1.3, 2.5, -3.5, -3.5, -1.6, 1.9, -4.5, -3.2, -0.9, -0.4,0]

def output_property(x, seq_len):
    N = x.shape[0]
    prop = np.zeros((N,))
    x = np.argmax(x, axis=-1)
    for i in range(N):
        l = int(seq_len[i])
        x_sample = x[i,0:l]
        sample_prop = 0
        for j in  range(l):
            sample_prop += hydro[x_sample[j]]
        
        prop[i] = sample_prop/l

    return prop