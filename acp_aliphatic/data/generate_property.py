import numpy as np

## different amino acids
amino_acid = ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G','X']# X is the uncommon amino acid, so total length is 6
a_id = amino_acid.index('A')
v_id = amino_acid.index('V')
i_id = amino_acid.index('I')
l_id = amino_acid.index('L')

def output_property(x, seq_len):
    N = x.shape[0]
    prop = np.zeros((N,))
    x = np.argmax(x, axis=-1)
    for i in range(N):
        l = int(seq_len[i])
        x_sample = x[i,0:l]
        # print(np.sum((x_sample==a_id)*1))
        num_a = np.sum((x_sample==a_id))/l
        num_v = np.sum((x_sample==v_id))/l
        num_i = np.sum((x_sample==i_id))/l
        num_l = np.sum((x_sample==l_id))/l
        prop[i] = (num_a + 2.9*num_v + 3.9*(num_i+num_l))*100

    return prop