import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import matplotlib as mpl
import os
import gc
import pandas as pd
import csv
from numpy import *
from torch.utils.tensorboard import SummaryWriter
from datetime import date
import time
import builtins
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error,r2_score, mean_squared_error
import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler ## this module takes in our data and distributes on different GPUs
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group 

   
# writer = SummaryWriter()
# writer = SummaryWriter(f"Training starting on:{date.today()}")
# writer = SummaryWriter(comment="transformer model")
amino_acid = ['A','B','C','D','E','X'] # X is the uncommon amino acid, so total length is 6

## Dataloader
batch_size = 256
class spiderdataset(Dataset) :
    def __init__(self,ohe, classes,seq_len,output, n_samples) :
        # data loading
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(int64))
        self.classes = torch.from_numpy(classes.astype(int64)) 
        self.output = torch.from_numpy(output.astype(np.float32)).reshape((-1,1))
        self.n_samples = n_samples
        
        
    def __getitem__(self,index) :
        return self.ohe[index], self.classes[index], self.seq_len[index], self.output[index]

    def __len__(self):    
        return self.n_samples      

def make_dataset(): 
        
    ohe_valid = np.load('../x_test.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load('../y_test.npy', allow_pickle=True)
    seq_len_valid = np.load('../len_test.npy', allow_pickle=True) 
  
 
    test_dataset = spiderdataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 3000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term_odd = torch.exp(torch.arange(0, d_model-1, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        self.register_buffer('pe', pe)

    def forward(self, x, rank):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        
        x = x + self.pe[:x.size(0)].to(rank)
        return self.dropout(x)
    
class network(nn.Module):
    def __init__(self, num_classes, max_m, rank):
        super(network, self).__init__()
        self.max_m = max_m
        self.rank = rank
        self.d_model = 6
        self.d_out = 32
        # ## before cnn  
        # cnn layers        
        # self.embedding = nn.Embedding(8, self.d_model, padding_idx=0)  
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.cnn1 = nn.Sequential( nn.Conv1d(self.d_model,32,2, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(32,64,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64,int(self.d_out),1, stride=1)
                                   ) ##4 size motifs
        self.ln1 = nn.LayerNorm(self.d_model)
               
        self.cnn2 = nn.Sequential( nn.Conv1d(self.d_model,32,2, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(32,64,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64,32,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(32,int(self.d_out),1, stride=1)
                                   ) ##8 size motifs
        self.ln2 = nn.LayerNorm(self.d_model)
        
        self.cnn3 = nn.Sequential( nn.Conv1d(self.d_model,32,3, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(32,64,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64,32,1, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(32,int(self.d_out),1, stride=1)
                                   ) ## 6 motifs
        self.ln3 = nn.LayerNorm(self.d_model)
        
        self.cnn4 = nn.Sequential( nn.Conv1d(self.d_model,32,3, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(32,64,1, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64,int(self.d_out),1, stride=1),
                                   ) ## 3 size motif
        self.ln4 = nn.LayerNorm(self.d_model)
        
        self.maxpool1 = nn.Sequential( nn.AvgPool1d(2, stride=1),
                                       nn.AvgPool1d(2, stride=1),
                                   )
        
        self.maxpool2 = nn.Sequential( nn.AvgPool1d(2, stride=1),
                                       nn.AvgPool1d(2, stride=1),
                                       nn.AvgPool1d(2, stride=1),
                                   )
        self.maxpool3 = nn.Sequential( nn.AvgPool1d(3, stride=1),
                                       nn.AvgPool1d(2, stride=1),
                                       nn.AvgPool1d(1, stride=1),
                                       nn.AvgPool1d(1, stride=1),
                                   )
        self.maxpool4 = nn.Sequential( nn.AvgPool1d(3, stride=1),
                                       nn.AvgPool1d(1, stride=1),
                                       nn.AvgPool1d(1, stride=1),
                                   )
              
        self.nn = nn.Sequential(
                                nn.Linear(int(self.d_model*self.d_out*self.max_m),128),
                                nn.ReLU(),
                                nn.Linear(128,32),
                                nn.ReLU(),
                                nn.Linear(32,8),
                                nn.ReLU(),
                                nn.Linear(8,num_classes)
                                )
        
    def make_possible_motifs(self,classes, s_len,fs):
        self.fs = fs
        motifs = []
        sweep = 0
        while (sweep+fs) <= s_len:
            mo = classes[0,sweep:sweep+fs]
            string_mo = ''            
            for i in range(mo.size(0)):
                string_mo = string_mo + amino_acid[mo[i]]
            motifs.append(string_mo)
            sweep += 1
        self.motifs = motifs
            
        
    def find_motifs(self,op, Lrep):
        s_len = len(self.overall_imp_segments)
        local_visit = torch.zeros((len(self.overall_imp_segments),))
        mo_effect = op*Lrep
        mo_effect = mo_effect[:,0:len(self.motifs)]
        mo_effect = (mo_effect-torch.mean(mo_effect))/torch.std(mo_effect)
        mo_effect = torch.abs(mo_effect)
        if torch.sum(mo_effect) == 0:
            return False
        else:
            descending_idx = torch.argsort(mo_effect, descending=True)[0]
            descending_idx = descending_idx[0:10]
            total_motif = len(descending_idx)
            for i, d_idx in enumerate(descending_idx):
                d_idx = int(d_idx)
                for k in range(d_idx, d_idx+self.fs):
                    if k < s_len:
                        new_value = self.importance*(total_motif-i)*\
                            (1- (torch.sum(local_visit)/s_len))                            
                        self.overall_imp_segments[k,] += new_value
                        self.trace_visitation[k,] += 1
                        if local_visit[k,] ==0:
                            local_visit[k,] += 1
                            
                    # store_all_motif.append(self.motifs[d_idx])
                    # store_all_imp.append(self.importance*(total_motif-i))
                    # store_all_pos.append(np.arange(d_idx, d_idx+self.fs).tolist())
            return True
        
    def forward(self, x,  classes, seq_len, f, overall_imp_segments, importance, trace_visitation):
        'x: [batch, seq_len, feature], classes: [N,L]'
        self.trace_visitation = trace_visitation
        self.importance =  importance
        self.overall_imp_segments = overall_imp_segments
        out = torch.permute(x,(0,2,1)) ## making it (N, f, L)
        out = self.positional_encoding(out.permute(2,0,1),self.rank) ##[L,N,f]
        out = out.permute(1,2,0)
        
        out_1 = self.cnn1(out) ##[N,f,L]
        out_2 = self.cnn2(out) ##[N,f,L]
        out_3 = self.cnn3(out) ##[N,f,L]
        out_4 = self.cnn4(out) ##[N,f,L]
        pool_1 = self.maxpool1(x.permute(0,2,1))*4 ## [N,f,L]
        pool_2 = self.maxpool2(x.permute(0,2,1))*8 ## [N,f,L]
        pool_3 = self.maxpool3(x.permute(0,2,1))*6 ## [N,f,L]
        pool_4 = self.maxpool4(x.permute(0,2,1))*1 ## [N,f,L]
        
        # print(f[0])
        if f[2] == 0:
            self.make_possible_motifs(classes, seq_len,4)
            _ = self.find_motifs(pool_1[:,f[0],:], out_1.permute(0,2,1)[:,:,f[1]])
        
        if f[2] == 1:
            self.make_possible_motifs(classes, seq_len,8)
            _ = self.find_motifs(pool_2[:,f[0],:], out_2.permute(0,2,1)[:,:,f[1]])
        
        if f[2] == 2:
            self.make_possible_motifs(classes, seq_len,6)
            _ = self.find_motifs(pool_3[:,f[0],:], out_3.permute(0,2,1)[:,:,f[1]])
        
        if f[2] == 3:
            self.make_possible_motifs(classes, seq_len,1)
            _ = self.find_motifs(pool_4[:,f[0],:], out_4.permute(0,2,1)[:,:,f[1]])
        
        return self.overall_imp_segments, self.trace_visitation
    
def initalize(rank, init_lr):
    
    model = torch.load('./model/best.pth', map_location='cuda:1')
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion, optimizer

## Training loop
def motif_identification(num_epochs, init_lr, max_m):     
    which_data = input('Enter the index of interest:')
    which_data = int(which_data)
    rank = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    test_loader, _  = make_dataset()
    model, _, _ = initalize(rank, init_lr)
    
    f_importance = np.load('f_importance.npy', allow_pickle=True)
    f_name = np.load('f_name.npy', allow_pickle=True)
    idx = np.argsort(f_importance) ## sorted in ascending order
    idx = idx[::-1] ## list changed to descending order
    f_importance = f_importance[idx]
    print('To check the correct sorting:', f_importance[0:5])
    f_name = f_name[idx]
    f_pointer = 0 
    while f_pointer < 10: 
        feature = f_name[f_pointer].split('_')
        l,b,h = int(feature[0]), int(feature[1]), int(feature[2])                 
        with torch.no_grad():     
            for _, (i_x,i_classes, i_seq, i_actual) in enumerate(test_loader):
                i_x = i_x.to(rank) #.type(dtype=torch.float32)
                i_seq = i_seq.to(rank).type(dtype=torch.float32)
                i_classes = i_classes.to(rank)
                i_actual = i_actual.to(rank).reshape((1,-1))
                prop_sort = np.argsort(i_actual.cpu().numpy())[0] # sorted in ascending order
                ## choosing the best sequence for motif identification
                prop_idx = prop_sort[which_data] #torch.argmin(i_actual).item()
                i_x = i_x[prop_idx].reshape((1,i_x.size(1), i_x.size(2)))
                i_classes = i_classes[prop_idx].reshape((1, i_classes.size((1))))
                i_seq = i_seq[prop_idx]
                
                if f_pointer == 0:
                    print('Property value is:', i_actual[0,prop_idx])
                    overall_imp_segments = torch.zeros((int(i_seq.item()),)).to(rank)
                    trace_visitation = torch.zeros((int(i_seq.item()),)).to(rank)
                    
                overall_imp_segments, trace_visitation = \
                model(i_x, i_classes, i_seq, [l,b,h], overall_imp_segments, f_importance[f_pointer], trace_visitation)    
                    
        f_pointer += 1
    
    sequence_of_int = ''
    for i in range(int(i_seq.item())):
        sequence_of_int = sequence_of_int + amino_acid[i_classes[0,i]]
    
    seq_out = np.load('../test_sequence_output.npy', allow_pickle=True)
    
    np.save('./motifs_results/sequence', sequence_of_int)
    np.save('./motifs_results/weight', overall_imp_segments.to('cpu'))
    np.save('./motifs_results/seq_output', seq_out[prop_idx])
    # np.save('./motifs_results/seq_out', seq_output[prop_idx])
        
if __name__=='__main__':
    # store_all_motif = []
    # store_all_imp = []
    # store_all_pos = []
    # seq_output = np.load('../seq_out_test.npy', allow_pickle=True)
    cp_1 = time.time()
    num_epochs = 1
    init_lr = 0.0003
    max_m = int(1)
    ##change
    motif_identification(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
    # np.save('./motifs_results/store_all_motif', store_all_motif)
    # np.save('./motifs_results/store_all_imp', store_all_imp)
    # np.save('./motifs_results/store_all_pos', store_all_pos)
    
