import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


amino_acid = np.load('./model/categorical_variables.npy', allow_pickle=True)
amino_acid = amino_acid.tolist()

   
class dataset(Dataset) :
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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model%2 != 0:
            div_term = torch.exp(torch.arange(0, d_model+1, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model+1)
        else:
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            
        position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        if d_model%2!=0:
            pe = pe[:,:,0:-1]
        self.register_buffer('pe', pe)

    def forward(self, x, rank):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print(x.size(), self.pe[:x.size(0)].size())
        x = x + self.pe[:x.size(0)].to(rank)
        return self.dropout(x)
    
class complor_network(nn.Module):
    def __init__(self, num_classes, d_model, d_out, max_m, rank):
        super(complor_network, self).__init__()
        self.max_m = max_m
        self.rank = rank
        self.d_model = d_model ## equivalent to q
        self.d_out = d_out ## equivalent to d

        # cnn layers   
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.cnn1 = nn.Sequential( nn.Conv1d(self.d_model,32,3, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(32,64,1, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64,int(self.d_out),1, stride=1)
                                   ) ##4 size motifs
        self.ln1 = nn.LayerNorm(self.d_out)
               
        # self.cnn2 = nn.Sequential( nn.Conv1d(self.d_model,32,2, stride=1), 
        #                            nn.ReLU(),
        #                            nn.Conv1d(32,64,2, stride=1),
        #                            nn.ReLU(),
        #                            nn.Conv1d(64,32,2, stride=1),
        #                            nn.ReLU(),
        #                            nn.Conv1d(32,int(self.d_out),1, stride=1)
        #                            ) ##8 size motifs
        # self.ln2 = nn.LayerNorm(self.d_out)
        
        # self.cnn3 = nn.Sequential( nn.Conv1d(self.d_model,32,3, stride=1), 
        #                            nn.ReLU(),
        #                            nn.Conv1d(32,64,2, stride=1),
        #                            nn.ReLU(),
        #                            nn.Conv1d(64,32,1, stride=1),
        #                            nn.ReLU(),
        #                            nn.Conv1d(32,int(self.d_out),1, stride=1)
        #                            ) ## 6 motifs
        # self.ln3 = nn.LayerNorm(self.d_out)
        
        # self.cnn4 = nn.Sequential( nn.Conv1d(self.d_model,32,1, stride=1), 
        #                            nn.ReLU(),
        #                            nn.Conv1d(32,64,1, stride=1),
        #                            nn.ReLU(),
        #                            nn.Conv1d(64,int(self.d_out),1, stride=1),
        #                            ) ## 1 size motif
        # self.ln4 = nn.LayerNorm(self.d_out)
        
        self.maxpool1 = nn.Sequential( nn.AvgPool1d(3, stride=1),
                                       nn.AvgPool1d(1, stride=1),
                                   )
        
        self.motif_size_1 = 3*1
        # self.maxpool2 = nn.Sequential( nn.AvgPool1d(2, stride=1),
        #                                nn.AvgPool1d(2, stride=1),
        #                                nn.AvgPool1d(2, stride=1),
        #                            )
        # self.motif_size_2 = 2*2*2
        
        # self.maxpool3 = nn.Sequential( nn.AvgPool1d(3, stride=1),
        #                                nn.AvgPool1d(2, stride=1),
        #                                nn.AvgPool1d(1, stride=1),
        #                             #    nn.AvgPool1d(1, stride=1),
        #                            )
        # self.motif_size_3 = 3*2
        
        # self.maxpool4 = nn.Sequential( nn.AvgPool1d(1, stride=1),
        #                                nn.AvgPool1d(1, stride=1),
        #                                nn.AvgPool1d(1, stride=1),
        #                            )
        # self.motif_size_4 = 1
              
        self.nn = nn.Sequential(
                                nn.Linear(int(self.d_model*self.d_out*self.max_m),32),
                                nn.ReLU(),
                                nn.Linear(32,8),
                                nn.ReLU(),
                                nn.Linear(8,num_classes)
                                )
        

        
    def forward(self, x,  classes, seq_len):
        'x: [batch, seq_len, feature], classes: [N,L]'

        out = torch.permute(x,(0,2,1)) ## making it (N, f, L)
        out = self.positional_encoding(out.permute(2,0,1),self.rank) ##[L,N,f]
        
        out = out.permute(1,2,0)
        
        out_1 = self.cnn1(out) ##[N,f,L]
        out_1 = torch.permute(self.ln1(torch.permute(out_1,(0,2,1))),(0,2,1))
        # out_2 = self.cnn2(out) ##[N,f,L]
        # out_2 = torch.permute(self.ln2(torch.permute(out_2,(0,2,1))),(0,2,1))
        # out_3 = self.cnn3(out)
        # out_3 = torch.permute(self.ln3(torch.permute(out_3,(0,2,1))),(0,2,1))
        # out_4 = self.cnn4(out)
        # out_4 = torch.permute(self.ln4(torch.permute(out_4,(0,2,1))),(0,2,1))


        pool_1 = self.maxpool1(x.permute(0,2,1))*self.motif_size_1 ## [N,f,L]
        for i in range(x.size(0)):
            pool_1[i,:,:] = pool_1[i,:,:]/seq_len[i]
        # pool_2 = self.maxpool2(x.permute(0,2,1))*self.motif_size_2 ## [N,f,L]
        # pool_3 = self.maxpool3(x.permute(0,2,1))*self.motif_size_3
        # pool_4 = self.maxpool4(x.permute(0,2,1))*self.motif_size_4
   
        
        out_1 = torch.matmul(pool_1, out_1.permute(0,2,1))
        out_1 = out_1.reshape((out_1.size(0), out_1.size(1), out_1.size(2),1))
        
        # out_2 = torch.matmul(pool_2, out_2.permute(0,2,1))
        # out_2 = out_2.reshape((out_2.size(0), out_2.size(1), out_2.size(2),1))
        
        # # out_3 = self.cnn3(out) ##[N,f,L]
        # out_3 = torch.matmul(pool_3, out_3.permute(0,2,1))
        # out_3 = out_3.reshape((out_3.size(0), out_3.size(1), out_3.size(2),1))
        
        # # out_4 = self.cnn4(out) ##[N,f,L]
        # out_4 = torch.matmul(pool_4, out_4.permute(0,2,1))
        # out_4 = out_4.reshape((out_4.size(0), out_4.size(1), out_4.size(2),1))
        
        # heat_map = torch.cat((out_1, out_2, out_3,out_4), dim=3)      
        heat_map = out_1
        # heat_map = nn.Softmax(dim=1)(heat_map)
        
        heat_map = heat_map.permute(0,2,3,1)
        # heat_map = self.ln4(heat_map)
        
        
        heat_map = nn.Flatten()(heat_map[:,:,:,:]) ## removing amino acid contribution from heat map as there are none        
        heat_map = self.nn(heat_map)
        
        return heat_map
    
    def forward_feature_importance(self, x,  classes, seq_len, f, permute):
        'x: [batch, seq_len, feature], classes: [N,L]'

        out = torch.permute(x,(0,2,1)) ## making it (N, f, L)
        out = self.positional_encoding(out.permute(2,0,1),self.rank) ##[L,N,f]
        
        out = out.permute(1,2,0)
        
        out_1 = self.cnn1(out) ##[N,f,L]
        out_1 = torch.permute(self.ln1(torch.permute(out_1,(0,2,1))),(0,2,1))
                   
        # out_2 = self.cnn2(out) ##[N,f,L]
        # out_2 = torch.permute(self.ln2(torch.permute(out_2,(0,2,1))),(0,2,1))
        # out_3 = self.cnn3(out)
        # out_3 = torch.permute(self.ln3(torch.permute(out_3,(0,2,1))),(0,2,1))
        # out_4 = self.cnn4(out)
        # out_4 = torch.permute(self.ln4(torch.permute(out_4,(0,2,1))),(0,2,1))
    
        pool_1 = self.maxpool1(x.permute(0,2,1))*self.motif_size_1 ## [N,f,L]
        for i in range(x.size(0)):
            pool_1[i,:,:] = pool_1[i,:,:]/seq_len[i]
        # pool_2 = self.maxpool2(x.permute(0,2,1))*self.motif_size_2 ## [N,f,L]
        # pool_3 = self.maxpool3(x.permute(0,2,1))*self.motif_size_3
        # pool_4 = self.maxpool4(x.permute(0,2,1))*self.motif_size_4
   
        
        out_1 = torch.matmul(pool_1, out_1.permute(0,2,1))
        out_1 = out_1.reshape((out_1.size(0), out_1.size(1), out_1.size(2),1))
        
        # out_2 = torch.matmul(pool_2, out_2.permute(0,2,1))
        # out_2 = out_2.reshape((out_2.size(0), out_2.size(1), out_2.size(2),1))
        
        # # out_3 = self.cnn3(out) ##[N,f,L]
        # out_3 = torch.matmul(pool_3, out_3.permute(0,2,1))
        # out_3 = out_3.reshape((out_3.size(0), out_3.size(1), out_3.size(2),1))
        
        # # out_4 = self.cnn4(out) ##[N,f,L]
        # out_4 = torch.matmul(pool_4, out_4.permute(0,2,1))
        # out_4 = out_4.reshape((out_4.size(0), out_4.size(1), out_4.size(2),1))
        
        # heat_map = torch.cat((out_1, out_2, out_3,out_4), dim=3)      
        heat_map = out_1
        # heat_map = nn.Softmax(dim=1)(heat_map)
        
        heat_map = heat_map.permute(0,2,3,1)
        # heat_map = self.ln4(heat_map)
        ## permuting the feature
        if permute:
            indices = torch.randperm(heat_map.shape[0])
            heat_map[:, f[1],f[2], f[0]] = heat_map[indices, f[1],f[2], f[0]]
        
        heat_map = nn.Flatten()(heat_map) ## removing amino acid contribution from heat map as there are none       
         
        heat_map = self.nn(heat_map)
        
        return heat_map
        
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
        # mo_effect = (mo_effect-torch.mean(mo_effect))/(torch.std(mo_effect)+1E-18)
        mo_effect = torch.abs(mo_effect)
        if torch.sum(mo_effect) == 0:
            return False
        else:
            descending_idx = torch.argsort(mo_effect, descending=True)[0]
            # descending_idx = descending_idx[0:10]
            ''' # 10 here means that we are using assigning importance to 
            only top 10 motifs related with respect to every element in R'''
            mo_effect = (mo_effect - torch.min(mo_effect)) / ((torch.max(mo_effect)-torch.min(mo_effect))+1E-18)            
            total_motif = len(descending_idx)
            for i, d_idx in enumerate(descending_idx):
                d_idx = int(d_idx)
                for k in range(d_idx, d_idx+self.fs):
                    if k < s_len:
                        new_value = self.importance*(total_motif-i)*\
                            (1- (torch.sum(local_visit)/s_len))*mo_effect[0,d_idx]                         
                        self.overall_imp_segments[k,] += new_value
                        self.trace_visitation[k,] += 1
                        if local_visit[k,] ==0:
                            local_visit[k,] += 1

            return True
        
    def importance_calculation(self, x,  classes, seq_len, f, overall_imp_segments, importance, trace_visitation):
        'x: [batch, seq_len, feature], classes: [N,L]'
        self.trace_visitation = trace_visitation
        self.importance =  importance
        self.overall_imp_segments = overall_imp_segments
        out = torch.permute(x,(0,2,1)) ## making it (N, f, L)
        out = self.positional_encoding(out.permute(2,0,1),self.rank) ##[L,N,f]
        out = out.permute(1,2,0)
        
        out_1 = self.cnn1(out) ##[N,f,L]
        out_1 = torch.permute(self.ln1(torch.permute(out_1,(0,2,1))),(0,2,1))
        # out_2 = self.cnn2(out) ##[N,f,L]
        # out_2 = torch.permute(self.ln2(torch.permute(out_2,(0,2,1))),(0,2,1))
        # out_3 = self.cnn3(out)
        # out_3 = torch.permute(self.ln3(torch.permute(out_3,(0,2,1))),(0,2,1))
        # out_4 = self.cnn4(out)
        # out_4 = torch.permute(self.ln4(torch.permute(out_4,(0,2,1))),(0,2,1))
        
        pool_1 = self.maxpool1(x.permute(0,2,1))*self.motif_size_1 ## [N,f,L]
        pool_1 = self.maxpool1(x.permute(0,2,1))*self.motif_size_1 ## [N,f,L]
        for i in range(x.size(0)):
            pool_1[i,:,:] = pool_1[i,:,:]/seq_len[i]
        # pool_2 = self.maxpool2(x.permute(0,2,1))*self.motif_size_2 ## [N,f,L]
        # pool_3 = self.maxpool3(x.permute(0,2,1))*self.motif_size_3 ## [N,f,L]
        # pool_4 = self.maxpool4(x.permute(0,2,1))*self.motif_size_4 ## [N,f,L]
        
        # print(f[0])
        if f[2] == 0:
            self.make_possible_motifs(classes, seq_len,self.motif_size_1)
            _ = self.find_motifs(pool_1[:,f[0],:], out_1.permute(0,2,1)[:,:,f[1]])
        
        # if f[2] == 1:
        #     self.make_possible_motifs(classes, seq_len,self.motif_size_2)
        #     _ = self.find_motifs(pool_2[:,f[0],:], out_2.permute(0,2,1)[:,:,f[1]])
        
        # if f[2] == 2:
        #     self.make_possible_motifs(classes, seq_len,self.motif_size_3)
        #     _ = self.find_motifs(pool_3[:,f[0],:], out_3.permute(0,2,1)[:,:,f[1]])
        
        # if f[2] == 3:
        #     self.make_possible_motifs(classes, seq_len,self.motif_size_4)
        #     _ = self.find_motifs(pool_4[:,f[0],:], out_4.permute(0,2,1)[:,:,f[1]])
        
        return self.overall_imp_segments, self.trace_visitation