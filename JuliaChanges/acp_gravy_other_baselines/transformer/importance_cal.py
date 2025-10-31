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
from datetime import date
import time
import builtins
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error,r2_score
from transformer_classes import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, EncoderLayer

# import sys
# sys.path.insert(0, '../')
# from generate_property import output_property
# cam = []

which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')
which_method = input('attn or grad:')

## Dataloader
batch_size = 64

class spiderdataset(Dataset) :
    def __init__(self,ohe, seq_len,output, n_samples) :
        # data loading
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(np.int64))
        self.output = torch.from_numpy(output.astype(np.float32)).reshape((-1,1))
        self.n_samples = n_samples

    def __getitem__(self,index) :
        return self.ohe[index], self.seq_len[index], self.output[index]

    def __len__(self):    
        return self.n_samples      

def make_classes_sequence(ohe, seq_len):
    class_seq = np.argmax(ohe, axis=-1)
    for i in range(len(class_seq)):
        l = int(seq_len[i])
        class_seq[i,0:l] += 1

    return class_seq


def make_dataset(): 
    path = '../data'       
    ohe_valid = np.load(path+f'/x_{which_data}.npy', allow_pickle=True)
    output_valid = np.load(path+f'/y_{which_data}.npy', allow_pickle=True)
    seq_len_valid = np.array([ohe_valid.shape[1]]*len(ohe_valid))
    test_dataset = spiderdataset(ohe_valid,seq_len_valid,output_valid,ohe_valid.shape[0])
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0], ohe_valid.shape[1]

def extract(grad):
    cam.append(grad)
        
class network(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,num_classes, rank):
        super(network, self).__init__()
        ##add embeddings here
        self.rank = rank
        self.d_model = d_model 
        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.proj = nn.Linear( src_vocab_size,d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc =  nn.Sequential(
                                nn.Linear(32,16),
                                nn.ReLU(),
                                nn.Linear(16,8),
                                nn.ReLU(),
                                nn.Linear(8,num_classes)
                                )
        self.dropout = nn.Dropout(dropout)
        ### cnn layers            
        self.cnn1 = nn.Sequential( nn.Conv1d(self.d_model,64,2, stride=1), 
                                   nn.ReLU(),
                                   nn.Conv1d(64,64,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64,32,2, stride=1),
                                   nn.ReLU(),
                                   nn.Conv1d(32,32,2, stride=1),
                                   )

    def forward(self, i_x, src_mask, seq_len, need_grad): 
        # ix: n,l,d
        # i_x = i_x.permute(1,0,2)
        i_x = self.proj(i_x) # n,l,d_model
        i_x = i_x.permute(1,0,2) # l,n,d_model
        # src = self.dropout(self.positional_encoding(i_x, self.rank))
        enc_output = i_x
        for _, enc_layer in enumerate(self.encoder_layers):
            enc_output, attn_prob, attn_grad = enc_layer(enc_output, src_mask, need_grad)
            '''attn_prob & attm: [N, heads,L,L]'''
            attn_prob = torch.mean(attn_prob, dim=1)
            
        enc_output = enc_output.permute(1,2,0)
        enc_output = self.cnn1(enc_output)
        # print(enc_output.size())
        
        # out_mean = enc_output[...,-1]
        out_mean = torch.mean(enc_output, dim=-1)
        output = self.fc(out_mean)  
        return output, attn_prob, attn_grad
    
def initalize(rank, max_m, init_lr):
    
    model = torch.load('./model/best.pth', map_location='cuda:6')
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion, optimizer

## Training loop
def train(num_epochs, init_lr, max_m):
    ''' top_per controls the top % tokens '''
    need_grad = True
    rank = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    test_loader, valid_size, max_len  = make_dataset()
    model, criterion, optimizer = initalize(rank, max_m, init_lr)
    store_all_importance = torch.zeros((valid_size, max_len)).to(rank)
    idx = 0
    # with torch.no_grad():
    for j, (i_x, i_seq, i_actual) in enumerate(test_loader):
        global cam
        cam = []
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank)#.type(dtype=torch.float32)
        i_actual = i_actual.to(rank)
        i_batch = len(i_actual)
        
        # forward pass     
        src_mask = None 
        iter_y_pred, attn_prob, attn_grad = model(i_x,src_mask ,i_seq, need_grad) 
        loss = torch.sum(iter_y_pred)
        # backward pass
        optimizer.zero_grad()
        loss.backward()

        if which_method == 'grad':
            # importance = torch.abs(attn_grad[0])*attn_prob
            importance = nn.ReLU()(attn_grad[0])*attn_prob
            importance = torch.mean(importance, dim=1).reshape((i_batch,-1))
        elif which_method == 'attn':
            importance = attn_prob
            importance = torch.mean(importance, dim=1).reshape((i_batch,-1))

        store_all_importance[idx:idx+i_batch,:] = importance
        idx += i_batch

    with torch.no_grad():
        np.save(f'./model/importance_{which_data}_{which_method}', store_all_importance.to('cpu'))
        
if __name__=='__main__':
    cp_1 = time.time()
    num_epochs = 1
    init_lr = 0.001
    max_m = int(3)
    ##change
    train(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
