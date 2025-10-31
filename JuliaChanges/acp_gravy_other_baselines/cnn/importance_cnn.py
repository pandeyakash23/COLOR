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
# from transformer_classes import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, EncoderLayer

# import sys
# sys.path.insert(0, '../')
# from generate_property import output_property

# cam = []

which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')
    
## Dataloader
batch_size = 256
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

    global input_dimension
    input_dimension = ohe_valid.shape[2]
    print(ohe_valid.shape)

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
        
        self.proj =  nn.Sequential(nn.Linear(src_vocab_size,self.d_model),
                                        )  
        
        ### cnn layers        
        self.cnn1 = nn.Sequential( nn.Conv1d(self.d_model,128,2, stride=1), 
                                   nn.ReLU(),
                                   nn.MaxPool1d(2,stride=2),
                                   nn.Conv1d(128,64,2, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2,stride=2),
                                   nn.Conv1d(64,32,2, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool1d(2,stride=2),
                                   nn.Conv1d(32,32,2, stride=1),
                                   nn.ReLU(),
                                #    nn.MaxPool1d(2,stride=2),
                                   nn.Conv1d(32,32,2, stride=1),
                                   )
        
        self.nn = nn.Sequential(
                                nn.Linear(32,16),
                                nn.ReLU(),
                                nn.Linear(16,8),
                                nn.ReLU(),
                                nn.Linear(8,num_classes)
                                )    


    def forward(self, x, src_mask, seq_len, need_grad):

        # enc_output = x.permute(1,0,2)
        enc_output = self.proj(x)
        enc_output.register_hook(extract)
        enc_output = enc_output.permute(0,2,1)

        enc_output = self.cnn1(enc_output)
        out_mean = torch.mean(enc_output, dim=-1)

        output = self.nn(out_mean)  
        return output, None, None
        
def initalize(rank, max_m, init_lr):
    
    model = torch.load('./model/best.pth', map_location='cuda:3')
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion, optimizer

## Training loop
def train(num_epochs, init_lr, max_m):
    ''' top_per controls the top % tokens '''
    need_grad = True
    rank = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
        # print(iter_y_pred.reshape(1,-1))
        loss = torch.sum(iter_y_pred)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        
        importance = cam[0] #[L,N,f]
        # print('Size should be [20,b,30]', importance.size())
        print('importance size before relu', importance.size())
        importance = nn.ReLU()(importance)
        importance = torch.mean(importance, dim=-1) #[N,L]
        print('importance size after mean', importance.size())
    
        # print(importance.size())
        store_all_importance[idx:idx+i_batch,:] = importance
        idx += i_batch

    with torch.no_grad():
        np.save(f'./model/importance_{which_data}', store_all_importance.to('cpu'))
        
if __name__=='__main__':
    cp_1 = time.time()
    num_epochs = 1
    init_lr = 0.001
    max_m = int(3)
    ##change
    train(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
