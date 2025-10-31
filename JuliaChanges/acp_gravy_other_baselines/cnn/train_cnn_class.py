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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
# from transformer_classes import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding, EncoderLayer

writer = SummaryWriter(f"Training starting on:{date.today()}")
writer = SummaryWriter(comment="CNN model")

## Dataloader
batch_size = 512
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
    
    ohe = np.load(path+'/x_train.npy', allow_pickle=True)
    output = np.load(path+'/y_train.npy', allow_pickle=True)
    seq_len = np.array([ohe.shape[1]]*len(ohe))

    global input_dimension
    input_dimension = ohe.shape[2]
    print(ohe.shape)

    train_dataset = spiderdataset(ohe,seq_len,output,ohe.shape[0])
        
    ohe_valid = np.load(path+'/x_valid.npy', allow_pickle=True)
    output_valid = np.load(path+'/y_valid.npy', allow_pickle=True)
    seq_len_valid = np.array([ohe_valid.shape[1]]*len(ohe_valid))

    test_dataset = spiderdataset(ohe_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)  
      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return train_loader, test_loader, ohe_valid.shape[0], ohe_valid.shape[1]
    
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
        '''x: [N,L,d]'''

        # enc_output = x.permute(1,0,2)
        enc_output = self.proj(x)
        enc_output = enc_output.permute(0,2,1)

        enc_output = self.cnn1(enc_output)
        out_mean = torch.mean(enc_output, dim=-1)

        output = self.nn(out_mean)  
        return output, None, None

    
def initalize(rank, max_m, init_lr, max_len, need_grad):
    src_vocab_size= input_dimension
    d_model= 64
    num_heads=5
    num_layers=1
    d_ff=2048
    max_seq_length=max_len
    dropout=0.1
    num_classes = 1
    model = network(src_vocab_size, d_model, num_heads, num_layers, d_ff, \
        max_seq_length, dropout,num_classes, rank).to(rank)   
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    return model, criterion, optimizer

        
# ## Training loop
def train(num_epochs, init_lr, max_m):
    need_grad = False ## True while test to store the gradient of output wrt attention
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, valid_size, max_len  = make_dataset()
    model, criterion, optimizer = initalize(rank, max_m, init_lr, max_len, need_grad)
    start_from = 0
    largest_r2 = -100
    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (i_x, i_seq, i_actual) in enumerate(train_loader):
            # TODO: perform random 10% masking here on i_x based on i_seq
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank)#.type(dtype=torch.float32)
            i_actual = i_actual.to(rank)
            src_mask = None
            # forward pass
            iter_y_pred, _, _ = \
                model(i_x,src_mask ,i_seq, need_grad) 
            # print('====', iter_y_pred.size(), i_actual.size())
                
            loss = criterion(iter_y_pred, i_actual)
            avg_loss = (avg_loss*i + loss.item())/(i+1)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
            # print(attn_grad)

        with torch.no_grad():
            predicted_label = torch.zeros((valid_size, 1))
            actual_label = torch.zeros((valid_size, 1))
            count_valid = 0       
            valid_loss = 0  
            for j, (i_x, i_seq, i_actual)  in enumerate(valid_loader):
                # TODO: perform random 10% masking here on i_x based on i_seq
                i_x = i_x.to(rank) #.type(dtype=torch.float32)
                i_seq = i_seq.to(rank)#.type(dtype=torch.float32)
                i_actual = i_actual.to(rank)
                src_mask = None
                # forward pass    
                iter_y_pred,_,_ = model(i_x,src_mask ,i_seq, need_grad) 
                size = iter_y_pred.size(0)
                predicted_label[count_valid:count_valid+size, :] = iter_y_pred 
                actual_label[count_valid:count_valid+size, :] = i_actual
                count_valid += size
            
            predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
            actual_label = actual_label.cpu().numpy().reshape((-1,1))
            valid_r2 = r2_score(actual_label, predicted_label)

        writer.add_scalar("MSE per epoch/train", avg_loss, epoch+1+start_from)
        writer.add_scalar("R2 per epoch/valid", valid_r2, epoch+1+start_from)
        # print(f'Done epoch {epoch+1+start_from}, MSE Loss: {avg_loss}, valid R2:{valid_r2}')
        if valid_r2 >= largest_r2:
            torch.save(model, f'./model/best.pth')
            largest_r2 = valid_r2
        
if __name__=='__main__':
    cp_1 = time.time()
    num_epochs = 2000
    init_lr = 0.001
    max_m = int(4)
    ##change
    train(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
