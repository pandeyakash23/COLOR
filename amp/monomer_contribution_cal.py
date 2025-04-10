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
from complor import dataset, complor_network

   
which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')

## Dataloader
batch_size = 256
class spiderdataset(Dataset) :
    def __init__(self,ohe, classes,seq_len,output, n_samples) :
        # data loading
        self.ohe = torch.from_numpy(ohe.astype(np.float32))
        self.seq_len = torch.from_numpy(seq_len.astype(int64))
        self.classes = torch.from_numpy(classes.astype(int64)) 
        self.output = torch.from_numpy(output.astype(int64)).reshape((n_samples,))
        self.n_samples = n_samples
        
        
    def __getitem__(self,index) :
        return self.ohe[index], self.classes[index], self.seq_len[index], self.output[index]

    def __len__(self):       
        return self.n_samples      

def make_dataset(): 
        
    ohe_valid = np.load(f'./data/x_{which_data}.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load(f'./data/y_{which_data}.npy', allow_pickle=True)
    seq_len_valid = np.load(f'./data/len_{which_data}.npy', allow_pickle=True)     
 
    test_dataset = spiderdataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0], ohe_valid.shape[1]

    
def initalize():    
    init_lr = np.load('./model/init_lr.npy', allow_pickle=True)
    # init_lr = init_lr[0]
    model = torch.load('./model/best.pth')
    rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion,optimizer

def motif_identification():  
    test_loader, test_size, max_seq_len  = make_dataset()
    model, criterion,optimizer = initalize()
    rank = next(model.parameters()).device 
    store_importance = torch.zeros((test_size, max_seq_len)).to(rank)
    count_test = 0
    for _, (i_x,i_classes, i_seq, i_actual) in enumerate(test_loader):
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank).type(dtype=torch.float32)
        i_classes = i_classes.to(rank)
        i_actual = i_actual.to(rank)
        i_batch = len(i_actual)
        iter_y_pred, cam = model.forward_motif_importance(i_x, i_classes, i_seq)
        base_loss = criterion(iter_y_pred, i_actual)
        optimizer.zero_grad()
        base_loss.backward()
        
        cam = torch.abs(cam[0])
        print('Size of the gradient',cam.size())

        
        for prot in range(cam.size(0)):
            cam[prot,...] = cam[prot,...]/(torch.max(cam[prot,...])+1E-18)
            
        for m_i in range(cam.size(-1)):
            mo_level_imp = \
                model.calculate_motif_level(cam[...,m_i], m_i+1)
            

            kernel_size = max_seq_len - mo_level_imp.size(-1) + 1
            store_importance[count_test:count_test+i_batch,...] += model.assigning_importance(mo_level_imp, kernel_size, max_seq_len)

        
        count_test += i_batch
    
    with torch.no_grad():   
        store_importance = store_importance.to('cpu').numpy()
        print(store_importance[15])
        np.save(f'./model/importance_{which_data}', store_importance)
    

        
if __name__=='__main__':
    cp_1 = time.time()
    motif_identification()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
    
    
