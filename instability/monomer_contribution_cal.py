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
batch_size = 1
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
    
    model = torch.load('./model/best.pth')
    rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion

def motif_identification():  
    all_f_imp = np.load(f'./model/f_imp_{which_data}.npy', allow_pickle=True)
    test_loader, test_size, max_seq_len  = make_dataset()
    model, _ = initalize()
    rank = next(model.parameters()).device 
    f_name = np.load(f'./model/f_name_{which_data}.npy', allow_pickle=True)
    store_all_imp = torch.zeros((test_size, max_seq_len)).to(rank)
    
    for sample, (i_x,i_classes, i_seq, _) in enumerate(test_loader):
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank)
        i_seq[0] = i_x.size(1)
        i_classes = i_classes.to(rank)
        # i_x = i_x[:,0:int(i_seq.item()),:]
        # i_classes = i_classes[:,0:int(i_seq.item())]
        
        f_importance = all_f_imp[sample]
        idx = np.argsort(f_importance)
        idx = idx[::-1]
        f_importance = f_importance[idx]
        # print(f_importance)
        use_names = f_name[idx]
        '''has to be done for each test example'''
        f_pointer = 0 
        while f_pointer < 50: 
            feature = use_names[f_pointer].split('_')
            l,b,h = int(feature[0]), int(feature[1]), int(feature[2])                 
            with torch.no_grad():  
                
                if f_pointer == 0:
                    overall_imp_segments = torch.zeros((int(i_seq.item()),)).to(rank)
                    trace_visitation = torch.zeros((int(i_seq.item()),)).to(rank)
                overall_imp_segments, trace_visitation = \
                model.importance_calculation(i_x, i_classes, i_seq, [l,b,h], overall_imp_segments, f_importance[f_pointer], trace_visitation)    
                        
            f_pointer += 1
        # print(overall_imp_segments[0:24])
        store_all_imp[sample,0:int(i_seq.item())] = overall_imp_segments
        print(f'Sequence {sample+1} done')
            
    with torch.no_grad():
        np.save(f'./model/importance_{which_data}', store_all_imp.to('cpu'))
    

        
if __name__=='__main__':
    cp_1 = time.time()
    motif_identification()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
    
    
