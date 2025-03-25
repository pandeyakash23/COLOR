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
batch_size = 512

def make_dataset(): 
        
    ohe_valid = np.load(f'./data/x_{which_data}.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load(f'./data/y_{which_data}.npy', allow_pickle=True)
    seq_len_valid = np.load(f'./data/len_{which_data}.npy', allow_pickle=True)     
 
    test_dataset = dataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

      
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

def sensitivity():
    test_loader, valid_size, max_len  = make_dataset()
    model, criterion = initalize()
    rank = next(model.parameters()).device 
    save_dict = np.load('./model/save_dict.npy', allow_pickle=True).item()
    l = save_dict['q']
    b = save_dict['d']
    h = save_dict['max_m']
    l,b,h = int(l),int(b),int(h)
    num_iter = 3
    ''' num_iter decides the number of times the 
    permutation based method is done for every feature'''
    f_importance = torch.zeros((valid_size,l*b*h,num_iter)).to(rank)
    f_name = np.zeros((l*b*h,), dtype=object)
    
    count_valid = 0        
    with torch.no_grad():      
        for j, (i_x,i_classes, i_seq, i_actual) in enumerate(test_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            i_batch = len(i_actual)
            iter_y_pred = model.forward_feature_importance(i_x, i_classes, i_seq, [], False)
            base_loss = criterion(iter_y_pred, i_actual)
            base_loss = base_loss.item()
            
            if which_data=='test':
                mae = mean_absolute_error(i_actual.cpu().numpy().reshape((-1,1)), iter_y_pred.cpu().numpy().reshape((-1,1)))
                print('MAE:',mae,'MSE loss:',base_loss)       
    
            for iter in range(num_iter):
                count_f = 0
                for f1 in range(l):
                    for f2 in range(b):
                        for f3 in range(h):                        
                            y_after_permute = model.forward_feature_importance(i_x, i_classes, i_seq, [f1,f2,f3], True)

                            f_importance[count_valid:count_valid+i_batch,count_f,iter] = \
                                abs((y_after_permute-iter_y_pred)/iter_y_pred).reshape((i_batch,))
                            f_name[count_f,] = str(f1) + '_' + str(f2) +'_' + str(f3)
                            count_f += 1

            count_valid +=  i_batch
            
        f_importance = f_importance.to('cpu').numpy()
        f_importance = np.mean(f_importance, axis=-1) #[N,#features]
        f_max = np.max(f_importance, axis=-1).reshape((-1,1))
        f_importance = f_importance/(f_max + 1E-18)
        print('Total features', count_f)
        # print('Checking (all should be value one)', np.max(f_importance, axis=-1))
        np.save(f'./model/f_imp_{which_data}', f_importance)
        np.save(f'./model/f_name_{which_data}', f_name)

        
if __name__=='__main__':
    cp_1 = time.time()
    sensitivity()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
