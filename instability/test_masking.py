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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error,r2_score
from complor import dataset, complor_network


top_per = np.load('./model/percentage_unmasked.npy', allow_pickle=True).item()
top_per = int(top_per)

## Dataloader
batch_size = 256

def masking_function(ohe, seq_len, importance):
    revised_x  = ohe
    num_ex = ohe.shape[0]
    for k in range(num_ex):
        l = int(seq_len[k])
        ex_token = np.argsort(importance[k,0:l], axis=-1)
        ex_token = ex_token[::-1]
        top_num_token = int(ceil(l*top_per/100))
        sample_imp = tuple(ex_token[top_num_token:].tolist())
        revised_x[k,sample_imp,:] = 0
    return revised_x  

def make_dataset(): 
        
    ohe_valid = np.load('./data/x_test.npy', allow_pickle=True)
    imp_token_valid = np.load('./model/importance_test.npy', allow_pickle=True)
    output_valid = np.load('./data/y_test.npy', allow_pickle=True)
    seq_len_valid = np.load('./data/len_test.npy', allow_pickle=True)   
    ohe_valid = masking_function(ohe_valid, seq_len_valid, imp_token_valid)
    classes_valid = np.argmax(ohe_valid, axis=2)  
 
    test_dataset = dataset(ohe_valid,classes_valid,seq_len_valid,\
        output_valid,ohe_valid.shape[0])

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0]

def initalize():
    
    model = torch.load('./model/best_masking.pth')
    rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    
    return model, criterion


def test():
    test_loader, valid_size  = make_dataset()
    model, criterion = initalize()
    rank = next(model.parameters()).device 
    with torch.no_grad():
        predicted_label = torch.zeros((valid_size, 1))
        actual_label = torch.zeros((valid_size, 1))
        count_valid = 0         
        for j, (i_x,i_classes, i_seq, i_actual) in enumerate(test_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
            # forward pass    
            iter_y_pred = model(i_x, i_classes, i_seq)
            base_loss = criterion(iter_y_pred, i_actual)
            base_loss = base_loss.item()
            size = iter_y_pred.size(0)
            predicted_label[count_valid:count_valid+size, :] = iter_y_pred 
            actual_label[count_valid:count_valid+size, :] = i_actual
            count_valid += size
        
        predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
        # print(predicted_label)
        actual_label = actual_label.cpu().numpy().reshape((-1,1))
        
        valid_r2 = r2_score(actual_label, predicted_label)
        mae = mean_absolute_error(actual_label, predicted_label)
        print('MSE:',base_loss)
        print(f'Test R2:{valid_r2}, MAE:{mae}')

        
if __name__=='__main__':
    cp_1 = time.time()
    test()
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)