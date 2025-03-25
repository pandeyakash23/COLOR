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
from complor import complor_network, dataset
   

## Dataloader
batch_size = 1

def make_dataset(): 
        
    ohe_valid = np.load('./data/x_test.npy', allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load('./data/y_test.npy', allow_pickle=True)
    seq_len_valid = np.load('./data/len_test.npy', allow_pickle=True) 
  
    test_dataset = dataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])
      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return  test_loader, ohe_valid.shape[0], ohe_valid.shape[1]

    
def initalize(init_lr):    
    model = torch.load('./model/best.pth')
    rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion, optimizer

## Training loop
def motif_identification(num_epochs, init_lr, max_m):  
    all_f_imp = np.load('./model/all_test_f_imp.npy', allow_pickle=True)
    test_loader, test_size, max_len  = make_dataset()
    model, _, _ = initalize(init_lr)
    rank = next(model.parameters()).device 
    f_name = np.load('./model/f_name.npy', allow_pickle=True)
    store_all_imp = torch.zeros((test_size, max_len)).to(rank)
    
    for sample, (i_x,i_classes, i_seq, i_actual) in enumerate(test_loader):
        i_x = i_x.to(rank) #.type(dtype=torch.float32)
        i_seq = i_seq.to(rank).type(dtype=torch.float32)
        i_classes = i_classes.to(rank)
        i_actual = i_actual.to(rank).reshape((1,-1))
        
        f_importance = all_f_imp[sample]
        idx = np.argsort(f_importance)
        idx = idx[::-1]
        f_importance = f_importance[idx]
        # print(f_importance)
        use_names = f_name[idx]
        f_pointer = 0 
        while f_pointer < 30:
            ''' # 10 here means that we are using 
            only top 10 features in representation R to obtain monomer importance.
            Anyways after certain imporance values, the elements in R do not contribute a lot to 
            the result. But this can surely be changed.'''
            feature = use_names[f_pointer].split('_')
            l,b,h = int(feature[0]), int(feature[1]), int(feature[2])                 
            with torch.no_grad():  
                
                if f_pointer == 0:
                    overall_imp_segments = torch.zeros((int(i_seq.item()),)).to(rank)
                    trace_visitation = torch.zeros((int(i_seq.item()),)).to(rank)
                    
                overall_imp_segments, trace_visitation = \
                model.importance_calculation(i_x, i_classes, i_seq, [l,b,h], overall_imp_segments, f_importance[f_pointer], trace_visitation)    
                        
            f_pointer += 1
            
    #     ''' for every example, place token X at masking location '''
        store_all_imp[sample,:] = overall_imp_segments
    #     important_tokens = torch.argsort(overall_imp_segments, descending=True)
    #     top_num_token = int(ceil(i_x.size(1)*top_per/100))
    #     sample_imp = tuple(important_tokens[top_num_token:].to('cpu').tolist())
    #     i_x[0,sample_imp,:] = 0
    #     i_x[0,sample_imp,-1] = 1
        
    #     if sample == 0:
    #         modified_x = i_x
    #     else:
    #         modified_x = torch.cat((modified_x, i_x), dim=0)
            
        print('Done:', sample+1)

    # modified_x = modified_x.to('cpu')
    # # print(modified_x.shape)
    # # np.save('masked_x', modified_x)
    
    store_all_imp = store_all_imp.to('cpu')
    np.save('./data/test_importance', store_all_imp)
    

        
if __name__=='__main__':
    cp_1 = time.time()
    num_epochs = 1
    init_lr = 0.0003
    max_m = int(1)
    ##change
    motif_identification(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
    
    
