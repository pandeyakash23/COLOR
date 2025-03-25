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
import argparse
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error,accuracy_score
from complor import dataset, complor_network

top_per = input('Top how much percentage tokens should remain unmasked? ')
top_per = int(top_per)
np.save('./model/percentage_unmasked', top_per)

writer = SummaryWriter(f"Training starting on:{date.today()}")
writer = SummaryWriter(comment="Com-PLOR Masking model")

parser = argparse.ArgumentParser(description='Com-PLOR')
parser.add_argument('--num_epochs', default=2500, type=int,
                    metavar='N',
                    )
parser.add_argument('--device', default='cuda', type=str
                    )

args = parser.parse_args()
num_epochs = args.num_epochs
device = args.device

## Dataloader
batch_size = 128

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
        revised_x[k,sample_imp,-1] = 0
    return revised_x  

def make_dataset():        
    ohe = np.load('./data/x_train.npy', allow_pickle=True)
    imp_token = np.load('./model/importance_train.npy', allow_pickle=True)    
    output = np.load('./data/y_train.npy', allow_pickle=True)
    seq_len = np.load('./data/len_train.npy', allow_pickle=True) 
    ohe = masking_function(ohe, seq_len, imp_token)
    classes = np.argmax(ohe, axis=2)

 
    train_dataset = dataset(ohe,classes,seq_len,output,ohe.shape[0])    
        
    ohe_valid = np.load('./data/x_valid.npy', allow_pickle=True)
    imp_token_valid = np.load('./model/importance_valid.npy', allow_pickle=True)  
    output_valid = np.load('./data/y_valid.npy', allow_pickle=True)
    seq_len_valid = np.load('./data/len_valid.npy', allow_pickle=True)
    ohe_valid = masking_function(ohe_valid, seq_len_valid, imp_token_valid)  
    classes_valid = np.argmax(ohe_valid, axis=2)
    
 
    test_dataset = dataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)  
      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return train_loader, test_loader, ohe_valid.shape[0], ohe_valid.shape[1]

    
def initalize(rank, init_lr):
    save_dict = np.load('./model/save_dict.npy', allow_pickle=True).item()
    q = int(save_dict['q'])
    d = int(save_dict['d'])
    max_m = int(save_dict['max_m'])
    num_classes = 2 ## one property prediction
    model = complor_network(num_classes, q,d,max_m,rank).to(rank)     
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    return model, criterion, optimizer

# ## Training loop
def train(num_epochs, init_lr):
    rank = device
    train_loader, valid_loader, valid_size, max_len  = make_dataset()
    model, criterion, optimizer = initalize(rank, init_lr)
    start_from = 0
    largest_acc = 0
    for epoch in range(num_epochs):
        avg_loss = 0
        for i, (i_x,i_classes, i_seq, i_actual) in enumerate(train_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank)#.type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
            iter_y_pred = model(i_x, i_classes, i_seq)
                
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
            for j, (i_x,i_classes, i_seq, i_actual) in enumerate(valid_loader):
                i_x = i_x.to(rank) #.type(dtype=torch.float32)
                i_seq = i_seq.to(rank).type(dtype=torch.float32)
                i_classes = i_classes.to(rank)
                i_actual = i_actual.to(rank)
                
                # forward pass     # forward pass    
                iter_y_pred = model(i_x, i_classes, i_seq)
                loss = criterion(iter_y_pred, i_actual)
                valid_loss = (valid_loss*j + loss.item())/(j+1)
                iter_y_pred = nn.Softmax(dim=1)(iter_y_pred)
                iter_y_pred = torch.argmax(iter_y_pred, dim=1)
                size = iter_y_pred.size(0)
                predicted_label[count_valid:count_valid+size, 0] = iter_y_pred 
                actual_label[count_valid:count_valid+size, 0] = i_actual
                count_valid += size
            
            predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
            actual_label = actual_label.cpu().numpy().reshape((-1,1))
            valid_acc = accuracy_score(actual_label, predicted_label)
            
                    
        writer.add_scalar("Cross entropy Loss per epoch/train", avg_loss, epoch+1+start_from)
        writer.add_scalar("Acc Loss per epoch/valid", valid_acc, epoch+1+start_from)
        # writer.add_scalar("Cross entropy per epoch/valid", valid_loss, epoch+1+start_from)
        
        if valid_acc > largest_acc:
            torch.save(model, f'./model/best_mask.pth')
            largest_acc = valid_acc
        
if __name__=='__main__':
    cp_1 = time.time()
    init_lr = 0.0003
    train(num_epochs, init_lr)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)

