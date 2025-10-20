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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix,mean_absolute_error
from complor import dataset, complor_network


writer = SummaryWriter(f"Training starting on:{date.today()}")
writer = SummaryWriter(comment="COLOR model")

parser = argparse.ArgumentParser(description='Com-PLOR')
parser.add_argument('--num_epochs', default=2500, type=int,
                    metavar='N',
                    )
parser.add_argument('--device', default='cpu', type=str
                    )


# Load the categorical features
amino_acids = np.load("./acp_gravy_onlymasktrain/model/categorical_variables.npy", allow_pickle=True)
global q
q = len(amino_acids)
print('Amino acids:', amino_acids)

args = parser.parse_args()
num_epochs = args.num_epochs
device = args.device

## Dataloader
batch_size = 256

def mask_amino_acids(ohe, per, amino_acids):
    """
    Randomly mask per% of amino acids in one-hot encoded sequences with 'X'.
    
    Parameters:
    -----------
    ohe : numpy.ndarray
        One-hot encoded array of shape (n_samples, seq_length, n_amino_acids)
    per : float
        Percentage of amino acids to mask (0-100)
    amino_acids : list
        List of amino acid codes. 'X' should be the last element.
    
    Returns:
    --------
    masked_ohe : numpy.ndarray
        One-hot encoded array with masked positions
    """
    # Create a copy to avoid modifying the original
    masked_ohe = ohe.copy()
    
    # Get the index of 'X' (masking token)
    mask_idx = np.where(amino_acids == 'X')[0][0]
    
    # Get dimensions
    n_samples, seq_length, n_features = masked_ohe.shape
    
    # For each sample
    for i in range(n_samples):
        # Find positions that have actual amino acids (not padding/zeros)
        # Sum across the feature dimension to find non-zero positions
        valid_positions = np.where(np.sum(masked_ohe[i], axis=1) > 0)[0]
        
        # Calculate number of positions to mask
        n_to_mask = int(len(valid_positions) * per / 100.0)
        
        if n_to_mask > 0:
            # Randomly select positions to mask
            mask_positions = np.random.choice(valid_positions, size=n_to_mask, replace=False)
            
            # Mask selected positions with 'X'
            for pos in mask_positions:
                # Zero out the current amino acid
                masked_ohe[i, pos, :] = 0
                # Set 'X' position to 1
                masked_ohe[i, pos, mask_idx] = 1
    
    return masked_ohe


def make_dataset_with_masking(mask_percentage=0):
    """
    Modified make_dataset function that applies masking to training and validation data.
    
    Parameters:
    -----------
    mask_percentage : float
        Percentage of amino acids to mask (0-100)
    
    Returns:
    --------
    train_loader : DataLoader
    test_loader : DataLoader
    n_valid_samples : int
    """
    # Load training data
    ohe = np.load("./acp_gravy/data/x_train.npy", allow_pickle=True)
    output = np.load('./acp_gravy/data/y_train.npy', allow_pickle=True)
    seq_len = np.load('./acp_gravy/data/len_train.npy', allow_pickle=True) 
    
    # Apply masking if percentage > 0
    if mask_percentage > 0:
        ohe = mask_amino_acids(ohe, mask_percentage, amino_acids)
    
    classes = np.argmax(ohe, axis=2)
    
    global q
    q = ohe.shape[-1]
 
    train_dataset = dataset(ohe, classes, seq_len, output, ohe.shape[0])    
        
    # Load validation data
    ohe_valid = np.load('./acp_gravy/data/x_valid.npy', allow_pickle=True)
    output_valid = np.load('./acp_gravy/data/y_valid.npy', allow_pickle=True)
    seq_len_valid = np.load('./acp_gravy/data/len_valid.npy', allow_pickle=True)
    
    # # Apply masking to validation data
    # if mask_percentage > 0:
    #     ohe_valid = mask_amino_acids(ohe_valid, mask_percentage, amino_acids)
    
    classes_valid = np.argmax(ohe_valid, axis=2)
 
    test_dataset = dataset(ohe_valid, classes_valid, seq_len_valid, output_valid, ohe_valid.shape[0])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)  
      
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)   
    
    return train_loader, test_loader, ohe_valid.shape[0]

# def make_dataset():        
#     ohe = np.load('./data/x_train.npy', allow_pickle=True)
#     classes = np.argmax(ohe, axis=2)
#     output = np.load('./data/y_train.npy', allow_pickle=True)
#     seq_len = np.load('./data/len_train.npy', allow_pickle=True) 
    
#     global q
#     q = ohe.shape[-1]
 
#     train_dataset = dataset(ohe,classes,seq_len,output,ohe.shape[0])    
        
#     ohe_valid = np.load('./data/x_valid.npy', allow_pickle=True)
#     classes_valid = np.argmax(ohe_valid, axis=2)
#     output_valid = np.load('./data/y_valid.npy', allow_pickle=True)
#     seq_len_valid = np.load('./data/len_valid.npy', allow_pickle=True)
    
 
#     test_dataset = dataset(ohe_valid,classes_valid,seq_len_valid,output_valid,ohe_valid.shape[0])

#     train_loader = DataLoader(dataset=train_dataset,
#                             batch_size=batch_size,
#                             shuffle=True)  
      
#     test_loader = DataLoader(dataset=test_dataset,
#                             batch_size=batch_size,
#                             shuffle=False)   
    
#     return train_loader, test_loader, ohe_valid.shape[0]


    
def initalize(rank, max_m, init_lr):
    d = 4
    num_classes = 1 ## one property prediction
    model = complor_network(num_classes, q,d,max_m,rank).to(rank)     
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    
    ## saving q,d,max_m for later use
    save_dict = {'q':q, 'd':d, 'max_m':max_m}
    np.save('./acp_gravy_onlymasktrain/model/save_dict.npy', save_dict) 
    
    return model, criterion, optimizer

def random_masking_data(train_loader, valid_loader, valid_size):
    pass


## Training loop
def train(num_epochs, init_lr, max_m):
    rank = device
    model, criterion, optimizer = initalize(rank, max_m, init_lr)
    start_from = 0
    largest_r2 = -1000
    for epoch in range(num_epochs):
        avg_loss = 0
        train_loader, valid_loader, valid_size  = make_dataset_with_masking(10)
        for i, (i_x,i_classes, i_seq, i_actual) in enumerate(train_loader):
            i_x = i_x.to(rank) #.type(dtype=torch.float32)
            i_seq = i_seq.to(rank).type(dtype=torch.float32)
            i_classes = i_classes.to(rank)
            i_actual = i_actual.to(rank)
            
            # forward pass    
            iter_y_pred = model(i_x, i_classes, i_seq) ## get the output in [batch, seq_len, feature_size]
            loss = criterion(iter_y_pred, i_actual)
            avg_loss = (avg_loss*i + loss.item())/(i+1)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

        with torch.no_grad():
            predicted_label = torch.zeros((valid_size, 1))
            actual_label = torch.zeros((valid_size, 1))
            count_valid = 0         
            for j, (i_x,i_classes, i_seq, i_actual) in enumerate(valid_loader):
                i_x = i_x.to(rank) #.type(dtype=torch.float32)
                i_seq = i_seq.to(rank).type(dtype=torch.float32)
                i_classes = i_classes.to(rank)
                i_actual = i_actual.to(rank)
                
                # forward pass    
                iter_y_pred = model(i_x, i_classes, i_seq)
                size = iter_y_pred.size(0)
                predicted_label[count_valid:count_valid+size, :] = iter_y_pred 
                actual_label[count_valid:count_valid+size, :] = i_actual
                count_valid += size
            
            predicted_label = predicted_label.cpu().numpy().reshape((-1,1))
            actual_label = actual_label.cpu().numpy().reshape((-1,1))
            valid_r2 = r2_score(actual_label, predicted_label)
            mae = mean_absolute_error(actual_label, predicted_label)
            
                    
        writer.add_scalar("MSE Loss per epoch/train", avg_loss, epoch+1+start_from)
        writer.add_scalar("R2 Loss per epoch/valid", valid_r2, epoch+1+start_from)
        writer.add_scalar("MAE per epoch/valid", mae, epoch+1+start_from)
        
        if valid_r2 >= largest_r2:
            torch.save(model, f'./acp_gravy_onlymasktrain/model/best.pth')
            largest_r2 = valid_r2
        
if __name__=='__main__':
    cp_1 = time.time()
    init_lr = 0.0005
    np.save('./acp_gravy_onlymasktrain/model/init_lr.npy', init_lr)
    max_m = int(1)
    ##change
    train(num_epochs, init_lr, max_m)
    cp_2 = time.time()
    print('Time Taken',cp_2-cp_1)
