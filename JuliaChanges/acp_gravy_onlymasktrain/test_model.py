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

## Dataloader
batch_size = 256

amino_acids = np.load("./acp_gravy_onlymasktrain/model/categorical_variables.npy", allow_pickle=True)


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
    ohe_valid = np.load("./acp_gravy_onlymasktrain/data/x_test.npy", allow_pickle=True)
    classes_valid = np.argmax(ohe_valid, axis=2)
    output_valid = np.load('./acp_gravy_onlymasktrain/data/y_test.npy', allow_pickle=True)
    seq_len_valid = np.load('./acp_gravy_onlymasktrain/data/len_test.npy', allow_pickle=True) 
    
    # Apply masking if percentage > 0
    if mask_percentage > 0:
        ohe_valid = mask_amino_acids(ohe_valid, mask_percentage, amino_acids)

 
    test_dataset = dataset(ohe_valid, classes_valid, seq_len_valid, output_valid, ohe_valid.shape[0])    
        
   
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)  
      
    
    return test_loader, ohe_valid.shape[0]


    
def initalize():
    
    model = torch.load('./acp_gravy_onlymasktrain/model/best.pth', weights_only=False)
    rank = next(model.parameters()).device 
    model.eval().to(rank) 
    print('Number of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    
    return model, criterion

def test():
    test_loader, valid_size  = make_dataset_with_masking(0)
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
