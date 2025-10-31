import numpy as np
import torch
import torch.nn as nn
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, precision_score
from datetime import date
import time
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
)
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

which_data = input('Enter the dataset for which you want to calculate the token importance (train,valid,test):')


writer = torch.utils.tensorboard.SummaryWriter(comment="CNN model")

rank = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
batch_size = 512
num_epochs = 2000
init_lr = 0.001
max_m = int(4)

# Dataset
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
        enc_output = self.proj(x)
        enc_output = enc_output.permute(0,2,1)
        enc_output = self.cnn1(enc_output)
        out_mean = torch.mean(enc_output, dim=-1)
        output = self.nn(out_mean)  
        return output, None, None

# Model wrapper for Captum
class CaptumModelWrapper(nn.Module):
    def __init__(self, model, src_mask, seq_len, need_grad,i_actual):
        super().__init__()
        self.model = model
        self.src_mask = src_mask
        self.seq_len = seq_len
        self.need_grad = need_grad
        self.criterion = nn.MSELoss()
        self.i_actual = i_actual

    def forward(self, x):
        # x,y =input[0], input[1]
        output, _, _ = self.model(x, self.src_mask, self.seq_len, self.need_grad)
        # if len(output)!=len(self.i_actual):
        #     n = int(len(output)/len(self.i_actual))
        #     y =  self.i_actual.repeat(n)  # if tensor is 2D
        # else:
        #     y = self.i_actual
        
        loss = output
            
        # loss = self.criterion(output,y)
        # loss = loss.unsqueeze(0)
        # loss = loss.repeat(len(output)).unsqueeze(-1)
        # print(loss.size())
        return loss

# Load dataset
def make_dataset(): 
    path = '../data' 
    ohe_valid = np.load(path + f'/x_{which_data}.npy', allow_pickle=True)
    output_valid = np.load(path + f'/y_{which_data}.npy', allow_pickle=True)
    seq_len_valid = np.array([ohe_valid.shape[1]] * len(ohe_valid))
    test_dataset = spiderdataset(ohe_valid, seq_len_valid, output_valid, ohe_valid.shape[0])
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)   
    return test_loader, ohe_valid.shape[0]

@torch.no_grad()
def initalize(rank, max_m, init_lr):
    model = torch.load('./model/best.pth', map_location=rank)
    model.eval().to(rank)
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    return model, criterion, optimizer

def test(num_epochs, init_lr, max_m):
    need_grad = False
    test_loader, valid_size = make_dataset()
    model, criterion, optimizer = initalize(rank, max_m, init_lr)

    # Attribution storage
    all_attributions_ig = []
    all_attributions_gs = []
    all_attributions_dl = []
    all_attributions_dlshap = []

    with torch.no_grad():
        for j, (i_x, i_seq, i_actual) in enumerate(test_loader):
            i_x = i_x.to(rank)
            i_seq = i_seq.to(rank)
            i_actual = i_actual.to(rank)
            src_mask = None

            # Captum wrapper
            wrapped_model = CaptumModelWrapper(model, src_mask, i_seq, need_grad, i_actual)
            wrapped_model.eval()

            # Run Captum methods and store
            
            # try:
            # ig = IntegratedGradients(model)
            ig = IntegratedGradients(wrapped_model)
            attributions, _= ig.attribute(i_x, target=0, return_convergence_delta=True)
            all_attributions_ig.append(attributions.detach().cpu())

            gs = GradientShap(wrapped_model)
            baseline_dist = torch.randn(10, *i_x.shape[1:]).to(rank) * 0.001
            attributions, _ = gs.attribute(i_x, target=0,  stdevs=0.09, n_samples=4, baselines=baseline_dist, return_convergence_delta=True)
            all_attributions_gs.append(attributions.detach().cpu())

            dl = DeepLift(wrapped_model)
            attributions, _ = dl.attribute(i_x, target=0,  baselines=torch.zeros_like(i_x), return_convergence_delta=True)
            all_attributions_dl.append(attributions.detach().cpu())

            
            dl_shap = DeepLiftShap(wrapped_model)
            baseline_dist = torch.randn(10, *i_x.shape[1:]).to(rank) * 0.0001
            attributions, _ = dl_shap.attribute(i_x, target=0,  baselines=baseline_dist, return_convergence_delta=True)
            all_attributions_dlshap.append(attributions.detach().cpu())

    # Concat and save attributions
    all_attributions_ig = torch.cat(all_attributions_ig, dim=0)
    all_attributions_gs = torch.cat(all_attributions_gs, dim=0)
    all_attributions_dl = torch.cat(all_attributions_dl, dim=0)
    all_attributions_dlshap = torch.cat(all_attributions_dlshap, dim=0)
    
    with torch.no_grad():
        all_attributions_ig = torch.mean(all_attributions_ig, dim=-1)
        all_attributions_gs = torch.mean(all_attributions_gs, dim=-1)
        all_attributions_dl = torch.mean(all_attributions_dl, dim=-1)
        all_attributions_dlshap = torch.mean(all_attributions_dlshap, dim=-1)
      
        all_attributions_ig = all_attributions_ig.to('cpu')
        all_attributions_gs = all_attributions_gs.to('cpu')
        all_attributions_dl = all_attributions_dl.to('cpu')
        all_attributions_dlshap = all_attributions_dlshap.to('cpu')

    np.save(f'./model/importance_{which_data}_ig', all_attributions_ig)
    np.save(f'./model/importance_{which_data}_gs',all_attributions_gs)
    np.save(f'./model/importance_{which_data}_dl',all_attributions_dl)
    np.save(f'./model/importance_{which_data}_dlshap',all_attributions_dlshap)

    print("All attribution scores saved in './model/'.")

    print(all_attributions_ig.shape, all_attributions_gs.shape, all_attributions_dl.shape, all_attributions_dlshap.shape)

# Run evaluation
cp_1 = time.time()
test(num_epochs=1, init_lr=0.001, max_m=3)
cp_2 = time.time()
print('Time Taken:', cp_2 - cp_1)