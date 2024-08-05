import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from torch import nn
from torch.autograd import Variable
import pandas as pd
from operator import add
import time
import argparse
import json


# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class DAPLModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.main = nn.Sequential(
            nn.Linear(281, 250),
            nn.ReLU(),
            nn.Linear(250, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1, bias=False)
        )
        
    def forward(self, x):
        return self.main(x)

def do_base_learning(model, x_batch, R_matrix_batch, ystatus_batch, lr_inner, n_inner, reg_scale):
    new_model = DAPLModel()
    new_model.load_state_dict(model.state_dict())
    inner_optimizer = torch.optim.AdamW(new_model.parameters(), lr=lr_inner, weight_decay=reg_scale)  # Use AdamW optimizer
    
    for i in range(n_inner):
        x_batch = Variable(torch.FloatTensor(x_batch), requires_grad=True)
        R_matrix_batch = Variable(torch.FloatTensor(R_matrix_batch), requires_grad=True)
        ystatus_batch = Variable(torch.FloatTensor(ystatus_batch), requires_grad=True)
        
        theta = new_model(x_batch)
        exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
        theta = torch.reshape(theta, [x_batch.shape[0]])
                            
        loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))), torch.reshape(ystatus_batch, [x_batch.shape[0]])))
      
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        
    return new_model

def do_base_eval(trained_model, x_test, y_test, ystatus_test):
    x_batch = torch.FloatTensor(x_test)
    pred_batch_test = trained_model(x_batch)
    cind = CIndex(pred_batch_test, y_test, np.asarray(ystatus_test))
        
    ystatus_batch = torch.FloatTensor(ystatus_test)
    R_matrix_batch = np.zeros([y_test.shape[0], y_test.shape[0]], dtype=int)
    for i in range(y_test.shape[0]): 
        for j in range(y_test.shape[0]):
            R_matrix_batch[i,j] = y_test[j] >= y_test[i]
    R_matrix_batch = torch.FloatTensor(R_matrix_batch)
        
    theta = trained_model(x_batch)
    exp_theta = torch.reshape(torch.exp(theta), [x_batch.shape[0]])
    theta = torch.reshape(theta, [x_batch.shape[0]])
                            
    loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta, R_matrix_batch), dim=1))), torch.reshape(ystatus_batch, [x_batch.shape[0]])))
                            
    return loss.item(), cind

def CIndex(pred, ytime_test, ystatus_test):
    concord = 0.
    total = 0.
    N_test = ystatus_test.shape[0]
    ystatus_test = np.asarray(ystatus_test, dtype=bool)
    theta = pred
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]: concord = concord + 1
                    elif theta[j] == theta[i]: concord = concord + 0.5
    return(concord/total)

def meta_learn(model, x_train, y_train, ystatus_train, x_val, y_val, ystatus_val, iterations, lr_inner, lr_outer, n_inner, batch_n, reg_scale, shots_n):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_outer)  # Use AdamW optimizer
    train_metalosses = []
    test_metalosses = []
    
    inner_optimizer_state = None
    
    for t in range(iterations):
        start = time.time()
        ind = random.sample(range(x_train.shape[0]), shots_n)
        x_batch = x_train[ind,]
        ystatus_batch = ystatus_train[ind,]
        y_batch = y_train[ind,]
        R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
        for i in range(y_batch.shape[0]): 
            for j in range(y_batch.shape[0]):
                R_matrix_batch[i,j] = y_batch[j] >= y_batch[i]
        new_model = do_base_learning(model, x_batch, R_matrix_batch, ystatus_batch, lr_inner, n_inner, reg_scale)
        
        diff = list()
        for p,new_p in zip(model.parameters(),new_model.parameters()):
            temp = Variable(torch.zeros(p.size()))
            temp.add_(p.data - new_p.data)
            diff.append(temp)
            
        for j in range(batch_n-1):
            ind = random.sample(range(x_train.shape[0]), shots_n)
            x_batch = x_train[ind,]
            ystatus_batch = ystatus_train[ind,]
            y_batch = y_train[ind,]
            R_matrix_batch = np.zeros([y_batch.shape[0], y_batch.shape[0]], dtype=int)
            for i in range(y_batch.shape[0]):
                for j in range(y_batch.shape[0]):
                    R_matrix_batch[i,j] = y_batch[j] >= y_batch[i]
            new_model = do_base_learning(model, x_batch, R_matrix_batch, ystatus_batch, lr_inner, n_inner, reg_scale)
            
            diff_next = list()
            for p,new_p in zip(model.parameters(),new_model.parameters()):
                temp = Variable(torch.zeros(p.size()))
                temp.add_(p.data - new_p.data)
                diff_next.append(temp)
                
            diff = list(map(add, diff, diff_next))
        
        diff_ave = [x/batch_n for x in diff]
        
        ind_k = 0
        for p in model.parameters():
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.add_(diff_ave[ind_k])
            ind_k += 1

        optimizer.step()
        optimizer.zero_grad()
        
        val_metaloss, val_cind = do_base_eval(model, x_val, y_val, ystatus_val)
          
        end = time.time()
        print("1 iteration time:", end-start)
        print('Iteration', t)
        print('Validation C-index:', val_cind)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    LR_INNER = config['lr_inner']
    LR_OUTER = config['lr_outer']
    SHOTS_N = config['shots_n']
    BATCH_N = config['batch_n']
    N_INNER = config['n_inner']
    REG_SCALE = config['reg_scale']
    ITER = config['iters']

    model_path = config['model_path']
    x_train = np.loadtxt(fname=config['train_feature'], delimiter=",", skiprows=1)
    y_train = np.loadtxt(fname=config['train_time'], delimiter=",", skiprows=1)
    ystatus_train = np.loadtxt(fname=config['train_status'], delimiter=",", skiprows=1)
    x_val = np.loadtxt(fname=config['val_feature'], delimiter=",", skiprows=1)
    y_val = np.loadtxt(fname=config['val_time'], delimiter=",", skiprows=1)
    ystatus_val = np.loadtxt(fname=config['val_status'], delimiter=",", skiprows=1)
    
    print("Training size", x_train.shape[0])
    daplmodel = DAPLModel()
    meta_learn(model=daplmodel, x_train=x_train, y_train=y_train, ystatus_train=ystatus_train,
               x_val=x_val, y_val=y_val, ystatus_val=ystatus_val,
               iterations=ITER, lr_inner=LR_INNER, lr_outer=LR_OUTER, n_inner=N_INNER,
               batch_n=BATCH_N, reg_scale=REG_SCALE, shots_n=SHOTS_N)
    
    filepath = model_path + "metamodel_lrinner" + str(LR_INNER) + "lrouter" + str(LR_OUTER) + "shotsn" + str(SHOTS_N) + "batchn" + str(BATCH_N) + "ninner" + str(N_INNER) + "regscale" + str(REG_SCALE) + "iter" + str(ITER) + ".pt"
    torch.save(daplmodel.state_dict(), filepath)
    print("Model saved in file: %s" % filepath)
