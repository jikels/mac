'''

The code was adapted from: https://github.com/resibots/kaushik_2020_famle

MIT License

Copyright (c) 2020 ResiBots

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

'''

import torch
import torch.utils.data
from torch import nn
import numpy as np
from mbmrl_torch.neural_networks.base import NNBase

class EmbeddingNN(NNBase):
    def __init__(
        self,
        dim_in,
        dim_out,
        embedding_dim,
        num_tasks,
        hidden,
        dropout=0.1,
        activation="tanh",
        cuda=True,
        seed=42):

        # init parent class
        super().__init__(cuda)

        # define parameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.dropout_p = dropout
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        self.activation_name = activation
        
        # choose activation functions
        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        try:
            self.activation = getattr(nn.functional, activation)
        except AttributeError:
            print("Model: Activation function not available -> default to tanh")
            self.activation = torch.Tanh()

        self.output_activation = nn.Identity()
    
        # construct layers
        self.layers = nn.ModuleList()
        self.embeddings = nn.Embedding(self.num_tasks, self.embedding_dim)
        in_size = self.dim_in + self.embedding_dim
        for i, next_size in enumerate(hidden):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.layers.append(fc)
        self.fcout = nn.Linear(hidden[-1], dim_out)
        self.dropout = nn.Dropout(p=dropout)

        # loss function
        self._loss = nn.MSELoss()

        # init data statistics
        self.data_mean_input = torch.zeros(dim_in).to(self._device)
        self.data_mean_output = torch.zeros(dim_out).to(self._device)
        self.data_std_input = torch.ones(dim_in).to(self._device)
        self.data_std_output = torch.ones(dim_out).to(self._device)
        self.fixed_task_id = None

        # set seed and cuda
        self.seed = seed
        torch.manual_seed(seed)
        if self.cuda_enabled:
            torch.cuda.manual_seed(seed)
        self.to(self._device)
    
    def forward(self, x, task_id=None):

        x = self._normalize_input(x)

        # use task_id to select task embedding
        # (embdedding_dim, which embedding to use)
        # e.g. task 0 -> [0,0,0,0,0,0]
        embedding = self.embeddings(task_id).reshape(-1, self.embedding_dim)
        
        # concatenate embedding and input
        x = torch.cat((x,embedding), 1)

        for layer in enumerate(self.layers):
            x = layer[1](x)
            x = self.activation(x)
        preactivation = self.fcout(x)
        return self.output_activation(preactivation)

    def predict(self, x, task_id=None):
        if task_id is None:
            shape = (x.shape[0],1) if torch.is_tensor(x) else (x.shape[0],1)
            task_id = torch.ones(shape).long() * self.fixed_task_id
        task_id = task_id.to(self._device)
        if torch.is_tensor(x):
            y_pred = self.forward(x,task_id)
            return self._denormalize_output(y_pred).detach()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self._device)
            y_pred = self.forward(x,task_id)
            return self._denormalize_output(y_pred).cpu().detach().numpy()
    
    def loss_function(self, y_pred,y):
        y = self._normalize_output(y)
        MSE = (y - y_pred).pow(2).sum()/y.size(0)
        return MSE

    def loss_function_numpy(self, y, y_pred):
        ''' y and y-pred is un-normalized'''
        y_normalized = (y - self.data_mean_output.cpu().numpy()) / self.data_std_output.cpu().numpy()
        y_pred_normalized = (y_pred - self.data_mean_output.cpu().numpy()) / self.data_std_output.cpu().numpy()
        MSE = np.power(y_normalized - y_pred_normalized,2).sum() / len(y)
        return MSE

    def fix_task(self, task_id=None):
        '''
        task_id : int
        Fix the task id for the network so that output can be predicted just sending the x alone.
        '''
        if task_id is not None:
            assert task_id >= 0 and task_id < self.num_tasks, "task_id must be a positive integer less than number of tasks"  
            self.fixed_task_id = task_id
    
    def get_embedding(self, task_id):
        task_id_tensor =torch.LongTensor(task_id).to(self._device)
        return self.embeddings(task_id_tensor).reshape(-1, self.embedding_dim).detach().cpu().numpy()
    
    def set_stats(self,stats):
        self.data_mean_input = torch.from_numpy(stats[0]).to(self._device).float()
        self.data_std_input =  torch.from_numpy(stats[1]).to(self._device).float()
        self.data_mean_output = torch.from_numpy(stats[2]).to(self._device).float()
        self.data_std_output = torch.from_numpy(stats[3]).to(self._device).float()

    def _normalize_input(self,x):
        return (x - self.data_mean_input) / self.data_std_input
    
    def _normalize_output(self,x):
        return (x - self.data_mean_output) / self.data_std_output

    def _denormalize_input(self,x_normalized):
        return (x_normalized * self.data_std_input + self.data_mean_input)

    def _denormalize_output(self,x_normalized):
        return (x_normalized * self.data_std_output + self.data_mean_output)
        
    def save(self, file_path):
        kwargs = {  "dim_in": self.dim_in,
                    "hidden": self.hidden, 
                    "dim_out": self.dim_out,
                    "cuda": self.cuda_enabled, 
                    "seed": self.seed, 
                    "dropout": self.dropout_p, 
                    "activation": self.activation_name,
                    "embedding_dim": self.embedding_dim, 
                    "num_tasks": self.num_tasks}
        state_dict = self.state_dict()
        other = {  "data_mean_input":self.data_mean_input,
                    "data_std_input": self.data_std_input,
                    "data_mean_output": self.data_mean_output,
                    "data_std_output": self.data_std_output,
                    "fixed_task_id": self.fixed_task_id} 
        torch.save({"kwargs":kwargs, "state_dict":state_dict, "other":other}, file_path)