import torch
import torch.utils.data
from torch import nn
import numpy as np
from mbmrl_torch.neural_networks.base import NNBase

class MLP(NNBase):
    def __init__(
        self,
        dim_in,
        hidden,
        dim_out,
        dropout=0.1,
        activation="tanh",
        cuda=True,
        seed=42):

        # define parameters
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden = hidden
        self.activation_name = activation
        self.dropout_p = dropout
        
        # init parent class
        super(MLP, self).__init__(cuda)
        
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
        in_size = self.dim_in
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

        # set seed and cuda
        self.seed = seed
        torch.manual_seed(seed)
        if self.cuda_enabled:
            torch.cuda.manual_seed(seed)
        self.to(self._device)
    
    def forward(self, x):
        x = self._normalize_input(x)
        for layer in enumerate(self.layers):
            x = layer[1](x)
            x = self.activation(x)
        preactivation = self.fcout(x)
        return self.output_activation(preactivation)

    def predict(self, x):
        if torch.is_tensor(x):
            y_pred = self.forward(x)
            return self._denormalize_output(y_pred).detach()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            x.cuda() if self.cuda_enabled else x.cpu()
            y_pred = self.forward(x)
            return self._denormalize_output(y_pred).detach().numpy()
    
    def loss_function(self,y,y_pred):
        y = self._normalize_output(y)
        MSE = (y - y_pred).pow(2).sum()/y.size(0)
        return MSE
    
    def set_stats(self,stats):
        self.data_mean_input = torch.from_numpy(stats[0]).to(self._device).float()
        self.data_std_input =  torch.from_numpy(stats[1]).to(self._device).float()
        self.data_mean_output = torch.from_numpy(stats[2]).to(self._device).float()
        self.data_std_output = torch.from_numpy(stats[3]).to(self._device).float()

    def _normalize_input(self,x):
        return (x - self.data_mean_input) / self.data_std_input
    
    def _normalize_output(self,y):
        return (y - self.data_mean_output) / self.data_std_output

    def _denormalize_input(self,x_normalized):
        return (x_normalized * self.data_std_input + self.data_mean_input)

    def _denormalize_output(self,y_normalized):
        return (y_normalized * self.data_std_output + self.data_mean_output)
        
    def save(self, file_path):
        kwargs = {  "dim_in": self.dim_in,
                    "hidden": self.hidden, 
                    "dim_out": self.dim_out,
                    "cuda": self.cuda_enabled, 
                    "seed": self.seed, 
                    "dropout": self.dropout_p, 
                    "activation": self.activation_name}
        state_dict = self.state_dict()
        stats = {  "data_mean_input":self.data_mean_input,
                    "data_std_input": self.data_std_input,
                    "data_mean_output": self.data_mean_output,
                    "data_std_output": self.data_std_output} 
        torch.save({"kwargs":kwargs, "state_dict":state_dict, "stats":stats}, file_path)
