
import torch
import abc

class NNBase(torch.nn.Module):

    def __init__(self, dim_in, dim_out, cuda):
        super(NNBase, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        if cuda:
            self._device = torch.device('cuda:0')
        else:
            self._device = torch.device('cpu')
            
    @property
    def device(self):
        return self._device
    
    @abc.abstractmethod
    def forward(self,input):
        pass

    @abc.abstractmethod
    def predict(self,predict):
        pass