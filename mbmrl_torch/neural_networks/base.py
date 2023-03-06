import torch
import abc

class NNBase(torch.nn.Module):

    '''
    Model base class
    '''

    def __init__(self, cuda=False):
        super(NNBase, self).__init__()

        # define if model is on cuda
        if cuda and torch.cuda.is_available():
            self.cuda_enabled = True
            self._device = torch.device('cuda:0')
        else:
            self.cuda_enabled = False
            self._device = torch.device('cpu')

    @property
    def device(self):
        return self._device

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractmethod
    def set_stats(self, stats):
        pass