import abc 
from torch import optim

class TrainerBase:
    
    def __init__(
        self,
        model,
        optimizer,
        scheduler):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _train_step(self, x, y):
        pass

    def _train_epoch(self, mini_batches, batch_size):
        pass

    def _train(self, epochs, mini_batches, batch_size, track_run, train_online):
        pass