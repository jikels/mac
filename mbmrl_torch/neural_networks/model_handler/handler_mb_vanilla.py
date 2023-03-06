import copy
import wandb
import torch
import numpy as np

#import neural networks handler
from .handler_mlp import MLP_handler

#import neural networks trainers for online training
from ..model_trainer.mlp_trainer import MLPTrainer


class MbVanillaHandler(MLP_handler):
    
    def __init__(self, config, path_model_data, path_task_data, device):
        super(MbVanillaHandler, self).__init__(config, path_model_data, path_task_data, device)

    def training(self, tasks_in, tasks_out,track_run, training_step=None, save = True):

        self._init_offline_training(tasks_in, tasks_out)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
        self.trainer = MLPTrainer(self.model, optimizer, scheduler)
        
        if track_run is not None and training_step is None:
            wandb.watch(self.model)

        self.trainer.train(
            epochs=self.config["epochs"],
            data_in = np.squeeze(tasks_in),
            data_out = np.squeeze(tasks_out),
            batch_size = self.config["batch_size"],
            track_run = track_run)

        if save == True:
            self.model.save(self.path_model_data)

    def create_model(self):
        self._create_model()
    
    def init_model_weights(self):
        self._init_model_weights()

    def load_model(self):
        self._load_model()

    def init_online_training(self):
        self._init_online_training()
    
    def return_meta_model(self):
        return self.model

    def return_online_model(self):
        return self.online_model
