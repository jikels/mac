import copy
import wandb
import torch

#import neural networks handler
from .handler_mlp import MLP_handler

#import neural networks trainers for online training
from ..model_trainer.mlp_trainer import MLPTrainer


#import meta learning algorithm
from ...meta_learning.algorithms.MAML import train_meta as train_meta_MAML

class MamlHandler(MLP_handler):

    def __init__(self, config, path_model_data, path_task_data, device):
        super(MamlHandler, self).__init__(config, path_model_data, path_task_data, device)

    def training(self, tasks_in, tasks_out,track_run, training_step=None, save = True):

        self._init_offline_training(tasks_in, tasks_out)
        
        if track_run is not None and training_step is None:
            wandb.watch(self.model)

        train_meta_MAML(
            self.model,
            tasks_in,
            tasks_out,
            meta_epochs=self.config["meta_iter"],
            inner_iter=self.config["inner_iter"],
            inner_step=self.config["inner_step"],
            outer_step=self.config["meta_step"],
            batch_size=self.config["meta_batch_size"],
            inner_sample_size=self.config["inner_sample_size"],
            track_run=track_run,
            lr_schedule = self.config["lr_schedule"],
            training_log_step=training_step,
            meta_train_test_split=self.config["meta_train_test_split"],
            save_path=self.path_model_data if save else None)
    
    def create_model(self):
        self._create_model()

    def init_online_training(self):
        self._init_online_training()

    def train_model_online(self,train_in, train_out):
        return self._train_model_online(train_in, train_out)
    
    def init_model_weights(self):
        self._init_model_weights()

    def load_model(self):
        self._load_model()
    
    def return_meta_model(self):
        return self.model

    def return_online_model(self):
        return self.online_model
