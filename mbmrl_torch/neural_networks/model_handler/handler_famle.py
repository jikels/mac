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
import copy
import wandb
import numpy as np

from mbmrl_torch.neural_networks.model_handler.handler_enn import EnnHandler


#import meta learning algorithm
from ...meta_learning.algorithms.famle import train_meta as train_meta_FAMLE

class FamleHandler(EnnHandler):
    
    def __init__(self, config, path_model_data, path_task_data, device, configs_training_tasks):
        super(EnnHandler, self).__init__(
            config,
            path_model_data,
            path_task_data,
            device)

        self.configs_training_tasks = configs_training_tasks
        self.task_likelihoods = np.random.rand(len(configs_training_tasks))
        self.task_index = 0
        
    def create_model(self):
        self._create_model()

    def init_online_training(self):
        self._init_online_training()
    
    def init_model_weights(self):
        self._init_model_weights()

    def load_model(self):
        self._load_model()
    
    def return_meta_model(self):
        return self.models

    def return_online_model(self):
        return self.online_model

    def init_online_training(self):
        self._init_online_training()
    
    def training(self, tasks_in, tasks_out,track_run, training_step=None, save = True):
        
        self._init_offline_training(tasks_in, tasks_out)
        
        if track_run is not None and training_step is None:
            wandb.watch(self.model)

        train_meta_FAMLE(
            self.model,
            tasks_in,
            tasks_out,
            meta_iter=self.config["meta_iter"],
            inner_iter=self.config["inner_iter"],
            inner_step=self.config["inner_step"],
            meta_step=self.config["meta_step"],
            minibatch=self.config["meta_batch_size"],
            track_run=track_run,
            schedule = self.config["lr_schedule"],
            save_path=self.path_model_data if save else None)
    
    def train_model_online(self, train_in, train_out):
        
        # compute most likely task
        self.compute_task_likelihoods_and_task_index(train_in,train_out)
        # copy online model to start from scratch (k-step adaption)
        self.online_model = copy.deepcopy(self.model)
        # fix task ID to activate most likely embedding
        self.online_model.fix_task(self.task_index)
        # reset normalizer
        self.normalizer.reset_to_init()
        # calculate new statistics
        for i in range(len(train_in)):
            self.normalizer.update_stats(train_in[i], train_out[i])
        # get new statistics
        normalization_stats = self.normalizer.get_stats()
        # set statistics
        self.online_model.set_stats(normalization_stats)

        mean_loss = self.online_trainer.train(
                    data_in=train_in,
                    data_out=train_out,
                    task_id=self.task_index,
                    epochs=self.config["epoch"],
                    batch_size=self.config["minibatch_size"],
                    train_online=True)
        
        return mean_loss
    
    def compute_task_likelihoods_and_task_index(self,x,y):
        '''
        Computes MSE loss and then softmax to have a probability
        '''
        lik = np.zeros(len(self.models))
        beta = self.config["beta"]

        for i, m in enumerate(self.models):
            y_pred = m.predict(x)
            loss = m.loss_function_numpy(y=y,y_pred=y_pred)
            lik[i] = np.exp(- beta * loss)
        
        self.task_likelihoods = lik/np.sum(lik) #softmax
        
        if self.config["sample_model"]:
            '''randomly sample model'''
            cum_sum = np.cumsum(self.task_likelihoods)
            num = np.random.rand()
            for i, cum_prob in enumerate(cum_sum):
                if num <= cum_prob:
                    self.task_index = i
        else:
            self.task_index = np.argmax(self.task_likelihoods)
            #self.task_index = 0
            #print(self.task_index)
        
        self.task_likelihoods=self.task_likelihoods*0 
        self.task_likelihoods[self.task_index]=1.0