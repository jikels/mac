'''
The code was adapted from: https://github.com/facebookresearch/higher/blob/main/examples/maml-omniglot.py

Copyright (c) Facebook, Inc. and its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
'''

from cmath import nan
import torch
from torch import optim
import numpy as np
import time
import numpy as np
import higher

def train_meta(
    model,
    tasks_in,
    tasks_out,
    meta_epochs=100,
    inner_iter=10,
    inner_step=1e-3,
    outer_step=1e-3,
    batch_size=32,
    meta_train_test_split = 0.9,
    inner_sample_size = None,
    track_run = None,
    lr_schedule = True,
    shuffle_data = True,
    training_log_step= None,
    save_path=None):

    tasks_in_tensor = [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in tasks_in]
    tasks_out_tensor= [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in tasks_out]

    #init data sampler
    d = Datasampler(tasks_in_tensor, tasks_out_tensor, inner_sample_size)

    #for status
    task_losses = np.zeros(len(tasks_in))

    # init meta optimizer
    meta_opt = optim.Adam(model.parameters(), lr=outer_step)

    # init lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt,T_max=meta_epochs,eta_min=inner_step)

    # init inner optimizer
    inner_opt = torch.optim.SGD(model.parameters(), lr=inner_step, momentum=0.9)

    for epoch in range(meta_epochs):

        # define some params
        start_time = time.time()
        outer_losses = []
        model.train(mode=True)
        meta_opt.zero_grad()

        # compute task specific parameters and backpropagation
        # thorugh task specific optimization procedure for every tasks before making the meta update
        for task_index in range(len(tasks_in)):
            
            # sample task batch
            x, y = d.sample_task_batch(task_index)

            #split data 
            size_x=x.size()
            size_y=y.size()
            size_x = int(round(size_x[0] * meta_train_test_split))
            size_y = int(round(size_y[0] * meta_train_test_split))

            x_inner=x[:size_x]
            x_outer=x[size_x:]
            y_inner=y[:size_x]
            y_outer=y[size_x:]
            
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):

                #compute task specific parameters in several inner iterations
                for _ in range(inner_iter):
                    pred_x_inner = fnet(x_inner)
                    inner_loss = model.loss_function(y_pred=pred_x_inner,y=y_inner)
                    diffopt.step(inner_loss)

                #compute task specific loss of task agnostic weights
                pred_x_outer = fnet(x_outer)
                outer_loss = model.loss_function(y_pred=pred_x_outer,y=y_outer)

                #track outer losses
                outer_losses.append(outer_loss.detach())

                #backpropagate through inner optimization procedure
                outer_loss.backward(retain_graph=True)
                
        # update outer level - meta update
        meta_opt.step()
        model.train(mode=False)

        # transform losses
        stack_task_losses = torch.stack(outer_losses)
        array_task_losses = stack_task_losses.cpu().detach().numpy()
        for i in range(len(tasks_in)):
            task_losses[i] = array_task_losses[i]
        
        task_losses_all = np.mean(task_losses)

        # step scheduler
        scheduler.step()

        iter_time = time.time() - start_time

        if np.isnan(task_losses).any(): 
            print("some task losses are not a number (nan). Logging '100' as loss and quit")
            if track_run is not None:
                log = {}
                log['mean_loss_across_tasks'] = np.nanmean(task_losses)*2
                track_run.log(log)
                quit()

        print(f'[Epoch {epoch:.2f}| Time: {iter_time:.2f}]')
        print(f'Mean loss across tasks: {task_losses_all}')

        if track_run is not None:
            log = {}
            keys = range(len(tasks_in))
            for i in keys:
                log["loss task"+str(i)] = task_losses[i]
            log['iter_time'] = iter_time
            log['meta_epoch'] = epoch
            log['mean_loss_across_tasks'] = task_losses_all
            if training_log_step is not None:
                log["training_step"]=training_log_step
                training_log_step = training_log_step+1
            track_run.log(log)
        
        if save_path is not None and (
            epoch % 100 == 0 or epoch == meta_epochs - 1):
            model.save(save_path)
            print("Saved model to: " + save_path)

class Datasampler:
    def __init__(self, x, y, inner_sample_size):
        self.x = x
        self.y = y
        self.num_task = len(x)-1
        self.iss = inner_sample_size
        self.max_i = len(x[0])//inner_sample_size
        self.i=0
        self.i_t=0
        self.shuffle()

    def sample_task_batch(self, task_index):
        x = self.x[task_index][self.i*self.iss : self.i*self.iss+self.iss]
        y = self.y[task_index][self.i*self.iss : self.i*self.iss+self.iss]
        self.count_t()
        return x, y

    def shuffle(self):
        for i in range(len(self.x)):
            self.permutation = np.random.permutation(self.x[i].size(0))
            self.x[i] = self.x[i][self.permutation]
            self.y[i] = self.y[i][self.permutation]

    def count_t(self):
        if self.i_t + 1 == self.num_task:
            self.i_t = 0
            self.count_i()
        else:
            self.i_t = self.i_t + 1
    
    def count_i(self):
        if self.i+1 == self.max_i:
            self.i = 0
            self.shuffle()
        else:
            self.i+=1
