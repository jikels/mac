# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from copy import deepcopy
import time

def train_meta(
    model,
    tasks_in,
    tasks_out,
    meta_iter=1000,
    inner_iter=10,
    inner_step=1e-3,
    meta_step=1e-3,
    minibatch=32,
    track_run=None,
    schedule = False,
    save_path=None):
    
    '''
    Train a model on a set of tasks using the FOMAML algorithm.

    Args:
        model: a model object
        tasks_in: state and action inputs for each task
        tasks_out: state differences for each task
        meta_iter: number of meta-iterations
        inner_iter: number of inner iterations
        inner_step: inner step size
        meta_step: outer step size
        minibatch: batch size
        track_run: wether to track the run
        schedule: wether to use a schedule for the meta step size
        save_path: path to save the model
    
    Returns:
        model: the trained meta model
    '''

    tasks_in_tensor = [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in tasks_in]
    tasks_out_tensor= [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in tasks_out]
    
    task_losses = np.zeros(len(tasks_in))

    #init data sampler
    d = Datasampler(tasks_in_tensor, tasks_out_tensor, minibatch)

    # init inner optimizer
    inner_opt = torch.optim.SGD(model.parameters(), lr=inner_step, momentum=0.9)

    for meta_count in range(meta_iter):
        start_time = time.time()

        # define current task index
        task_index = int(meta_count % len(tasks_in))
        
        # sample task batch
        x, y = d.sample_task_batch(task_index)

        # inner step
        # train on task batch
        _loss = 0.0
        inner_loss = 0.0
        weights_before = deepcopy(model.state_dict())
        model.train(mode=True)
        for _ in range(inner_iter):
            model.zero_grad()
            y_pred = model(x)
            loss = model.loss_function(y=y,y_pred=y_pred)
            loss.backward()
            inner_opt.step()
            _loss+=loss.item()
        model.train(mode=False) 
        inner_loss += (_loss / inner_iter)
        
        # update task losses
        task_losses[task_index] = inner_loss/inner_iter
        
        # outer step
        # adjust model weights with second order gradient approximation
        weights_after = model.state_dict()
        if schedule == True:
            stepsize = meta_step * (1 - meta_count / meta_iter) # linear schedule
        else:
            stepsize = meta_step
        model.load_state_dict({name : weights_before[name] + (weights_after[name] - weights_before[name]) * stepsize for name in weights_before})

        # compute mean loss across tasks
        task_losses_all = np.mean(task_losses)

        # check if loss exploded
        if np.isnan(task_losses).any(): 
            print("some task losses are not a number (nan). Logging '100' as loss and quit")
            if track_run is not None:
                log = {}
                log['mean_loss_across_tasks'] = np.nanmean(task_losses)*2
                track_run.log(log)
                quit()

        # log
        iter_time = time.time() - start_time
        if track_run is not None:
            log = {}
            keys = range(len(tasks_in))
            for i in keys:
                log["loss task"+str(i)] = task_losses[i]
            log['outer_lr'] = stepsize
            log['iter_time'] = iter_time
            log['meta_epoch'] = meta_count
            log['mean_loss_across_tasks'] = task_losses_all
            track_run.log(log)
        else:
            print("Iter " + str(meta_count) + " | Mean task Losses: " + str(task_losses.tolist()))

        if save_path is not None and (
            meta_count % 100 == 0 or meta_count == meta_iter - 1):
            model.save(save_path)
            print("Saved model to: " + save_path)

class Datasampler:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.num_task = len(x)-1
        self.bs = batch_size
        self.max_i = len(x[0])//batch_size
        self.i=0
        self.i_t=0
        self.shuffle()

    def sample_task_batch(self, task_index):
        x = self.x[task_index][self.i*self.bs : self.i*self.bs+self.bs]
        y = self.y[task_index][self.i*self.bs : self.i*self.bs+self.bs]
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