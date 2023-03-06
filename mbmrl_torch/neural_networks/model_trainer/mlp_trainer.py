from mbmrl_torch.neural_networks.model_trainer.base import TrainerBase
import torch
import numpy as np
import time

class MLPTrainer(TrainerBase):
    
    def __init__(
        self,
        model,
        optimizer,
        scheduler):

        super().__init__(model, optimizer, scheduler)
    
    def train(
        self,
        data_in,
        data_out,
        epochs,
        batch_size,
        track_run=None,
        train_online=False):
        
        # Make tensors to process data
        # todo: make data loader that directly batches
        if self.model.cuda_enabled:
            self.training_inputs_tensor = torch.Tensor(data_in).cuda()
            self.training_targets_tensor = torch.Tensor(data_out).cuda()
        else:
            self.training_inputs_tensor=torch.Tensor(data_in)
            self.training_targets_tensor=torch.Tensor(data_out)
        
        if batch_size is None:
            batch_size = len(data_in)
        
        if batch_size > len(data_in):
            batch_size = len(data_in)

        mini_batches = int(np.ceil(float(len(data_in))/float(batch_size)))

        avg_epoch_loss = self._train(
            epochs,
            mini_batches,
            batch_size,
            track_run,
            train_online)
        
        if avg_epoch_loss != None:
            return avg_epoch_loss
    
    def _train(self, epochs, mini_batches, batch_size, track_run, train_online):
        epoch_losses = []
        self.model.train(mode=True)
        for epoch in range(epochs):
            t1 = time.time()
            epoch_loss = self._train_epoch(mini_batches, batch_size)
            
            # collect epoch losses during online training
            # log epoch losses during offline training
            if train_online:
                epoch_losses.append(epoch_loss)
            else:
                self._log_offline_training(
                    epoch_loss,
                    time.time()-t1,
                    epoch,
                    track_run)
            
            # step scheduler
            if self.scheduler is not None:
                self.scheduler.step(epoch_loss)

        self.model.train(mode=False)

        # return mean epoch loss if model is trained online
        # else reurn None to indicate that the model is trained offline
        if train_online==True:
            return self._log_online_training(epoch_losses)
        else:
            return None
        
    def _log_offline_training(self,loss,time,epoch, track_run):
        if track_run is not None:
            log = {}
            log["mean_loss"] = loss
            log['iter_time'] = time
            log['epoch'] = epoch
            track_run.log(log)
        else:
            print("Epoch " + str(epoch) + " | Loss: " + str(loss))

    def _log_online_training(self, epoch_losses):
        if len(epoch_losses) == 1:
            mean_loss_across_epochs = epoch_losses[0]
        else:
            mean_loss_across_epochs = np.mean(epoch_losses)
        return mean_loss_across_epochs

    def _train_epoch(self, mini_batches, batch_size):
        r_loss = 0.0
        permutation = torch.randperm(self.training_inputs_tensor.size()[0])
        for i in range(mini_batches):
            x = self.training_inputs_tensor[permutation[i*batch_size : i*batch_size + batch_size]]
            y = self.training_targets_tensor[permutation[i*batch_size : i*batch_size + batch_size]]
            loss = self._train_step(x, y)
            # update running training loss
            r_loss += loss
        # divide the training loss 
        # by the number of batches
        # to get the average loss per epoch
        # see: https://discuss.pytorch.org/t/on-running-loss-and-average-loss/107890
        return r_loss/mini_batches
    
    def _train_step(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.model.loss_function(y_pred=y_pred,y=y)
        loss.backward()
        self.optimizer.step()
        # loss_item is total cost
        # averaged across all 
        # training examples of the current batch)
        return loss.item()

