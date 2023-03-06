from mbmrl_torch.neural_networks.model_trainer.mlp_trainer import MLPTrainer
import torch
import numpy as np

class ENNTrainer(MLPTrainer):
    
    def __init__(
        self,
        model,
        optimizer,
        scheduler):
        
        super().__init__(
            model,
            optimizer,
            scheduler)

    def train(
        self,
        data_in,
        data_out,
        task_id,
        epochs,
        batch_size,
        track_run=None,
        train_online=False):

        self.task_id = task_id

        if batch_size is None:
            batch_size = len(data_in)
        
        if batch_size > len(data_in):
            batch_size = len(data_in)

        mini_batches = int(np.ceil(float(len(data_in))/float(batch_size)))
        
        # Make tensors to process data
        # todo: make data loader that directly batches
        if self.model.cuda_enabled:
            self.training_inputs_tensor = torch.Tensor(data_in).cuda()
            self.training_targets_tensor = torch.Tensor(data_out).cuda()
        else:
            self.training_inputs_tensor=torch.Tensor(data_in)
            self.training_targets_tensor=torch.Tensor(data_out)

        self.tasks_tensor = self.make_tasks_tensor(batch_size)

        avg_epoch_loss = self._train(
            epochs,
            mini_batches,
            batch_size,
            track_run,
            train_online)
        
        if avg_epoch_loss != None:
            return avg_epoch_loss
    
    def make_tasks_tensor(self, batch_size):
        # only one task (no meta trainer)
        if self.model.cuda_enabled:
            return torch.LongTensor([[self.task_id] for _ in range(batch_size)]).cuda()
        else:
            return torch.LongTensor([[self.task_id] for _ in range(batch_size)])

    def _train_step(self, x, y):
        self.optimizer.zero_grad()
        
        # check if batch size has changed to adjust
        # tasks tensor
        if x.size(0)<self.tasks_tensor.size(0):
            tasks_tensor = self.make_tasks_tensor(x.size(0))
        else:
            tasks_tensor = self.tasks_tensor

        y_pred = self.model(x, tasks_tensor)

        loss = self.model.loss_function(y_pred=y_pred,y=y)
        loss.backward()
        self.optimizer.step()
        return loss.item()