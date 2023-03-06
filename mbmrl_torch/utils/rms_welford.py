# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copyreg import pickle
import numpy as np

#normalizer class for continous tracking of normalization statistics
class RunningMeanStd(object):

    def __init__(self, shape):
        self.mean = np.zeros(shape, np.float64) + 1e-10
        self.std = np.ones(shape, np.float64)
        self.count = 0
          
    def update(self, data):
        # Implementation of the Welford online Algorithm:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
        # explanation: https://changyaochen.github.io/welford/
        self.count = self.count + 1
        mean_old = self.mean
        self.mean = mean_old + (data-mean_old)/self.count
        self.std = np.sqrt(np.square(self.std) + (((data-mean_old)*(data-self.mean)-np.square(self.std))/(self.count+1)))

    def get_statistics(self):
        statistics = [self.mean,self.std,self.count]
        return statistics

    def set_statistics(self, statistics):
        self.mean = statistics[0]
        self.std = statistics[1]
        self.count = statistics[2]
    
    def save_statistics(self,path):
        statistics = [self.mean,self.std,self.count]
        np.save(statistics,path+"/normalization_statistics.pt")
        return statistics

    def reset_to_init(self):
        self.mean = self.mean_init
        self.data_mean_output = self.data_mean_output_init
        self.data_std_input = self.data_std_input_init
        self.data_std_output = self.data_std_output_init
        self.count = self.count_init

class NNNormalizer():
    def __init__(self,state_dim, action_dim):
        self.action_dim = action_dim
        self.ones = np.ones(action_dim)
        self.zeros = np.zeros(action_dim)
        self.stats_in_state = RunningMeanStd(state_dim)
        self.stats_out_state = RunningMeanStd(state_dim)
    
    def calculate_stats(self, tasks_in, tasks_out):
        # read in data to get normalization statistics
        self.count = 0
        all_data_in = []
        all_data_out = []
        for task_id in range(len(tasks_in)):
            for i in range(len(tasks_in[task_id])):
                self.count = self.count + 1
                all_data_in.append(tasks_in[task_id][i][:-self.action_dim])
                all_data_out.append(tasks_out[task_id][i])
        
        # calculate normalization statistics
        self._data_mean_input = np.mean(all_data_in, axis=0)
        self._data_std_input = np.std(all_data_in, axis=0)+ 1e-10
        self._data_mean_output = np.mean(all_data_out, axis=0)
        self._data_std_output = np.std(all_data_out, axis=0)+ 1e-10

        # update rms classes
        self.stats_in_state.set_statistics(
            [self._data_mean_input,self._data_std_input,self.count])
        self.stats_out_state.set_statistics(
            [self._data_mean_output,self._data_std_output,self.count])

    def update_stats(self, data_in, data_out):
        # discard action dim
        data_in = data_in[:-self.action_dim]
        # update rms classes
        self.stats_in_state.update(data_in)
        self.stats_out_state.update(data_out)

    def get_stats(self):
        stats_out = self.stats_out_state.get_statistics()
        stats_in = self.stats_in_state.get_statistics()
        
        try:
            mean_in = np.concatenate([stats_in[0],self.zeros], axis=0)
            std_in = np.concatenate([stats_in[1],self.ones], axis=0)
        except:
            #mean_in = stats_in[0]
            #std_in = stats_in[1]
            print("Error in get_stats")
            quit()
        
        mean_out = stats_out[0]
        std_out = stats_out[1]
        self.count = stats_in[2]

        statistics = [
            mean_in,
            std_in,
            mean_out,
            std_out,
            self.count,
        ]
        return statistics

    def save(self, path):
        stats_out = self.stats_out_state.get_statistics()
        stats_in = self.stats_in_state.get_statistics()
        stats = np.stack([stats_out[0],stats_in[0]], axis=0)
        np.save(path + "/normalization_statistics.npy",stats)

    def load(self, path):
        stats = np.load(path, allow_pickle=True)
        self.init_stats = stats
        self.set_stats(stats)

    def reset_to_init(self):
        self.set_stats(self.init_stats)
    
    def set_stats(self,stats):
        self.stats_out_state.set_statistics(stats[0])
        self.stats_in_state.set_statistics(stats[1])
        
