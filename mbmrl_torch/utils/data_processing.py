# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cherry as ch
import numpy as np
import os
import torch

# make data consumable by dynamics model
def process_to_meta_training_set(training_task_directory):
    tasks_in = []
    tasks_out = []

    for file in os.listdir(training_task_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pt") and not filename.endswith(
            "normalization_statistics.pt"
        ):
            ExperienceReplay = ch.ExperienceReplay()
            ExperienceReplay.load(os.path.join(training_task_directory, filename))

            training_in = []
            training_out = []

            for transition in ExperienceReplay:
                # get state
                s = np.squeeze(transition.state.numpy(), axis=0)
                # get action
                a = np.squeeze(transition.action.numpy(), axis=0)
                # concatenate state and actions for model input x
                training_in.append(np.concatenate((s, a)))
                # get next states and subtract from state as label fpr state actions pairs
                s1 = np.squeeze(transition.next_state.numpy(), axis=0)
                training_out.append(np.subtract(s1, s))

            ExperienceReplay.empty()

            x, y, high, low = (
                np.array(training_in),
                np.array(training_out),
                np.max(training_in, axis=0),
                np.min(training_in, axis=0),
            )

            tasks_in.append(x)
            tasks_out.append(y)
        else:
            continue

    return tasks_in, tasks_out, high, low


# make data consumable by dynamics model for learning to adapt algorithm
def process_to_meta_training_set_lta(training_task_directory, m, k):
    tasks_in = []
    tasks_out = []

    for file in os.listdir(training_task_directory):
        filename = os.fsdecode(file)

        if filename.endswith(".pt") and not filename.endswith(
            "normalization_statistics.pt"
        ):
            ExperienceReplay = ch.ExperienceReplay()[-m - k :]
            ExperienceReplay.load(os.path.join(training_task_directory, filename))

            training_in = []
            training_out = []

            for transition in ExperienceReplay[-m - k : -k]:
                # get state
                s = np.squeeze(transition.state.numpy(), axis=0)
                # get action
                a = np.squeeze(transition.action.numpy(), axis=0)
                # concatenate state and actions for model input x
                training_in.append(np.concatenate((s, a)))
                # get next states and subtract from state as label fpr state actions pairs
                s1 = np.squeeze(transition.next_state.numpy(), axis=0)
                training_out.append(np.subtract(s1, s))

            for transition in ExperienceReplay[-k:]:
                # get state
                s = np.squeeze(transition.state.numpy(), axis=0)
                # get action
                a = np.squeeze(transition.action.numpy(), axis=0)
                # concatenate state and actions for model input x
                training_in.append(np.concatenate((s, a)))
                # get next states and subtract from state as label for state actions pairs
                s1 = np.squeeze(transition.next_state.numpy(), axis=0)
                training_out.append(np.subtract(s1, s))

            x, y, high, low = (
                np.array(training_in),
                np.array(training_out),
                np.max(training_in, axis=0),
                np.min(training_in, axis=0),
            )

            tasks_in.append(x)
            tasks_out.append(y)

            ExperienceReplay.empty()
        else:
            continue

    return tasks_in, tasks_out


# make data consumable by rewards model // todo: state and action dim output as above + normaization as above
def process_to_reward_model_training_set(training_task_directory, get_dim=False):
    tasks_in = []
    tasks_out = []
    for file in os.listdir(training_task_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pt"):
            ExperienceReplay = ch.ExperienceReplay()
            ExperienceReplay.load(os.path.join(training_task_directory, filename))

            training_in = []
            training_out = []

            for transition in ExperienceReplay:
                # get state
                s = np.squeeze(transition.state.numpy(), axis=0)
                # get action
                a = np.squeeze(transition.action.numpy(), axis=0)
                # concatenate state and actions for model input x
                training_in.append(np.concatenate((s, a)))
                # get and append reward as laberl for state actions pairs
                training_out.append(np.squeeze(transition.reward.numpy(), axis=0))

            ExperienceReplay.empty()

            x, y, high, low = (
                np.array(training_in),
                np.array(training_out),
                np.max(training_in, axis=0),
                np.min(training_in, axis=0),
            )

            tasks_in.append(x)
            tasks_out.append(y)

            if get_dim == True:
                break

            continue
        else:
            continue

    # return np.concatenate(training_in,axis=0), np.concatenate(training_out, axis=0)
    return tasks_in, tasks_out


def process_experience_replay(ExperienceReplay):
    training_in = []
    training_out = []
    reward = []
    for transition in ExperienceReplay:
        s = np.squeeze(transition.state.numpy(), axis=0)
        a = np.squeeze(transition.action.numpy(), axis=0)
        training_in.append(np.concatenate((s, a)))
        # get next states and subtract from state as label fpr state actions pairs
        s1 = np.squeeze(transition.next_state.numpy(), axis=0)
        training_out.append(np.subtract(s1, s))
        reward.append(np.squeeze(transition.reward.numpy(), axis=0))
    x, y, high, low, reward = (
        np.array(training_in),
        np.array(training_out),
        np.max(training_in, axis=0),
        np.min(training_in, axis=0),
        np.array(reward),
    )
    return x, y, high, low, reward


# normalizer class for continous tracking of normalization statistics
class normalizer(object):
    def __init__(self, tasks_in=None, tasks_out=None, set_statistics=False):

        if set_statistics == False:
            self.count = 0
            self.all_data_in = []
            self.all_data_out = []
            for task_id in range(len(tasks_in)):
                for i in range(len(tasks_in[task_id])):
                    self.count = self.count + 1
                    self.all_data_in.append(tasks_in[task_id][i])
                    self.all_data_out.append(tasks_out[task_id][i])

            self.data_mean_input = torch.Tensor(np.mean(self.all_data_in, axis=0))
            self.data_mean_output = torch.Tensor(np.mean(self.all_data_out, axis=0))
            self.data_std_input = torch.Tensor(np.std(self.all_data_in, axis=0)) + 1e-10
            self.data_std_output = (
                torch.Tensor(np.std(self.all_data_out, axis=0)) + 1e-10
            )

            self.data_mean_input_init = torch.Tensor(np.mean(self.all_data_in, axis=0))
            self.data_mean_output_init = torch.Tensor(
                np.mean(self.all_data_out, axis=0)
            )
            self.data_std_input_init = (
                torch.Tensor(np.std(self.all_data_in, axis=0)) + 1e-10
            )
            self.data_std_output_init = (
                torch.Tensor(np.std(self.all_data_out, axis=0)) + 1e-10
            )
        else:
            self.data_mean_input = set_statistics[0]
            self.data_mean_output = set_statistics[1]
            self.data_std_input = set_statistics[2]
            self.data_std_output = set_statistics[3]
            self.count = set_statistics[4]

            self.data_mean_input_init = set_statistics[0]
            self.data_mean_output_init = set_statistics[1]
            self.data_std_input_init = set_statistics[2]
            self.data_std_output_init = set_statistics[3]
            self.count_init = set_statistics[4]

    def track_statistics(self, data_in, data_out):
        # Implementation of the Welford online Algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
        # explanation: https://changyaochen.github.io/welford/
        x = torch.tensor(data_in)
        y = torch.tensor(data_out)

        for i in range(x.size(dim=0)):

            self.count = self.count + 1

            state_action = x[i]
            next_state = y[i]

            mean_old_input = self.data_mean_input
            mean_old_output = self.data_mean_output

            self.data_mean_input = (
                mean_old_input + (state_action - mean_old_input) / self.count
            )
            self.data_mean_output = (
                mean_old_output + (next_state - mean_old_output) / self.count
            )

            self.data_std_input = torch.sqrt(
                torch.square(self.data_std_input)
                + (
                    (
                        (state_action - mean_old_input)
                        * (state_action - self.data_mean_input)
                        - torch.square(self.data_std_input)
                    )
                    / (self.count + 1)
                )
            )
            self.data_std_output = torch.sqrt(
                torch.square(self.data_std_output)
                + (
                    (
                        (next_state - mean_old_output)
                        * (next_state - self.data_mean_output)
                        - torch.square(self.data_std_output)
                    )
                    / (self.count + 1)
                )
            )

    def get_statistics(self):
        statistics = [
            self.data_mean_input,
            self.data_mean_output,
            self.data_std_input,
            self.data_std_output,
            self.count,
        ]
        return statistics

    def save_statistics(self, path):
        statistics = [
            self.data_mean_input,
            self.data_mean_output,
            self.data_std_input,
            self.data_std_output,
            self.count,
        ]
        torch.save(statistics, path + "/normalization_statistics.pt")
        return statistics

    def reset_to_init(self):
        self.data_mean_input = self.data_mean_input_init
        self.data_mean_output = self.data_mean_output_init
        self.data_std_input = self.data_std_input_init
        self.data_std_output = self.data_std_output_init
        self.count = self.count_init
