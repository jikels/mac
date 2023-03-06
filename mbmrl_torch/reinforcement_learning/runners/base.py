# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import wandb
import math
import os

#import utilities to process and collect data
from ...utils import experiment_utilities
from mbmrl_torch.gym.utils.env_init import init_env

from pyvirtualdisplay import Display
try:
    _display = Display(visible=False, size=(200, 200))
    with Display() as disp:
        display_name= os.environ.get("DISPLAY")
    xvfb = True
except:
    print("It seems like xvfb is not installed or not working properly."
          ,"Setting record_video to False.")
    xvfb = False

class RunnerBase:

    def __init__(
        self,
        config,
        configs_training_tasks=None,
        configs_testing_tasks=None):
        
        print("Initializing runner base...")

        self.config = config
        
        self.experiment_name = config["exp_name"]
        self.res_dir = config["exp_resdir"]
        self.action_dim = config["action_dim"]

        self.step = 0

        self.configs_training_tasks = configs_training_tasks
        self.configs_testing_tasks = configs_testing_tasks
        
        self.count_task_config_old = 1
        self.count_task_config = 0 

        self.count = 0
        self.total_reward = 0
        
        if not xvfb:
            self.config["record_video"] = False

    def run(self):
        # start the virtual display to be able to render and record the environment
        if not self.config["record_video"]:
            display_name = None
        else:
            _display.start()
        
        for self.iteration in range(self.config["iterations"]):

            # sample new task either from training task distribution or test task distribution or do nothing
            self.state = self.sample_task()

            # initialize video recorder
            path = self.config["exp_resdir"] + "/videos/"+self.config["exp_name"] 
            self.recorder = experiment_utilities.record_video(
                env = self.env,
                path = path,
                video_name = "test",
                framerate=20.0,
                display_name=display_name)

            # run an episode
            self.run_episode()
            
            wandb.log(
                {'average_reward_per_episode': self.total_reward/(self.iteration+1)},
                step=self.step)

            # save video of current iteration
            if self.config["record_video"]:
                self.recorder.save_video()

            time.sleep(2)

            if self.config["record_video"]:
                wandb.log(
                    {"video": wandb.Video(self.res_dir + "/videos/"+self.experiment_name+"test"+".mp4")},
                    step=self.step)

            # count task config
            self.task_counter()

        if self.config["record_video"]:
            _display.stop()
    
    def task_counter(self):
        # count to select new task
        self.count_task_config_old = self.count_task_config
        self.count = self.count +1
        
        if self.count == self.config["sample_new_task_after_n_iter"]:
            self.count_task_config=self.count_task_config+1
            self.count = 0

    def sample_task(self):
        if self.configs_testing_tasks is not None:
            if self.count_task_config_old is not self.count_task_config:
                if self.count_task_config < len(self.configs_testing_tasks):
                    print("Sample task ", self.count_task_config, "from testing task distribution")
                    task = self.configs_testing_tasks[self.count_task_config]
                    #set new task
                    self.env = init_env(
                        env_name=self.config["env_name"],
                        seed=self.config["seed"])
                    state = self.env.reset()
                    self.env.set_task(task)
                else:
                    _iter = math.ceil(((self.count_task_config+1)/len(self.configs_testing_tasks))-1)
                    task_config=self.count_task_config-(_iter*len(self.configs_testing_tasks))
                    task = self.configs_testing_tasks[task_config]
                    #set new task
                    seed = self.config["seed"]+_iter
                    print(
                        "Sample task ", task_config, "from testing task distribution with seed ", seed)
                    self.env = init_env(
                        env_name=self.config["env_name"],
                        seed=seed)
                    state = self.env.reset()
                    self.env.set_task(task)
            else:
                state = self.env.reset()
        else:
            self.env = init_env(
                env_name=self.config["env_name"],
                seed=self.config["seed"]+self.iteration)
            state = self.env.reset()

        return state

    def run_episode(self):
        pass