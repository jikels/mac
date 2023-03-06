# Copyright (c) 2023 Joel Ikels
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import random
from datetime import datetime
import os
import cv2
#import pyscreenshot as ImageGrab
from PIL import Image, ImageGrab
import pandas as pd
import mbmrl_torch

#import utilities to process and collect data
from ..utils import data_collection
from ..utils import data_processing

from stable_baselines3.sac.sac import SAC #to better understand the algorithm

def init_torch_cuda_set_device(seed=42,cuda=True,device_number=0):
    cuda = torch.cuda.is_available()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = 'cpu'
    if cuda:
        torch.cuda.manual_seed(seed)
        device_name = 'cuda:'+str(device_number)
    device = torch.device(device_name)
    return device

class record_video():

    def __init__(self, env, path, video_name, framerate,display_name):
        frame = cv2.cvtColor(np.array(ImageGrab.grab(xdisplay=display_name)), cv2.COLOR_BGR2RGB)
        height, width, depth = frame.shape

        self.env = env
        self.path = path
        self.video_name = video_name
        self.framerate = 1/self.env.dt
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(self.path+self.video_name+".mp4", self.fourcc, self.framerate, (self.width,self.height))
        self.display_name = display_name
        self.frame = None

    def render_screenshot(self):
        self.env.render()
        #im = ImageGrab.grab()
        self.frame = cv2.cvtColor(np.array(ImageGrab.grab(xdisplay=self.display_name)), cv2.COLOR_BGR2RGB)
        #cv2.waitKey(50)
        return self.frame

    def capture(self):
        frame = self.render_screenshot()
        self.video.write(frame)
    
    def get_last_frame(self):
        return self.frame

    def save_last_frame(self,name):
        cv2.imwrite(filename=self.path+name+'.jpg',img=self.frame)
        
    def save_video(self):
        cv2.destroyAllWindows()
        self.video.release()
    
def init_experiment(config):
    now = datetime.now()
    experiment_name = now.strftime("%d_%m_%Y_%H_%M_%S")
    # get absolute path to project data dir 
    config_data_dir = os.path.dirname(os.path.abspath(mbmrl_torch.__file__))
    res_dir = os.path.join(config_data_dir,"experiments",config["env_name"], experiment_name)
    try:
        i = 0
        while True:
            res_dir += "_" + str(i)
            i += 1
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)
                os.makedirs(res_dir+"/videos")
                break
    except:
        print("Could not make the result directory!!!")

    with open(res_dir + "/details.txt", "w+") as f:
        f.write(config["exp_details"])
        
    return experiment_name, res_dir

def data_gen_load(env,path_task_data,config):
    # test if task data exists
    if os.path.exists(path_task_data):
        print('loading existing task configs')
        configs_testing_tasks= np.load(
            path_task_data+"/config_testing_tasks.npy",
            allow_pickle=True)
        configs_training_tasks=np.load(
            path_task_data+"/config_training_tasks.npy",
        allow_pickle=True)
    else:
        # test if presets should be loaded
        if config['load_task_presets']:
            print('loading preset task configs')
            # get absolute path to project data dir 
            _dir = os.path.join(
                os.path.abspath(
                    mbmrl_torch.__file__)[:-11],
                    'configurations/task_config')
            configs_testing_tasks= np.load(
                _dir+"/test_"+config["env_name"]+".npy",
                allow_pickle=True)
            configs_training_tasks=np.load(
                _dir+"/train_"+config["env_name"]+".npy",
            allow_pickle=True)
        else:
            #create and train test split tasks if there is not preset config
            print('making task configs')
            configs_training_tasks,configs_testing_tasks=data_collection.generate_train_test_tasks(
                env=env,n_tasks =config["n_tasks_distribution"],
                meta_train_test_split=0.8)

        # save tast task configuration
        # to task data directory
        print('saving task configs')
        os.makedirs(path_task_data)
        np.save(
            path_task_data+"/config_testing_tasks.npy",
            configs_testing_tasks, allow_pickle=True)
        np.save(
            path_task_data+"/config_training_tasks.npy",
            configs_training_tasks, allow_pickle=True)

        # choose policy to sample task data
        # todo: make base policy class for more policies
        if config['collection_policy'] is not None:
            collection_policy = SAC.load(config['collection_policy'], env=env)
        else:
            collection_policy = None

        #collect data and save in directory for one task
        #randomly collect data or collect data according to policy
        print('creating task data')
        data_collection.generate_training_data(rollouts=config["rollouts"],
                                                episode_length=config["episode_length"],
                                                env=env,
                                                configs_training_tasks=configs_training_tasks,
                                                path=path_task_data,
                                                done_reset = config["reset_env_when_done"],
                                                policy = collection_policy)
    return configs_testing_tasks, configs_training_tasks

def save_plot(res_dir, results,name,configs_testing_tasks,xlabel,ylabel,title):
    data = pd.DataFrame(results)

    #check if list contains other lists to prepare datafram accordingly
    #x = any(isinstance(el, list) for el in data)
    #if x is True:
    data = data.transpose()
    

    #if config["iterations"] > len(configs_testing_tasks):
    count=1
    columns = len(data.columns)
    column_names =[]
    for i in range(columns):
        column_name = 'task_'+str(count)+"_iter_"+str(i+1)
        data.rename(columns={data.columns[i]: column_name},inplace=True)
        column_names.append(column_name)
        i=i+1

        if count < len(configs_testing_tasks):
            count = count +1
        else:
            count=0

    data.astype('int32').dtypes
    plot = data.plot(y=column_names, use_index=True,title=title,xlabel=xlabel,ylabel=ylabel)
    fig = plot.get_figure()
    fig.savefig(res_dir+"/results_"+name+"_.png")
    



