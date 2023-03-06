# create specified config for algorithmic evaluation

import numpy as np
import mbmrl_torch
import os

# get absolute path to project data dir 
save_dir = os.path.join(
    os.path.abspath(mbmrl_torch.__file__)[:-11],'configurations/task_config')

# hc hfield
train=[{'hfield': 'gentle'}, {'hfield': None}, {'hfield': 'hfield'}, {'hfield': 'hill'}]
test =[ {'hfield': 'basin'}, {'hfield': 'steep'}]
np.save(save_dir+"/train_HalfCheetahHField-v1.npy", train, allow_pickle=True)
np.save(save_dir+"/test_HalfCheetahHField-v1.npy", test, allow_pickle=True)

# hc blocks 
train = [
    {'damping': np.array([6.0, 4.5, 3.0, 4.5, 3.0, 1.5])}, #reference task
    {'damping': np.array([6.9, 6.2, 4.2, 8.5, 7.6, 5.1])},
    {'damping': np.array([4.3, 9.4, 2.5, 8.1, 3.1, 5.6])},
    {'damping': np.array([4.2, 9.5, 8.0, 4.8, 2.5, 4.4])},
    {'damping': np.array([2.4, 2.5, 4.6, 6.0, 8.3, 7.4])},
    {'damping': np.array([5.2, 2.7, 2.9, 3.0, 7.4, 6. ])},
    {'damping': np.array([3.1, 8.7, 2.7, 3.0, 6.2, 6.2])},
    {'damping': np.array([3.9, 8.4, 7.8, 7.9, 8.1, 7.9])},
    {'damping': np.array([3.3, 5.1, 6.3, 8.4, 2.0, 5.6])}]
test = [
    {'damping': np.array([2.7, 3.0, 6.6, 4.5, 7.5, 5.3])},
    {'damping': np.array([7.2, 6.2, 9.9, 6.0, 9.2, 3.9])}]
np.save(save_dir+"/train_HalfCheetahBlocks-v1.npy", train, allow_pickle=True)
np.save(save_dir+"/test_HalfCheetahBlocks-v1.npy", test, allow_pickle=True)

# hc cripple
train=[
    {'crippled_joint': None}, #reference task
    {'crippled_joint': 0},
    {'crippled_joint': 1},
    {'crippled_joint': 2},
    {'crippled_joint': 3}]
test=[
    {'crippled_joint': 4},
    {'crippled_joint': 5}]
np.save(save_dir+"/train_HalfCheetahCripple-v1.npy", train, allow_pickle=True)
np.save(save_dir+"/test_HalfCheetahCripple-v1.npy", test, allow_pickle=True)

# ant gravity
gravities = np.array([-9.81,-6, -6.5,-7.5, -7,-8.5, -8, -9,-10.5, -11,-13,-14,-14.5,-15,-16,-16.5,-17])
train = [{'gravity': gravity} for gravity in gravities]
gravities = np.array([-3, -5, -4, -10, -12, -18, -25, -30])
test= [{'gravity': gravity} for gravity in gravities]
np.save(save_dir+"/train_AntGravity-v1.npy", train, allow_pickle=True)
np.save(save_dir+"/test_AntGravity-v1.npy", test, allow_pickle=True)

# ant cripple
train = [
    {'crippled_leg':None},
    {'crippled_leg':0},
    {'crippled_leg':1},
    {'crippled_leg':2}]

test = [{'crippled_leg':3}]

np.save(save_dir+"/train_AntCripple-v1.npy", train, allow_pickle=True)
np.save(save_dir+"/test_AntCripple-v1.npy", test, allow_pickle=True)

# ant direction
train = [
    {'direction': [1.0, 0.0]},
    {'direction': [0.0, 1.0]},
    {'direction': [0.5, 0.0]}
    ]
test = [{'direction': [0.0, 0.5]}]
np.save(save_dir+"/train_AntDirection-v1.npy", train, allow_pickle=True)
np.save(save_dir+"/test_AntDirection-v1.npy", test, allow_pickle=True)
