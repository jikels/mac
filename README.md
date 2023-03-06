
# Robotic Control Using Model Based Meta Adaption
This is the official code for the paper ["Robotic Control Using Model Based Meta Adaption"](https://arxiv.org/pdf/2210.03539.pdf) accepted to [ICRA 2023](https://www.icra2023.org). The code allows users to reproduce the results reported in the paper. Detailed results can be found in [this report](https://api.wandb.ai/links/joelikels/lghjks2l).

![Alt Text](meta_testing.gif)

# Installation
1. Create and activate new environment 
```bash
conda create --name mac python=3.9
```
```bash
conda activate mac
```

2. navigate to this directory and install the required packages
```bash
cd mac
```
```bash
pip install -r requirements.txt
```

3. install mbmrl_torch as a package
```bash
cd mac
```
```bash
pip install -e .
```

4. OPTIONAL: If you are using a computer with Apple silicon (e.g. Apple M1) follow these instructions https://github.com/openai/mujoco-py/issues/682 to install mujoco while using the following script:
```bash
sh install-mujoco.sh
```

## How to test MAC?

### Configuration
1. Go to mbmrl_torch/configurations/conf/famle/config_famle.yaml and define "wandb_entity" by inserting your username for https://wandb.ai
2. Go to mbmrl_torch/configurations/conf/fomaml/config_fomaml.yaml and define "wandb_entity" by inserting your username for https://wandb.ai
2. Optional: Adust config as received through hyperparameter search (e.g. in famle/stored_configs=test_mac_ant_cripple).

### Execution
#### Ant Disabled
```bash
python3 mbmrl_torch/launcher/run.py +famle/stored_configs=test_mac_ant_cripple
```

#### Ant Gravity
```bash
python3 mbmrl_torch/launcher/run.py +famle/stored_configs=test_mac_ant_gravity
```

#### Halfcheetah Blocks
```bash
python3 mbmrl_torch/launcher/run.py +famle/stored_configs=test_mac_halfcheetah_blocks
```

#### Halfcheetah Disabled
```bash
python3 mbmrl_torch/launcher/run.py +famle/stored_configs=test_mac_halfcheetah_cripple
```

# Reproducibility of Results
Please note that although the number of sources of nondeterministic behavior is limited, experiment results vary across different hardware configurations (see: https://pytorch.org/docs/stable/notes/randomness.html, https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047). Therefore, to reproduce the papers results exactly, a hyperparameter search must be executed for each algorithm-env-configuration.

## How to Make a Hyperparameter Search?
#### Ant Disabled
```bash
wandb sweep mbmrl_torch/configurations/sweep_test_mac_ant_cripple.yaml
```

#### Ant Gravity
```bash
wandb sweep mbmrl_torch/configurations/sweep_test_mac_ant_gravity.yaml
```

#### Halfcheetah Blocks
```bash
wandb sweep mbmrl_torch/configurations/sweep_test_mac_halfcheetah_blocks.yaml
```

#### Halfcheetah Disabled
```bash
wandb sweep mbmrl_torch/configurations/sweep_test_mac_halfcheetah_cripple.yaml
```

# Video with task description
https://user-images.githubusercontent.com/74645937/221852470-1378748d-c373-4668-afea-70352ea76193.mp4





