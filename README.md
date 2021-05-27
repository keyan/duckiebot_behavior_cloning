# duckiebot_behavior_cloning

Code for training a CNN to learn end-to-end driving controls through behavior cloning using simulated and real robot execution logs.

![aido](https://user-images.githubusercontent.com/6401746/118901520-58570500-b8c8-11eb-9bff-6795ee6cd470.gif)

## Repo Structure
```
├── aido_submission         # Complete AIDO submissions, one per subdir
├── data_converter          # Scripts to aid in data collection/cleaning
├── fetch_training_data.sh  # Script to fetch the current largest training dataset
├── log_utils               # Scripts to aid in visualizing dataset frames and merging datasets
└── model_training          # Pytorch CNN models and training code
```

## Setup/Usage

Because the repo is split up into different directories that handle different tasks, each task/directory has different dependencies. See the README for each directory or the sections below for more detail.

For most tasks you will need Docker installed. For viewing dataset log files you need to use the conda environment:
```
conda env create -f environment.yml
conda activate behavior_clone
```

## Datasets generation

These are the datasets currently fetched by the `fetch_training_data.sh` script:
1. `sim_small_plus_tori.log`
    - very small, mix of some real robot and simulated robot logs, meant for local dev only
1. `simulated_plus_maserati.log`
    - the current best dataset, about 9000 frames total of simulated and real logs
1. `maserati_bill_simulated_amadobot_base`
    - the biggest dataset currently, has all the usable simulated and real robot data (~5GB)

### Simulated

Using pure pursuit we can generate control/image data to use for model training using gym-duckietown, this notebook implements this and was used to generate the `simulated.log` dataset:
https://colab.research.google.com/drive/1Heh65KRqc6HhEyyYOlqNmORvKRWrE8Es?usp=sharing

To generate data use this command in [this directory](https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning/tree/master/duckieSchool/duckieGym) of the template repo:
```
python automatic.py --domain-rand True --nb-episodes 10 --steps 1000 --downscale
```

### Real robot

The `data_converter` directory contains scripts to process ROS bags into synchronized training data logs. There is an additional README in that directory that explains usage in more detail. Real robot logs were extracted from public logs hosted on http://logs.duckietown.org/.

## Model training

```
./fetch_training_data.sh

# Trains the default dataset downloaded in the prior command.
python train.py --log_file simulated_plus_maserati.log --epochs 100
```

In another session you can start Tensorboard and monitor training:
```
tensorboard --logdir model_training/runs/
```

This is the command for training the current best model (on Colab):
```
python model_training/train.py --model v1 --save_name modelv1 --split 0.9 --epochs 100 --batch_size 64 --log_file maserati_bill_simulated_amadobot_base.log --using_colab
```

## Evaluation

Currently all evaluation is being done using the AIDO submission format and performing local or remote evaluation against the simulator. More details are in the README in `/aido_submissions`.

## Controls Background

The following ROS topics are used to create a synchronized dataset of images to controls used:
```
# Compressed camera output
/camera_node/image/compressed

# Joystick configuration message, see https://docs.ros.org/en/api/sensor_msgs/html/msg/Joy.html
/joy
```

The axes values from the joystick configuration are used to determine linear and angular velocity controls for the robot, which we can then convert to wheel velocities using a hand tuned wrapper, see https://git.io/Js2m9.
