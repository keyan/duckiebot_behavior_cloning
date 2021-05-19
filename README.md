# duckiebot_behavior_cloning

Code for training a CNN to learn end-to-end driving controls through behavior cloning using simulated and real robot execution logs.

## Repo Structure
```
// TODO
```

## Setup/Usage

Because the repo is split up into different directories that handle different tasks, each task/directory has different dependencies. See the README for each directory or the sections below for more detail.

For most tasks you will need Docker installed. For viewing dataset log files you need to use the conda environment:
```
conda env create -f environment.yml
conda activate behavior_clone
```

## Dataset generation

The full dataset used for the final model is XGB and is hosted here: //TODO

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

## Evaluation

## Controls Background

The following ROS topics are used to create a synchronized dataset of images to controls used:
```
# Compressed camera output
/camera_node/image/compressed

# Joystick configuration message, see https://docs.ros.org/en/api/sensor_msgs/html/msg/Joy.html
/joy
```

The axes values from the joystick configuration are used to determine linear and angular velocity controls for the robot, which we can then convert to wheel velocities using a hand tuned wrapper, see https://git.io/Js2m9.
