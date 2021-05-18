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

### Real robot

The `data_converter` directory contains scripts to process ROS bags into synchronized training data logs. There is an additional README in that directory that explains usage in more detail. Real robot logs were extracted from public logs hosted on http://logs.duckietown.org/.

## Model training

## Evaluation
