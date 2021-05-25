# paper

LaTeX source for the final report as well as figures and training data collected from tensorboard.

## Figure generation

There is a script in the `./data` directory that can take as input tensorboard log data in the format expected to be written by the training scripts in `./model_training` and outputs seaborn generated png files showing training and validation performance.

To use them:
```
python tensorboard_to_plot.py --logdir <path to the tensorboard data> --prefix <same name to prefix image files>
```

It is a little brittle and requires being passed data from only one training run.
