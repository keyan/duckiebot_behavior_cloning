"""
python tensorboard_to_plot.py --logdir tensorboard_run_dir/
"""
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns

ORANGE = '#ffa500'
BLUE = '#4169E1'

sns.set(style="darkgrid")
sns.set_context("paper")

def plot(params):
    log_path = params['logdir']

    acc = EventAccumulator(log_path)
    acc.Reload()

    # only support scalar now
    scalar_list = acc.Tags()['scalars']

    tag_groups = [
        ('Loss', ['Loss/train', 'Loss/validation']),
        ('Linear Accuracy', ['Accuracy/train/linear', 'Accuracy/validation/linear']),
        ('Angular Accuracy', ['Accuracy/train/angular', 'Accuracy/validation/angular']),
    ]
    # for tags in scalar_list:
    for title, tags in tag_groups:
        for tag in tags:
            x = [int(s.step) for s in acc.Scalars(tag)]
            y = [s.value for s in acc.Scalars(tag)]
            color = BLUE if 'train' in tag else ORANGE
            plt.plot(x, y, color=colors.to_rgba(color, alpha=0.9))

        plt.legend(
            title='', loc='upper left',
            labels=['Traning', 'Validation'],
        )
        plt.title(title)
        plt.xlabel('Epoch')
        if 'accuracy' in tag.lower():
            plt.ylabel('Accuracy %')
        plt.savefig(f"{params['prefix']}_{title.replace(' ', '_').lower()}.png")
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='./logdir', type=str, help='logdir to event file')
    parser.add_argument('--prefix', required=True, type=str, help='Prefix to use for saving graphics')

    args = parser.parse_args()
    params = vars(args)

    plot(params)
