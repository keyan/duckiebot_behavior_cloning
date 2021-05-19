import argparse
import logging

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from log_reader import Reader
from model import VelocityModel

logging.basicConfig(level=logging.INFO)

# Defaults
EPOCHS = 10
INIT_LR = 1e-3
BATCH_SIZE = 64
TRAIN_PERCENT = 0.8
MOMENTUM = 0.9


class VelDataset(Dataset):
    def __init__(self, observations, lin_velocities, ang_velocities):
        self.observations = torch.from_numpy(observations).float()
        self.lin_velocities = torch.from_numpy(lin_velocities).float()
        self.ang_velocities = torch.from_numpy(ang_velocities).float()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        sample = {
            'image': self.observations[idx],
            'lin_vel': self.lin_velocities[idx],
            'ang_vel': self.ang_velocities[idx],
        }

        return sample


def parse_args():
    parser = argparse.ArgumentParser(description='Training Setup')

    parser.add_argument(
        '--epochs', help='Set the total training epochs', default=EPOCHS
    )
    parser.add_argument(
        '--learning_rate', help='Set the initial learning rate', default=INIT_LR
    )
    parser.add_argument('--batch_size', help='Set the batch size', default=BATCH_SIZE)
    parser.add_argument(
        '--log_file', help='Set the training log file name', required=True,
    )
    parser.add_argument(
        '--split',
        help='Set the training and test split point (input the percentage of training)',
        default=TRAIN_PERCENT,
    )

    return parser.parse_args()


def get_data(log_file):
    reader = Reader(log_file)

    observation, linear, angular = reader.read()

    logging.info(
        f"""Observation Length: {len(observation)}
        Linear Length: {len(linear)}
        Angular Length: {len(angular)}"""
    )
    return (np.array(observation), np.array(linear), np.array(angular))


def train(args):
    logging.info(f'Loading Datafile {args.log_file}')

    try:
        observation, linear, angular = get_data(args.log_file)
    except Exception as e:
        logging.error(e)
        logging.error('Loading dataset failed... exiting...')
        exit(1)
    logging.info(f'Loading Datafile completed')

    (
        observation_train,
        observation_validation,
        linear_train,
        linear_validation,
        angular_train,
        angular_validation,
    ) = train_test_split(
        observation, linear, angular,
        test_size=1-args.split,
        random_state=2021,
        shuffle=True,
    )

    train_dataset = VelDataset(observation_train, linear_train, angular_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = VelDataset(observation_validation, linear_validation, angular_validation)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Training on device: {device}')

    model = VelocityModel()
    criterion_lin = nn.MSELoss()
    criterion_ang = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=MOMENTUM)

    logging.info('Starting training')
    for epoch in tqdm(range(args.epochs)):
        logging.info(f'---------- Epoch {epoch} ----------')

        training_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            inputs, target_lin, target_ang = batch['image'], batch['lin_vel'], batch['ang_vel']

            optimizer.zero_grad()
            # Loss from different loss functions can be summed, see:
            #   https://discuss.pytorch.org/t/a-model-with-multiple-outputs/10440/2
            output_lin, output_ang = model(inputs)
            loss_lin = criterion_lin(output_lin, target_lin)
            loss_ang = criterion_ang(output_ang, target_ang)
            total_loss = loss_lin + loss_ang
            total_loss.backward()
            optimizer.step()

            training_loss += total_loss.item()

        validation_loss = 0.0
        for batch in tqdm(val_loader, total=len(val_loader)):
            inputs, target_lin, target_ang = batch['image'], batch['lin_vel'], batch['ang_vel']

            with torch.no_grad():
                output_lin, output_ang = model(inputs)
                loss_lin = criterion_lin(output_lin, target_lin)
                loss_ang = criterion_ang(output_ang, target_ang)
                validation_loss += (loss_lin + loss_ang).item()

        tqdm.write(f'Training loss: {training_loss}')
        tqdm.write(f'Validation loss: {validation_loss}')

    torch.save(model.state_dict(), './maserati_plus_sim.pth')
    logging.info('Finished Training')


if __name__ == '__main__':
    args = parse_args()
    train(args)
