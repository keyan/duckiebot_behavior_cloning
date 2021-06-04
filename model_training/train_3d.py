import argparse
import logging
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from log_reader import Reader
from model3d import Model3D

logging.basicConfig(level=logging.INFO)

# Defaults
EPOCHS = 10
INIT_LR = 1e-3
BATCH_SIZE = 2
TRAIN_PERCENT = 0.9
MOMENTUM = 0.9
DEFAULT_SAVE_NAME = 'saved_model'
# How close a prediction needs to be to the true label
# in order to be considered accurate.
PCT_CLOSE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description='Training Setup')

    parser.add_argument(
        '--epochs', help='Set the total training epochs', default=EPOCHS, type=int,
    )
    parser.add_argument(
        '--learning_rate', help='Set the initial learning rate', default=INIT_LR, type=float,
    )
    parser.add_argument('--batch_size', help='Set the batch size', default=BATCH_SIZE, type=int)
    parser.add_argument(
        '--log_file', help='Set the training log file name', required=True, nargs="+",
    )
    parser.add_argument(
        '--save_name', help='Set the saved model file name', default=DEFAULT_SAVE_NAME,
    )
    parser.add_argument(
        '--split',
        help='Set the training and test split point (input the percentage of training)',
        default=TRAIN_PERCENT, type=float,
    )
    # parser.add_argument(
    #     '--model',
    #     help='Specify the model to use: [v0, v1]',
    #     default='v0', type=str,
    # )
    # parser.add_argument(
    #     '--using_colab',
    #     help='Uses different settings to account for fewer system resources',
    #     action='store_true', default=False,
    # )
    parser.add_argument(
        '--save_checkpoint',
        help='Store a full checkpoint including optimizer state_dict, allows for fine tuning or resuming training',
        action='store_true', default=False,
    )
    parser.add_argument(
        '--from_checkpoint',
        help='If provided, the given path is used to load a checkpoint to resume training from',
        default='', type=str,
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
        observation_validation = {}
        linear_validation = {}
        angular_validation = {}

        observation_train = {}
        linear_train = {}
        angular_train = {}
        for log_file in args.log_file[1:]:
            observation_train[log_file], linear_train[log_file], angular_train[log_file] = get_data(log_file)

        observation_validation[log_file], linear_validation[log_file], angular_validation[log_file] = get_data(args.log_file[0])
    except Exception as e:
        logging.error(e)
        logging.error('Loading dataset failed... exiting...')
        exit(1)
    logging.info(f'Loading Datafile completed')

    # (
    #     observation_train,
    #     observation_validation,
    #     linear_train,
    #     linear_validation,
    #     angular_train,
    #     angular_validation,
    # ) = train_test_split(
    #     observation, linear, angular,
    #     test_size=1-args.split,
    #     random_state=2021,
    #     shuffle=True,
    # )

    train_dataset = VelDataset(observation_train, linear_train, angular_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VelDataset(observation_validation, linear_validation, angular_validation)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')
    logging.info(f'Training on device: {device}')

    # if args.model == 'v1':
    #     model = Modelv1()
    # else:
    #     model = Modelv0()
    model = Model3D()

    model.to(device)
    criterion_lin = nn.MSELoss()
    criterion_ang = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=MOMENTUM)

    if args.from_checkpoint:
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # For logging to Tensorboard.
    tb_writer = SummaryWriter()

    logging.info('Starting training')
    for epoch in tqdm(range(args.epochs)):
        logging.info(f'---------- Epoch {epoch} ----------')

        ########################################################################
        # Training
        ########################################################################
        model.train()
        training_loss = 0.0
        training_lin_loss = 0.0
        training_ang_loss = 0.0
        num_correct_pred_lin = torch.tensor(0).to(device)
        num_correct_pred_ang = torch.tensor(0).to(device)
        for i, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            inputs = batch['image'].to(device)
            target_lin = batch['lin_vel'].to(device)
            target_ang = batch['ang_vel'].to(device)

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
            training_lin_loss += loss_lin.item()
            training_ang_loss += loss_ang.item()
            # Keep track of how many predictions are within PCT_CLOSE of label.
            num_correct_pred_lin += torch.sum(
                torch.abs(output_lin - target_lin) < torch.abs(PCT_CLOSE * target_lin)
            )
            num_correct_pred_ang += torch.sum(
                torch.abs(output_ang - target_ang) < torch.abs(PCT_CLOSE * target_ang)
            )

        training_loss /= len(train_loader)
        training_lin_loss /= len(train_loader)
        training_ang_loss /= len(train_loader)
        num_correct_pred_lin = num_correct_pred_lin.to(cpu_device)
        training_accuracy_lin_pct = (num_correct_pred_lin.item() / len(train_dataset)) * 100
        num_correct_pred_ang = num_correct_pred_ang.to(cpu_device)
        training_accuracy_ang_pct = (num_correct_pred_ang.item() / len(train_dataset)) * 100

        ########################################################################
        # Validation
        ########################################################################
        model.eval()
        validation_loss = 0.0
        validation_lin_loss = 0.0
        validation_ang_loss = 0.0
        num_correct_pred_lin = torch.tensor(0).to(device)
        num_correct_pred_ang = torch.tensor(0).to(device)
        for batch in tqdm(val_loader, total=len(val_loader)):
            inputs = batch['image'].to(device)
            target_lin = batch['lin_vel'].to(device)
            target_ang = batch['ang_vel'].to(device)

            with torch.no_grad():
                output_lin, output_ang = model(inputs)
                loss_lin = criterion_lin(output_lin, target_lin)
                loss_ang = criterion_ang(output_ang, target_ang)

                validation_loss += (loss_lin + loss_ang).item()
                validation_lin_loss += loss_lin.item()
                validation_ang_loss += loss_ang.item()
                # Keep track of how many predictions are within PCT_CLOSE of label.
                num_correct_pred_lin += torch.sum(
                    torch.abs(output_lin - target_lin) < torch.abs(PCT_CLOSE * target_lin)
                )
                num_correct_pred_ang += torch.sum(
                    torch.abs(output_ang - target_ang) < torch.abs(PCT_CLOSE * target_ang)
                )

        validation_loss /= len(val_loader)
        validation_lin_loss /= len(val_loader)
        validation_ang_loss /= len(val_loader)
        num_correct_pred_lin = num_correct_pred_lin.to(cpu_device)
        validation_accuracy_lin_pct = (num_correct_pred_lin.item() / len(val_dataset)) * 100
        num_correct_pred_ang = num_correct_pred_ang.to(cpu_device)
        validation_accuracy_ang_pct = (num_correct_pred_ang.item() / len(val_dataset)) * 100

        ########################################################################
        # Logging
        ########################################################################
        tqdm.write(f'Training loss: {training_loss}')
        tqdm.write(f'Validation loss: {validation_loss}')
        tqdm.write(f'Training Lin. loss: {training_lin_loss}')
        tqdm.write(f'Validation Lin. loss: {validation_lin_loss}')
        tqdm.write(f'Training Ang. loss: {training_ang_loss}')
        tqdm.write(f'Validation Ang. loss: {validation_ang_loss}')
        tqdm.write(f'Training acc linear vel: {training_accuracy_lin_pct:.2f}%')
        tqdm.write(f'Training acc ang vel: {training_accuracy_ang_pct:.2f}%')
        tqdm.write(f'Validation acc linear vel: {validation_accuracy_lin_pct:.2f}%')
        tqdm.write(f'Validation acc ang vel: {validation_accuracy_ang_pct:.2f}%')

        # Tensorboard
        tb_writer.add_scalar('Loss/train', training_loss, epoch)
        tb_writer.add_scalar('Loss/validation', validation_loss, epoch)
        tb_writer.add_scalar('Accuracy/train/linear', training_accuracy_lin_pct, epoch)
        tb_writer.add_scalar('Accuracy/train/angular', training_accuracy_ang_pct, epoch)
        tb_writer.add_scalar('Accuracy/validation/linear', validation_accuracy_lin_pct, epoch)
        tb_writer.add_scalar('Accuracy/validation/angular', validation_accuracy_ang_pct, epoch)

    if args.save_checkpoint:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'./checkpoint_{args.save_name}.pth')
    else:
        torch.save(model.state_dict(), f'./{args.save_name}.pth')
    logging.info('Finished Training')


if __name__ == '__main__':
    args = parse_args()

    # Potential (though not necessary probably) TODO:
    # On Colab there isn't enough RAM to convert the torch tensors to floats because
    # this reallocates memory. So here we convert to torch tensors but only convert
    # to floats get getting items.
    # When we have enough RAM, training is faster when we do the float
    # conversion just once when loading the dataset.
    class VelDataset(Dataset):
        SENTENCE_LENGTH = 128
        SENTENCE_STEP = 10

        def __init__(self, observations, lin_velocities, ang_velocities):
            num_examples = 0
            shape = None
            for log in observations.keys():

                # Pad first image for "starting position"
                obs = observations[log]
                obs = np.concatenate((np.ones((self.SENTENCE_LENGTH - 1, *obs[0].shape), dtype=obs.dtype), obs))
                observations[log] = obs

                num_examples += max(0, 1 + ((obs.shape[0] - self.SENTENCE_LENGTH) // self.SENTENCE_STEP))
                if shape is None:
                    shape = obs.shape[1:]

            idx = 0
            ang_vel = torch.empty((num_examples))
            for log in observations.keys():
                ang = torch.from_numpy(ang_velocities[log]).float()
                for start in range(0, obs.shape[0] - self.SENTENCE_LENGTH, self.SENTENCE_STEP):
                    ang_vel[idx] = ang[start]
                    idx += 1

            ang_idxs = torch.abs(ang_vel) > 0.25

            num_ang_idxs = torch.sum(ang_idxs)
            not_idxs = np.argwhere(torch.logical_not(ang_idxs))
            not_idxs = torch.randperm(len(not_idxs))[:num_ang_idxs]
            ang_idxs[not_idxs] = True

            self.observations = torch.empty((num_examples, self.SENTENCE_LENGTH, *shape))
            self.lin_velocities = torch.empty((num_examples))
            self.ang_velocities = torch.empty((num_examples))

            idx = 0
            placement_idx = 0
            for log in observations.keys():
                obs = torch.from_numpy(observations[log]).float()
                lin = torch.from_numpy(lin_velocities[log]).float()
                ang = torch.from_numpy(ang_velocities[log]).float()
                for start in range(0, obs.shape[0] - self.SENTENCE_LENGTH, self.SENTENCE_STEP):
                    if not ang_idxs[idx]:
                        idx += 1
                        continue
                    self.observations[placement_idx] = obs[start:start + self.SENTENCE_LENGTH]

                    # These are start rather than start + self.SENTENCE_LENGTH - 1, because of padding earlier
                    self.lin_velocities[placement_idx] = lin[start]
                    self.ang_velocities[placement_idx] = ang[start]
                    placement_idx += 1
                    idx += 1

            self.observations = self.observations[:placement_idx]
            self.lin_velocities = self.lin_velocities[:placement_idx]
            self.ang_velocities = self.ang_velocities[:placement_idx]

        def __len__(self):
            return len(self.observations)

        def __getitem__(self, idx):
            sample = {
                'image': self.observations[idx],
                'lin_vel': self.lin_velocities[idx],
                'ang_vel': self.ang_velocities[idx],
            }

            return sample

    train(args)
