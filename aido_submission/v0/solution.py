#!/usr/bin/env python3
import io
import os
from typing import Tuple

import torch
import cv2
import numpy as np
from PIL import Image
# Not clear if this is going to work because there is no DB21M agent present:
#   https://github.com/duckietown/aido-protocols/blob/daffy/src/aido_schemas/schemas.py
from aido_schemas import (
    Context, DB20Commands, DB20Observations,
    EpisodeStart, JPGImage, LEDSCommands,
    logger, protocol_agent_DB20, PWMCommands, RGB, wrap_direct,
)

from modelv0 import Modelv0
from helpers import image_resize, SteeringToWheelVelWrapper


class Agent:
    def __init__(self, expect_shape=(480, 640, 3)):
        self.expect_shape: Tuple[int, int, int] = expect_shape

    def init(self, context: Context):
        context.info("Check GPU...")

        self.check_gpu_available(context)

        # Model predicts linear and angular velocity but we need to issue
        # wheel velocity commands to robot, this wrapper converts to the latter.
        self.convertion_wrapper = SteeringToWheelVelWrapper()

        context.info('init()')

        model = Modelv0()
        model.load_state_dict(
            torch.load(
                'modelv0_maserati_plus_simulated.pth',
                map_location=self.device,
        ))
        model.eval()
        self.model = model.to(self.device)

        self.current_image = np.zeros(self.expect_shape)
        self.input_image = np.zeros((150, 200, 3))
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

    def check_gpu_available(self, context: Context):
        available = torch.cuda.is_available()
        req = os.environ.get('AIDO_REQUIRE_GPU', None)
        context.info(f'torch.cuda.is_available = {available!r} AIDO_REQUIRE_GPU = {req!r}')
        context.info('init()')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if available:
            i = torch.cuda.current_device()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(i)
            context.info(f'device {i} of {count}; name = {name!r}')
        else:
            if req is not None:
                msg = 'No GPU found'
                context.error(msg)

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20Observations):
        """
        Processes images.

        Robot and simulator images are 480x600, but our model expects
        150x200, so resize before setting self.to_predictor tensor
        which is passed to the model.
        """
        camera: JPGImage = data.camera
        self.current_image = jpg2rgb(camera.jpg_data)
        self.input_image = image_resize(self.current_image, width=200)
        self.input_image = self.input_image[0:150, 0:200]
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_RGB2YUV)
        self.to_predictor = np.expand_dims(self.input_image, axis=0)

    def compute_action(self, observation):
        """
        Use prior observation to predict best action, return predictions for velocities.
        """
        obs = torch.from_numpy(observation).float().to(self.device)
        linear, angular = self.model(obs)
        linear = linear.to(torch.device('cpu')).data.numpy().flatten()
        angular = angular.to(torch.device('cpu')).data.numpy().flatten()
        return linear, angular

    # ! Major Manipulation here. Should not always change.
    def on_received_get_commands(self, context: Context):
        """
        Converts requested linear/angular velocities to control commands
        and issues them to the robot.

        Don't change this, standard stuff from the submission template.
        """
        linear, angular = self.compute_action(self.to_predictor)
        # Inverse kinematics
        pwm_left, pwm_right = self.convertion_wrapper.convert(linear, angular)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        # LED commands
        grey = RGB(0.0, 0.0, 0.0)
        red = RGB(1.0, 0.0, 0.0)
        blue = RGB(0.0, 0.0, 1.0)
        led_commands = LEDSCommands(red, grey, blue, red, blue)

        # Send PWM command
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """
    Reads JPG bytes as RGB.
    """
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


if __name__ == '__main__':
    node = Agent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)
