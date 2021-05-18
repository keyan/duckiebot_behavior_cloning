#!/usr/bin/env python3

import numpy as np
import pandas as pd
from copy import copy
import cv2

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def image_preprocessing(image):

    # Incomming is 480x640

    # define new image's height and width

    new_img = image_resize(image,width=200)


    return new_img


def synchronize_data(df_imgs, df_cmds, bag_ID):
    # initialize a dataframe to append all new values
    synch_data = pd.DataFrame()
    synch_imgs = pd.DataFrame()
    skip_counter = 0
    first_time = True

    # for each omega velocity, find the respective image
    for cmd_index, cmd_time in enumerate(df_cmds['vel_timestamp']):

        # we keep only the data for which the duckiebot is moving (we do not want the duckiebot to learn to remain at rest)
        if ( df_cmds['vel_linear'][cmd_index] != 0) or ( df_cmds['vel_angular'][cmd_index] != 0):
            # find index of image with the closest timestamp to wheels' velocities timestamp
            img_index = ( np.abs( df_imgs['img_timestamp'].values - cmd_time ) ).argmin()

            # The image precedes the omega velocity, thus image's timestamp must be smaller
            if ( ( df_imgs['img_timestamp'][img_index] - cmd_time ) > 0 ) & (img_index - 1 < 0):

                # if the image appears after the velocity and there is no previous image, then
                # there is no safe synchronization and the data should not be included
                skip_counter += 1
                continue
            else:

                # if the image appears after the velocity, in this case we know that there is previous image and we
                # should prefer it
                if ( df_imgs['img_timestamp'][img_index] - cmd_time ) > 0 :

                    img_index = img_index - 1

                # create a numpy array for all data except the images
                temp_data = np.array( [[
                    df_imgs['img_timestamp'][img_index],
                    df_cmds["vel_timestamp"][cmd_index],
                    df_cmds['vel_linear'][cmd_index],
                    df_cmds['vel_angular'][cmd_index],
                    bag_ID
                ]] )

                # create a new numpy array only for images (images are row vectors of size (1,4608) and it is more
                # convenient to save them separately
                temp_imgs = df_imgs['img'][img_index]

                if first_time:

                    synch_data = copy(temp_data)
                    synch_imgs = copy(temp_imgs)
                    first_time = False

                else:

                    synch_data = np.vstack((synch_data, temp_data))
                    synch_imgs = np.vstack((synch_imgs, temp_imgs))

    print()
    print("Synchronization of {}.bag file is finished. From the initial {} images and {} velocities commands, the extracted "
          "synchronized data are {}.synchronized image are {}.".format(bag_ID, df_imgs.shape[0], df_cmds.shape[0], synch_data.shape[0],synch_imgs.shape[0]))
    print()
    print("Skipped {} images due to asynchronous".format(skip_counter))
    # return the synchronized data to the main function
    return synch_data, synch_imgs
