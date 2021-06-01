#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import collections
import rosbag
import cv_bridge
from copy import copy
from extract_data_functions import image_preprocessing, synchronize_data
from log_util import Logger
from log_schema import Episode, Step
import cv2

# Change this based on the bag files being used.
VEHICLE_NAME = 'maserati'

# A collection of ros messages coming from a single topic.
MessageCollection = collections.namedtuple(
    "MessageCollection", ["topic", "type", "messages"])

frank_logger = Logger(log_file=f'converted/{VEHICLE_NAME}')


def extract_messages(path, requested_topics):

    # check if path is string and requested_topics a list
    assert isinstance(path, str)
    assert isinstance(requested_topics, list)

    bag = rosbag.Bag(path)

    _, available_topics = bag.get_type_and_topic_info()
    # check if the requested topics exist in bag's topics and if yes extract the messages only for them
    extracted_messages = {}
    for topic in requested_topics:
        if topic in available_topics:
            extracted_messages[topic] = MessageCollection(
                topic=topic, type=available_topics[topic].msg_type, messages=[])

    for msg in bag.read_messages():
        topic = msg.topic
        if topic not in requested_topics:
            continue
        extracted_messages[topic].messages.append(msg)
    bag.close()

    return extracted_messages


def main():

    # define the list of topics that you want to extract
    ros_topics = [
        # the duckiebot name can change from one bag file to the other, so define
        # the topics WITHOUT the duckiebot name in the beginning
        "/camera_node/image/compressed",
        "/joy"
    ]

    # define the bags_directory in order to extract the data
    bags_directory = os.path.join(os.getcwd(), "bag_files")

    # define data_directory
    data_directory = 'converted'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    cvbridge_object = cv_bridge.CvBridge()

    # create a dataframe to store the data for all bag files
    # df_all = pd.DataFrame()

    first_time = True

    for file in os.listdir(bags_directory):
        if not file.endswith(".bag"):
            continue

        # extract bag_ID to include it in the data for potential future use (Useful in case of weird data distributions
        # or final results, since you will be able to associate the data with the bag files)
        bag_ID = file.partition(".bag")[0]

        # extract the duckiebot name to complete the definition of the nodes
        #duckiebot_name = file.partition("_")[2].partition(".bag")[0]
        duckiebot_name = VEHICLE_NAME
        # complete the topics names with the duckiebot name in the beginning
        ros_topics_temp = copy(ros_topics)
        for num, topic in enumerate(ros_topics_temp):
            ros_topics_temp[num] = "/" + duckiebot_name + topic

        # define absolute path of the bag_file
        abs_path = os.path.abspath(os.path.join(bags_directory, file))

        print("Extract data for {} file.".format(file))
        try:
            msgs = extract_messages(abs_path, ros_topics_temp)
            if not msgs:
                continue
        except rosbag.bag.ROSBagException:
            print("Failed to open {}".format(abs_path))
            continue

        ######## This following part is implementation specific ########

        # The composition of the ros messages is different (e.g. different names in the messages) and also different
        # tools are used to handle the different extracted data (e.g. cvbridge for images). As a result, the following
        # part of the script can be used as a basis to extract the data, but IT HAS TO BE MODIFIED based on your topics.

        # easy way to find the structure of your ros messages : print dir(msgs[name_of_topic])

        # extract the images and car_cmds messages
        ext_images = msgs["/" + duckiebot_name +
                          "/camera_node/image/compressed"].messages
        ext_car_cmds = msgs["/" + duckiebot_name + "/joy"].messages

        img_start_timestamp = ext_images[0].timestamp.secs + ext_images[0].timestamp.nsecs * \
            10 ** -len(str(ext_images[0].timestamp.nsecs))
        img_end_timestamp = ext_images[-1].timestamp.secs + ext_images[-1].timestamp.nsecs * \
            10 ** -len(str(ext_images[-1].timestamp.nsecs))
        control_start_timestamp = ext_car_cmds[0].timestamp.secs + ext_car_cmds[0].timestamp.nsecs * \
            10 ** -len(str(ext_car_cmds[0].timestamp.nsecs))
        control_end_timestamp = ext_car_cmds[-1].timestamp.secs + ext_car_cmds[-1].timestamp.nsecs * \
            10 ** -len(str(ext_car_cmds[-1].timestamp.nsecs))

        last_timestamp_to_extract = min(
            img_start_timestamp + 0.9 * (img_end_timestamp - img_start_timestamp),
            control_start_timestamp + 0.9 * (control_end_timestamp - control_start_timestamp),
        )

        # create dataframe with the images and the images' timestamps
        for num, img in enumerate(ext_images):
            # hack to get the timestamp of each image in <float 'secs.nsecs'> format instead of <int 'rospy.rostime.Time'>
            temp_timestamp = ext_images[num].timestamp
            img_timestamp = temp_timestamp.secs + temp_timestamp.nsecs * \
                10 ** -len(str(temp_timestamp.nsecs))

            if img_timestamp > last_timestamp_to_extract:
                continue

            # get the rgb image
            #### direct conversion to CV2 ####
            # np_arr = np.fromstring(img.data, np.uint8)
            # img = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
            # print("img", img, img.shape)
            img = cvbridge_object.compressed_imgmsg_to_cv2(img.message)
            img = image_preprocessing(img)

            temp_df = pd.DataFrame({
                'img': [img],
                'img_timestamp': [img_timestamp]
            })

            if num == 0:
                df_imgs = temp_df.copy()
            else:
                df_imgs = df_imgs.append(temp_df, ignore_index=True)

        # create dataframe with the car_cmds and the car_cmds' timestamps
        for num, cmd in enumerate(ext_car_cmds):

            # read wheel commands messages
            cmd_msg = cmd.message
            # hack to get the timestamp of each image in <float 'secs.nsecs'> format instead of <int 'rospy.rostime.Time'>
            temp_timestamp = ext_car_cmds[num].timestamp
            vel_timestamp = temp_timestamp.secs + temp_timestamp.nsecs * \
                10 ** -len(str(temp_timestamp.nsecs))

            if vel_timestamp > last_timestamp_to_extract:
                continue

            temp_df = pd.DataFrame({
                'vel_timestamp': [vel_timestamp],
                'vel_linear': [cmd_msg.axes[1]],
                'vel_angular': [cmd_msg.axes[3]],
            })
            if num == 0:
                df_cmds = temp_df.copy()
            else:
                df_cmds = df_cmds.append(temp_df, ignore_index=True)

        # synchronize data
        print()
        print("Starting synchronization of data for {} file.".format(file))

        temp_synch_data, temp_synch_imgs = synchronize_data(
            df_imgs, df_cmds, bag_ID)

        if first_time:
            synch_data = copy(temp_synch_data)
            synch_imgs = copy(temp_synch_imgs)
            first_time = False
        else:
            synch_data = np.vstack((synch_data, temp_synch_data))
            synch_imgs = np.vstack((synch_imgs, temp_synch_imgs))

        print("\nShape of total data: {} , shape of total images: {}\n".format(
            synch_data.shape, synch_imgs.shape))

        for i in range(synch_data.shape[0]):
            action = synch_data[i]
            tobelogged_action = np.array([action[2], action[3]],dtype=float)
            tobelogged_image = synch_imgs[i*150:(i+1)*150, :, :]
            tobelogged_image = cv2.cvtColor(
                tobelogged_image, cv2.COLOR_BGR2YUV)
            done = False if (i < synch_data.shape[0]) else True
            step = Step(tobelogged_image, None, tobelogged_action, done)
            frank_logger.log(step, None)
        frank_logger.on_episode_done()
    print("Synchronization of all data is finished.\n")
    frank_logger.close()


if __name__ == "__main__":
    main()
    print("All complete. You can find your log in converted folder.")
