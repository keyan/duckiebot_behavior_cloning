import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Activation,Lambda,Flatten,Dense
#! Model in reference to Nvidia's Behavior Cloning Paper.

class FrankNet:
    @staticmethod
    def build_linear_branch(inputs=(150, 200, 3)):
        # ? Input Normalization
        x = Lambda(lambda x: x/255.0)(inputs)

        # ? L1: CONV => RELU
        x = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L2: CONV => RELU
        x = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L3: CONV => RELU
        x = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L4: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L5: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)

        # ? Flatten
        x = Flatten()(x)

        # ? Fully Connected
        x = Dense(1164, kernel_initializer='normal', activation='relu')(x)
        x = Dense(100, kernel_initializer='normal', activation='relu')(x)
        x = Dense(50, kernel_initializer='normal', activation='relu')(x)
        x = Dense(10, kernel_initializer='normal', activation='relu')(x)
        x = Dense(1, kernel_initializer='normal', name="Linear")(x)

        return x

    @staticmethod
    def build_angular_branch(inputs=(150, 200, 3)):
        # ? Input Normalization
        x = Lambda(lambda x: x/255.0)(inputs)

        # ? L1: CONV => RELU
        x = Conv2D(24, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L2: CONV => RELU
        x = Conv2D(36, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L3: CONV => RELU
        x = Conv2D(48, (5, 5), strides=(2, 2), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L4: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)
        # ? L5: CONV => RELU
        x = Conv2D(64, (3, 3), padding="valid")(x)
        x = Activation("relu")(x)

        # ? Flatten
        x = Flatten()(x)

        # ? Fully Connected
        x = Dense(1164, kernel_initializer='normal', activation='relu')(x)
        x = Dense(100, kernel_initializer='normal', activation='relu')(x)
        x = Dense(50, kernel_initializer='normal', activation='relu')(x)
        x = Dense(10, kernel_initializer='normal', activation='relu')(x)
        x = Dense(1, kernel_initializer='normal', name="Angular")(x)

        return x

    @staticmethod
    def build(width=150, height=200):
        input_shape = (height, width, 3)
        inputs = tf.keras.Input(shape=input_shape)
        linearVelocity = FrankNet.build_linear_branch(inputs)
        angularVelocity = FrankNet.build_angular_branch(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[
            linearVelocity, angularVelocity], name="FrankNet")

        return model
