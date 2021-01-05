import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, UpSampling2D, \
    Conv2D, Flatten, Dropout, AveragePooling2D

from generator import Generator


class Discriminator:
    def __init__(self):
        self.model = tf.keras.Sequential([
            # When padding='some' and stride=2, calculate padding as if input=output and stride=1.
            # Then use that padding with stride 2, and you will get exactly output 2x smaller than input

            # 1x1 convolution (TODO: why?)
            Conv2D(filters=64, kernel_size=1, strides=1, padding='same', input_shape=[128, 128, 3]),
            LeakyReLU(),

            Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
            LeakyReLU(),
            Dropout(0.3),

            AveragePooling2D(pool_size=(2, 2)),

            Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
            LeakyReLU(),
            Dropout(0.3),

            AveragePooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(1)
        ])

    def get_model(self) -> tf.keras.Sequential:
        return self.model


def main():
    noise = tf.random.normal([1, 100])
    image = Generator().get_model().call(inputs=noise, training=False)

    print(Discriminator().get_model().call(inputs=image, training=False))


if __name__ == '__main__':
    main()