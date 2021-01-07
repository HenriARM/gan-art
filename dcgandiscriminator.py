import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2D, BatchNormalization


def get_discriminator() -> tf.keras.Sequential:
    return tf.keras.Sequential([

        Conv2D(filters=64, kernel_size=4, strides=2, padding='same', use_bias=False, input_shape=(64, 64, 3)),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=512, kernel_size=4, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(filters=1, kernel_size=4, strides=1, padding='valid', use_bias=False)
    ])


def test_discriminator():
    noise = tf.random.normal([1, 64, 64, 3])
    discriminator = get_discriminator()
    discriminator.summary()
    prediction = discriminator(noise)
    assert prediction.numpy().shape == (1, 1, 1, 1)


def main():
    test_discriminator()


if __name__ == '__main__':
    main()
