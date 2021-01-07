import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2DTranspose


# TODO: BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


def get_generator() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        Conv2DTranspose(filters=64 * 8, kernel_size=4, strides=1, output_padding=0, use_bias=False,
                        input_shape=(1, 1, 100)),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')
    ])


def test_generator():
    BATCH_SIZE = 1
    noise = tf.random.normal([BATCH_SIZE, 1, 1, 100])
    model = get_generator()
    model.summary()
    image = model(noise)
    assert image.numpy().shape == (BATCH_SIZE, 64, 64, 3)


def main():
    test_generator()


if __name__ == '__main__':
    main()
