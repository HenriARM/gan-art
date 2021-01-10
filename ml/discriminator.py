import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Flatten, Dropout, AveragePooling2D, Activation
from generator import Generator


class Discriminator(tf.keras.Model):

    def __init__(self, name='discriminator', **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        # When padding='some' and stride=2, calculate padding as if input=output and stride=1.
        # Then use that padding with stride 2, and you will get exactly output 2x smaller than input
        # 1x1 convolution (TODO: why?)
        self.conv1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', input_shape=(128, 128, 3))
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.conv4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)

        x = self.conv2(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)

        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = self.conv3(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)

        x = AveragePooling2D(pool_size=(2, 2))(x)

        x = self.conv4(x)
        x = LeakyReLU()(x)
        # x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(1)(x)
        # returns a tensor with values squashed between [0..1]
        x = tf.keras.activations.sigmoid(x)
        return x


def get_discriminator() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        Conv2D(filters=64, kernel_size=1, strides=1, padding='same', input_shape=(128, 128, 3)),
        LeakyReLU(),
        # Dropout(0.3),

        Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(),
        # Dropout(0.3),

        AveragePooling2D(pool_size=(2, 2)),

        Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(),
        # Dropout(0.3),

        AveragePooling2D(pool_size=(2, 2)),

        Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
        LeakyReLU(),
        # Dropout(0.3),

        Flatten(),
        Dense(1),
        # returns a tensor with values squashed between [0..1]
        Activation('sigmoid')
    ])


def test_discriminator():
    noise = tf.random.normal([1, 100])
    generator = Generator()
    fake_image = generator(noise)
    discriminator = Discriminator()

    prediction = discriminator(fake_image)
    assert prediction.numpy().shape == (1, 1)

    # model = get_discriminator()
    # model.summary()


def main():
    test_discriminator()


if __name__ == '__main__':
    main()
