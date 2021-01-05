import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Flatten, Dropout, AveragePooling2D
from generator import Generator


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        # When padding='some' and stride=2, calculate padding as if input=output and stride=1.
        # Then use that padding with stride 2, and you will get exactly output 2x smaller than input
        # 1x1 convolution (TODO: why?)
        self.conv1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', input_shape=(128, 128, 3))
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = LeakyReLU()(x)

        x = self.conv2(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)

        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        return x


def test_discriminator():
    noise = tf.random.normal([1, 100])
    generator = Generator()
    fake_image = generator(noise)
    discriminator = Discriminator()

    prediction = discriminator(fake_image)
    assert prediction.numpy().shape == (1, 1)


def main():
    test_discriminator()


if __name__ == '__main__':
    main()