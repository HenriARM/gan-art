import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, UpSampling2D, Conv2D, Activation
import requests
import json
import numpy as np


class Generator(tf.keras.Model):

    def __init__(self, name='generator', **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.fc = Dense(32 * 32 * 32, use_bias=False, input_shape=(100,))
        self.conv1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv3 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False)

    def call(self, x, **kwargs):
        # TODO: BatchNormalization put in init
        x = self.fc(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Reshape((32, 32, 32))(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = self.conv1(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = self.conv2(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = self.conv3(x)
        # returns a tensor with values squashed between [-1..1]
        x = tf.keras.activations.tanh(x)
        return x


def get_generator() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        Dense(32 * 32 * 32, use_bias=False, input_shape=(100,)),
        LeakyReLU(),
        BatchNormalization(momentum=0.8),
        Reshape((32, 32, 32)),

        UpSampling2D(size=(2, 2)),
        Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False),
        LeakyReLU(),
        BatchNormalization(momentum=0.8),

        UpSampling2D(size=(2, 2)),
        Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False),
        LeakyReLU(),
        BatchNormalization(momentum=0.8),

        Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False),
        # returns a tensor with values squashed between [-1..1]
        Activation('tanh')
    ])


def test_generator():
    BATCH = 1
    noise = tf.random.normal([BATCH, 100])
    model = Generator()
    image = model(noise)
    assert image.numpy().shape == (BATCH, 128, 128, 3)

    # TODO: returns error because of BatchNormalization layer
    # model.save('my-model')

    # Same Sequential model is working
    # BATCH = 1
    # noise = tf.random.normal([BATCH, 100])
    # model = get_generator()
    # image = model(noise)
    # model.save('my-model')
    # model.summary()


def main():
    # test_generator()

    # # The saved_model.pb file stores the actual TensorFlow program, or model,
    # # and a set of named signatures, each identifying a function that accepts tensor
    # # inputs and produces tensor outputs.
    # # The assets directory contains files used by the TensorFlow graph,
    # # for example text files used to initialize vocabulary tables.
    # path = './models/1'
    # tf.saved_model.save(obj=model, export_dir=path)
    # tf.saved_model.load(export_dir=path)

    noise = tf.random.normal([32, 100])

    # Creating payload for TensorFlow serving request
    payload = json.dumps({
        'inputs': [list(noise.numpy().flatten().astype('float64'))]
    })

    # Making POST request
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8502/v1/models/generator:predict', data=payload, headers=headers)
    image = json.loads(json_response.text)['outputs']
    print(np.asarray(image).shape)

    # Decoding results from TensorFlow Serving server
    # pred = json.loads(r.content.decode('utf-8'))
    # print(pred.shape)


    # gen = Generator()
    # # tf checkpoint
    # # gen.save_weights(filepath='./easy_checkpoint.h5', save_format='h5')
    # # In tf mode creates 3 filels - checkpoint, .index, .data
    # gen.save_weights(filepath='easy_checkpoint', save_format='tf')


if __name__ == '__main__':
    main()
