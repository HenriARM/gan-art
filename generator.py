import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, UpSampling2D, Conv2D
import requests
import json
import numpy as np


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc = Dense(32 * 32 * 128, use_bias=False, input_shape=(100,))
        self.conv1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv3 = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False)

    def call(self, x, **kwargs):
        x = self.fc(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Reshape((32, 32, 128))(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = self.conv1(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = self.conv2(x)
        x = LeakyReLU()(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = self.conv3(x)
        return x


def test_generator():
    noise = tf.random.normal([32, 100])
    model = Generator()
    image = model(noise)
    assert image.numpy().shape == (32, 128, 128, 3)


def main():
    test_generator()
    pass


    # # The saved_model.pb file stores the actual TensorFlow program, or model,
    # # and a set of named signatures, each identifying a function that accepts tensor
    # # inputs and produces tensor outputs.
    # # The assets directory contains files used by the TensorFlow graph,
    # # for example text files used to initialize vocabulary tables.
    # path = './models/1'
    # tf.saved_model.save(obj=model, export_dir=path)
    # tf.saved_model.load(export_dir=path)

    # # Creating payload for TensorFlow serving request
    # payload = json.dumps({
    #     'inputs': [list(noise.numpy().flatten().astype('float64'))]
    # })
    #
    # # Making POST request
    # headers = {"content-type": "application/json"}
    # json_response = requests.post('http://localhost:8502/v1/models/generator:predict', data=payload, headers=headers)
    # image = json.loads(json_response.text)['outputs']
    # print(np.asarray(image).shape)
    #
    # # Decoding results from TensorFlow Serving server
    # # pred = json.loads(r.content.decode('utf-8'))
    # # print(pred.shape)


    # gen = Generator()
    # # tf checkpoint
    # # gen.save_weights(filepath='./easy_checkpoint.h5', save_format='h5')
    # # In tf mode creates 3 filels - checkpoint, .index, .data
    # gen.save_weights(filepath='easy_checkpoint', save_format='tf')


if __name__ == '__main__':
    main()
