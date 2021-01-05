import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape , UpSampling2D, Conv2D
import requests
import json

import numpy as np

class Generator:
    def __init__(self):
        self.model = tf.keras.Sequential([
            # can use model.build(input_shape), before running model.summary() or training
            Dense(32 * 32 * 128, use_bias=False, input_shape=(100,)),
            LeakyReLU(),
            BatchNormalization(momentum=0.8),
            Reshape((32, 32, 128)),

            UpSampling2D(size=(2, 2)),

            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False),
            LeakyReLU(),
            BatchNormalization(momentum=0.8),

            UpSampling2D(size=(2, 2)),

            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False),
            LeakyReLU(),
            BatchNormalization(momentum=0.8),

            Conv2D(filters=3, kernel_size=3, strides=1, padding='same', use_bias=False)
        ])

    def get_model(self) -> tf.keras.Sequential:
        return self.model

# TODO: how to code like this whole Sequential
# class Generator(tf.keras.Model):
#
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc = Dense(7 * 7 * 256, use_bias=False, input_shape=(100,))
#
#     def call(self, x, **kwargs):
#         x = self.fc(x)
#         # BatchNormalization()
#         x = LeakyReLU(x)


def main():
    noise = tf.random.normal([1, 100])

    # model = Generator().get_model()
    # # print(model.call(inputs=noise, training=False))
    #
    # # The saved_model.pb file stores the actual TensorFlow program, or model,
    # # and a set of named signatures, each identifying a function that accepts tensor
    # # inputs and produces tensor outputs.
    # # The assets directory contains files used by the TensorFlow graph,
    # # for example text files used to initialize vocabulary tables.
    # path = './models/1'
    # tf.saved_model.save(obj=model, export_dir=path)
    # tf.saved_model.load(export_dir=path)

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




if __name__ == '__main__':
    main()
