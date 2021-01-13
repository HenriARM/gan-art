from flask import Flask, request, make_response, jsonify
from flask_cors import CORS, cross_origin
import requests

from config import DevelopmentConfig

import tensorflow as tf
import numpy as np
from PIL import Image

import json
import io
import os
import random
import string

from minio import Minio
from minio.error import S3Error

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# ================================ METHODS =====================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_random_string_8() -> str:
    length = 8
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def cors(response):
    response = make_response(response)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def initialize_minio():
    # Create a client with the MinIO server playground, its access key and secret key.
    client = Minio(
        endpoint="0.0.0.0:9001",
        access_key="HFKQGYD2P6DJO4U7N33U",
        secret_key="hO09epo9zT6tslwhh192l0UHbtpCjCYTnE3TYtec",
        secure=False
    )

    bucket_name = 'gan'
    # Make bucket if not exist.
    found = client.bucket_exists(bucket_name)

    if not found:
        client.make_bucket(bucket_name)
    else:
        print('Bucket ' + bucket_name + ' already exists')
    return client, bucket_name


# ================================ API =====================================

@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/images/', methods=['GET'])
@cross_origin()
def generate_gallery():
    # read amount of images wanted to be generated
    batch_size = request.args.get('size')
    try:
        batch_size = int(batch_size)
    except ValueError:
        batch_size = 16

    # Create payload for TensorFlow Serving request
    noise = tf.random.normal([batch_size, 1, 1, 100])
    payload = json.dumps({'inputs': noise.numpy().astype('float64')}, cls=NumpyEncoder)

    # Make POST request
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/generator:predict', data=payload, headers=headers)

    # get on response list of images and convert to np.array
    images = json.loads(json_response.text)['outputs']
    images = np.asarray(images)
    print(images.shape)

    # convert images from normalized to RGB suitable
    images *= 127.5 + 127.5
    images = images.astype(np.uint8)

    # initialize Minio
    client, bucket_name = initialize_minio()

    # prepare return payload
    image_names = []

    # loop images
    for i in range(images.shape[0]):
        image = images[i]

        # convert image from np.array to PIL object
        # (save on FS required temporarily)
        image = Image.fromarray(image)
        file_path = './tmp.png'
        image.save(file_path, 'PNG')

        with open(file_path, 'rb') as f:
            # convert image from PIL object to IO stream
            stream = io.BytesIO(f.read())

            # create random name for each image and save
            image_name = get_random_string_8() + '.png'
            image_names.append(image_name)

            # save stream in Minio
            client.put_object(bucket_name=bucket_name,
                              object_name=image_name,
                              data=stream,
                              content_type='image/png',
                              length=stream.getbuffer().nbytes)

        # remove temporary file
        os.remove(file_path)

    return jsonify(image_names)


if __name__ == "__main__":
    config = DevelopmentConfig()
    app.config.from_object(config)
    app.run(host=config.IP, port=config.PORT)
