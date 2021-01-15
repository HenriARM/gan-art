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
from minio.deleteobjects import DeleteObject

# from minio.error import S3Error

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

minio_client = None
bucket_name = None


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


def clear_minio_bucket(mc, bn):
    images = mc.list_objects(bucket_name=bn, prefix=None, recursive=True)

    for image in images:
        print(image.object_name)
        mc.remove_object(bucket_name=bn, object_name=image.object_name)

    print('Minio bucket is cleaned')


def interpolate_vectors(v1, v2, num_steps):
    # normalized vector = vector / length of vector
    v1_norm = tf.norm(v1)
    v2_norm = tf.norm(v2)
    # v2 / v2_norm is length
    # v2_normalized = v2/v2_norm * v1_norm
    v2_normalized = v2 * (v1_norm / v2_norm)

    vectors = []
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm = tf.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        vectors.append(interpolated_normalized)
    return tf.stack(vectors)


# ================================ API =====================================

@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/images', methods=['GET'])
@cross_origin()
def generate_gallery():
    # read dataset type for model
    dataset_type = request.args.get('dataset')

    if dataset_type is None:
        return jsonify('No datasetType was set')

    # read amount of images wanted to be generated
    batch_size = request.args.get('size')

    try:
        batch_size = int(batch_size)
    except ValueError:
        batch_size = 16

    if dataset_type == 'celeba128':
        # Create payload for TensorFlow Serving request
        noise = tf.random.normal([batch_size, 512])
        payload = json.dumps({'inputs': noise.numpy().astype('float64')}, cls=NumpyEncoder)
    else:
        # Create payload for TensorFlow Serving request
        noise = tf.random.normal([batch_size, 1, 1, 100])
        payload = json.dumps({'inputs': noise.numpy().astype('float64')}, cls=NumpyEncoder)

    # Make POST request
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/' + dataset_type + ':predict',
                                  data=payload,
                                  headers=headers)

    # get on response list of images and convert to np.array
    images = json.loads(json_response.text)['outputs']
    images = np.asarray(images)
    print(images.shape)

    # convert images from normalized to RGB suitable
    images *= 127.5 + 127.5
    images = images.astype(np.uint8)

    # prepare return payload
    images_payload = []

    # loop images
    for i in range(images.shape[0]):
        image = images[i]

        # convert image from np.array to PIL object
        # (save on FS required temporarily)
        image = Image.fromarray(image)
        file_path = './tmp.png'
        image.save(file_path, 'PNG')

        image_name = None
        with open(file_path, 'rb') as f:
            # convert image from PIL object to IO stream
            stream = io.BytesIO(f.read())

            # create random name for each image and save
            image_name = get_random_string_8()
            image_path = image_name + '.png'
            image_link = 'http://' + '0.0.0.0:9001' + '/' + 'gan' + '/' + image_path

            # save in payload image link and name
            images_payload.append((image_name, image_link))

            # save stream in Minio
            client.put_object(bucket_name=bucket_name,
                              object_name=image_path,
                              data=stream,
                              content_type='image/png',
                              length=stream.getbuffer().nbytes)

        # save latent vector in numpy format for interpolation
        latent_vector_name = image_name + '_latent.npz'
        np.savez_compressed(latent_vector_name, noise[i])

        with open(latent_vector_name, 'rb') as f:
            # convert image from PIL object to IO stream
            stream = io.BytesIO(f.read())
            client.put_object(bucket_name=bucket_name,
                              object_name=latent_vector_name,
                              data=stream,
                              # content_type='image/png',
                              length=stream.getbuffer().nbytes)

        # remove temporary file
        os.remove(latent_vector_name)

    return jsonify(images_payload)


@app.route('/interpolate', methods=['GET'])
@cross_origin()
def interpolate():
    # read dataset type for model
    dataset_type = request.args.get('dataset')

    if dataset_type is None:
        return jsonify('No datasetType was set')

    # read both images to interpolate
    img1 = request.args.get('img1')
    img2 = request.args.get('img2')

    if img1 is None or img2 is None:
        return jsonify('Not all images are set for interpolation')

    response = client.get_object(bucket_name=bucket_name, object_name=img1 + '_latent.npz')
    img1_latent_file_path = img1 + '_latent.npz'
    img1_latent_file = open(img1_latent_file_path, 'wb')
    img1_latent_file.write(response.data)
    img1_latent_file.close()

    img1_latent_vector = np.load(img1_latent_file_path)['arr_0']
    os.remove(img1_latent_file_path)

    response = client.get_object(bucket_name=bucket_name, object_name=img2 + '_latent.npz')
    img2_latent_file_path = img2 + '_latent.npz'
    img2_latent_file = open(img2_latent_file_path, 'wb')
    img2_latent_file.write(response.data)
    img2_latent_file.close()

    img2_latent_vector = np.load(img2_latent_file_path)['arr_0']
    os.remove(img2_latent_file_path)

    images = interpolate_vectors(img1_latent_vector, img2_latent_vector, 50)
    noise = images

    if dataset_type == 'celeba128':
        # Create payload for TensorFlow Serving request
        payload = json.dumps({'inputs': noise.numpy().astype('float64')}, cls=NumpyEncoder)
    else:
        # Create payload for TensorFlow Serving request
        payload = json.dumps({'inputs': noise.numpy().astype('float64')}, cls=NumpyEncoder)

    # Make POST request
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/' + dataset_type + ':predict',
                                  data=payload,
                                  headers=headers)

    # get on response list of images and convert to np.array
    images = json.loads(json_response.text)['outputs']
    images = np.asarray(images)
    print(images.shape)

    # convert images from normalized to RGB suitable
    images *= 127.5 + 127.5
    images = images.astype(np.uint8)

    # prepare return payload
    images_payload = []

    # loop images
    for i in range(images.shape[0]):
        image = images[i]

        # convert image from np.array to PIL object
        # (save on FS required temporarily)
        image = Image.fromarray(image)
        file_path = './tmp.png'
        image.save(file_path, 'PNG')

        image_name = None
        with open(file_path, 'rb') as f:
            # convert image from PIL object to IO stream
            stream = io.BytesIO(f.read())

            # create random name for each image and save
            image_name = get_random_string_8()
            image_path = image_name + '.png'
            image_link = 'http://' + '0.0.0.0:9001' + '/' + 'gan' + '/' + image_path

            # save in payload image link and name
            images_payload.append((image_name, image_link))

            # save stream in Minio
            client.put_object(bucket_name=bucket_name,
                              object_name=image_path,
                              data=stream,
                              content_type='image/png',
                              length=stream.getbuffer().nbytes)

    return jsonify(images_payload)


if __name__ == "__main__":
    config = DevelopmentConfig()
    app.config.from_object(config)

    # initialize Minio
    client, bucket_name = initialize_minio()

    # remove all images
    clear_minio_bucket(client, bucket_name)

    app.run(host=config.IP, port=config.PORT)
