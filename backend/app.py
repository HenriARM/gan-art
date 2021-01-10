from flask import Flask, send_file, request, make_response
from flask_cors import CORS, cross_origin
from config import ProductionConfig, DevelopmentConfig
import tensorflow as tf
import numpy as np

from PIL import Image


# import base64
import json
import io
# from io import BytesIO
import requests

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# def cors(response):
#     response = make_response(response)
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     response.headers.add("Access-Control-Allow-Headers", "*")
#     response.headers.add("Access-Control-Allow-Methods", "*")
#     return response


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/images/', methods=['GET'])
@cross_origin()
def image_classifier():
    batch_size = request.args.get('size')
    try:
        batch_size = int(batch_size)
    except ValueError:
        batch_size = 16

    noise = tf.random.normal([batch_size, 1, 1, 100])

    # Creating payload for TensorFlow serving request
    payload = json.dumps({'inputs': noise.numpy().astype('float64')}, cls=NumpyEncoder)

    # Making POST request
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/generator:predict', data=payload, headers=headers)
    images = json.loads(json_response.text)['outputs']
    images = np.asarray(images)
    print(images.shape)

    img = images[0] * 127.5 + 127.5
    img = img.astype(np.uint8)

    new_img = Image.fromarray(img)
    # filename = './img.png'
    # new_img.save(filename, 'PNG')

    output = io.BytesIO()
    new_img.save(output, format='JPEG')
    hex_data = output.getvalue()

    # return send_file(filename, mimetype='image/png')
    return send_file(output, mimetype='image/png')

    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

# TODO: add upload of 2 images

if __name__ == "__main__":
    config = DevelopmentConfig()
    app.config.from_object(config)
    app.run(host=config.IP, port=config.PORT)
