import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import tensorflow as tf
tf.python.control_flow_ops = tf

from ..env import carla_env2


from carla_env2 import CarlaEnv


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # Get Current Steering Angle of the car
    steering_angle = data["steering_angle"]

    # Get Current Throttle of the car
    throttle = data["throttle"]

    # Get Current Speed of the car
    speed = data["speed"]

    # Get Current Image of the Center Camera
    imgString = data["image"]


    # Get Image from imgString
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    # Store image as np array
    image_array = np.asarray(image)

    # Transformed Image Array
    transformed_image_array = image_array[None, :, :, :]
    steering_angle = float(model.predict(transformed_image_array, batch_size=1)) * 5.0

    speed = float(speed)

    if speed > 15:
        throttle = 0.1

    else:
        throttle = 0.2

    print("steering_angle & throttle: ", steering_angle, throttle)

    send_control(steering_angle, throttle)

@sio.on('connect')
def control(sid, environ):
    print("connect", sid)
    send_control(0,0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': steering_angle.__str__(), 'throttle':throttle.__str__()}, skip_sid=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')

    args = parser.parse_args()

    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.

        # Restore trained weight to the model
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    # weights_file = "tmp/comma-4b.08-0.03.hdf5"
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)



