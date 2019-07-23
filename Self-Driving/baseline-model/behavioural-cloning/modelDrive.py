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




    



