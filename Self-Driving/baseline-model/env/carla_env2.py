from __future__ import print_function

import carla

from carla import ColorConverter as cc

import os
import glob
import sys
import time
import numpy as np

import weakref
import random

try:
	sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import cv2

class CarlaEnv(object):


	ACTION_NAMES = ["steer", "throttle", "brake", "reverse"]
	STEER_LIMIT_LEFT = -1.0
	STEER_LIMIT_RIGHT = 1.0
	THROTTLE_MIN = 0.0
	THROTTLE_MAX = 1.0
	BRAKE_MIN = 0.0
	BRAKE_MAX = 1.0
	REVERSE_MIN = 0.0
	REVERSE_MAX = 1.0
	VAL_PER_PIXEL = 255

	def __init__(self,
				 host='127.0.0.1',
				 port=2000,
				 isSave=False,
				 headless=False):
		print("__init__")
		self.isSave = isSave
		self.host, self.port = host, port
		self.width = 1920
		self.height = 1080
		self.steps = 0
		self.num_episodes = 0

		self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN, self.BRAKE_MIN, self.REVERSE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX, self.BRAKE_MAX, self.REVERSE_MAX]), dtype=np.float32 )

		self.player = None
		self.image_array = np.zeros((self.height,self.width,0))
		self.lidar_array = np.zeros((self.height,self.width,0))
		self._make_carla_client(self.host, self.port)
		self._get_world()
		self._make_vehicle()


	def step(self, action):
		self._control = carla.VehicleControl()
		self._control.steer = float(action[0])
		self._control.throttle = float(action[1])
		self._control.brake = float(action[2]) > 0.5
		self._control.hand_brake = False
		self._control.reverse = float(action[3]) > 0.5
		self.player.apply_control(self._control)

		obs = self.get_observation()

		return obs, 0.1, False, {}

	def render(self):

		obs = self.get_observation()

		return obs

	def reset(self):

		pass

	def disconnect(self):

		pass

	def get_observation(self):

		return self.image_array

	def get_camera_image(self):
		return self.image_array

	def get_lidar_image(self):
		return self.lidar_array

	def close(self):
		self.player.destroy()
		self.camera.destroy()
		self.lidar.destroy()


	def _make_carla_client(self, host, port):
		print("_make_carla_client")
		self._client = carla.Client(host, port)
		self._client.set_timeout(2.0)
		# while True:
		# 	try:
		# 		self._client = carla.Client(host, port)
		# 		self._client.set_timeout(2.0)
				

		# 	except:
		# 		print("Connection error")
		# 		time.sleep(1)

	def _get_world(self):
		print("_get_world")
		self.world = self._client.get_world()
		self.map = self.world.get_map()

	def _make_vehicle(self):
		print("_make_vehicle")
		print(self.world.get_blueprint_library().filter("vehicle.audi.*"))
		blueprint = random.choice(self.world.get_blueprint_library().filter("vehicle.audi.*"))
		blueprint.set_attribute('role_name', 'hero')

		if blueprint.has_attribute('color'):
			color = random.choice(blueprint.get_attribute('color').recommended_values)
			blueprint.set_attribute('color', color)
		if self.player is None:
			spawn_points = self.map.get_spawn_points()
			spawn_point = spawn_points[0]
			# spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
			self.player = self.world.try_spawn_actor(blueprint, spawn_point)

		#create camera attach on vehicle
		self._sensors = [
			['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
			['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
			['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
			['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
			['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
			['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
			['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]

		bp_library = self.world.get_blueprint_library()
		for item in self._sensors:
			bp = bp_library.find(item[0])
			if item[0].startswith('sensor.camera'):
				bp.set_attribute('image_size_x', str(self.width))
				bp.set_attribute('image_size_y', str(self.height))
			elif item[0].startswith('sensor.lidar'):
				bp.set_attribute('range', '5000')

		#camera
		bp = self.world.get_blueprint_library().find(self._sensors[0][0])
		self.camera = self.world.spawn_actor(bp,
				carla.Transform(carla.Location(x=1.6, z=1.7)),
				attach_to=self.player)
		weak_self = weakref.ref(self)
		self.camera.listen(lambda image: self._parse_rgb_image(weak_self, image))

		#lidar
		bp = self.world.get_blueprint_library().find(self._sensors[6][0])
		self.lidar = self.world.spawn_actor(bp,
				carla.Transform(carla.Location(x=1.6, z=1.7)),
				attach_to=self.player)
		weak_self = weakref.ref(self)
		self.lidar.listen(lambda image: self._parse_lidar_image(weak_self, image))

	@staticmethod
	def _parse_rgb_image(weak_self, image):
		self = weak_self()
		if not self:
			return
		image.convert(self._sensors[0][1])
		array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		array = np.reshape(array, (image.height, image.width, 4))
		array = array[:, :, :3]
		array = array[:, :, ::-1]

		self.image_array = array
		if self.isSave:
			image.save_to_disk('_out/%08d' % image.frame_number)

		print("_parse_rgb_image")


	@staticmethod
	def _parse_lidar_image(weak_self, image):
		self = weak_self()
		if not self:
			return

		points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
		points = np.reshape(points, (int(points.shape[0]/3), 3))
		lidar_data = np.array(points[:, :2])
		lidar_data *= min(self.width, self.height) / 100.0
		lidar_data += (0.5 * self.width, 0.5 * self.height)
		lidar_data = np.fabs(lidar_data)
		lidar_data = lidar_data.astype(np.int32)
		lidar_data = np.reshape(lidar_data, (-1, 2))
		lidar_img_size = (self.width, self.height, 3)
		lidar_img = np.zeros(lidar_img_size)
		lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
		self.lidar_array = lidar_img
		lidar_img = lidar_img.swapaxes(0, 1)
		if self.isSave:
			cv2.imwrite('_out/lidar%08d.png' % image.frame_number, lidar_img)
			# image.save_to_disk('_out/lidar%08d' % image.frame_number)


