import tensorflow as tf
import numpy as np
from vizdoom import*

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



# Create Env
# Doom Env takes (1) configuration file => to handle all the options (size of frames, possible actions ... )
# (2) scenario file => Generates the correct scenario

# 3 Possible actions [[0,0,1],[1,0,0],[0,1,0]]
# Monster is spawned randomly somewhere along the opposite wall
# 1 hit to kill a monster
# Episode ends when monster is killed OR on timeout (300)

# Reward:
# +101 for killing the monster
# -5 for missing
# Episode ends after killing the monster or on timeout
# Living reward = -1


def create_environment():
	game = DoomGame()

	# Load configuration
	game.load_config("basic.cfg")

	# Load scenario
	game.set_doom_scenario_path("basic.wad")

	game.init()

	# Possible actions
	left = [1,0,0]
	right = [0,1,0]
	shoot = [0,0,1]
	possible_actions = [left,right, shoot]

	return game, possible_actions

# Perform random action to test the environment
def test_environment():
	game = DoomGame()
	game.load_config("basic.cfg")
	game.set_doom_scenario_path("basic.wad")
	game.init()
	shoot = [0,0,1]
	left = [1,0,0]
	right = [0,1,0]
	actions = [shoot, left, right]


	episodes = 10

	for i in range(episodes):
		game.new_episode()

		# while game is not finished
		while not game.is_episode_finished():
			state = game.get_state()
			img = state.game_variables
			misc = state.game_variables
			action = random.choice(actions)
			print(action)
			reward = game.make_action(action)
			print("\treward:", reward)
			time.sleep(0.02)
		print("Result:", game.get_total_reward())
		time.sleep(2)

	game.close()

# Create the DoomGame env
game, possible_actions = create_environment()

# Define the Preprocessing Functions
# Preprocess => so that we can reduce the complexity of our states -> reduce the computation time needed for training

'''
Step 1 : Grayscale each frames
Step 2 : Crop the screen
Step 3 : Normalize pixel values
Step 4 : Resize preprocessed frame 
'''
def preprocess_frame(frame):
	# Step 1: Grayscale (handled by the environment)
	# Step 2: Crop the screen
	cropped_frame = frame[30:-10,30:-30]

	# Step 3: Normalized frame
	normalized_frame = cropped_frame/255.0

	# Step 4: Resize
	preprocess_frame = transform.resize(normalized_frame, [84,84])

	return preprocess_frame

# Define number of frame to be stacked
stack_size = 4

# Initialize deque with ZERO-IMAGES
# collections.deque() is a double-ended queue. Can be used to add or remove elements from both ends
stacked_frames = deque([np.zeros((84,84),dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
	frame = preprocess_frame(state)

	if is_new_episode:
		# Clear the stacked_frames
		stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

		# Put the frame into the stacked frame
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)

		stacked_state = np.stack(stacked_frames, axis=2)

	else:
		stacked_frames.append(frame)
		stacked_state = np.stack(stacked_frames, axis=2)

	return stacked_state, stacked_frames


'''
Hyperparameters set up
'''

state_size = [84,84,4]
action_size = game.get_available_buttons_size() # left, right, shoot
learning_rate = 0.0002

total_episodes = 500
max_steps = 100
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

# Q-Learning hyperparameters
gamma = 0.95

# Memory Hyperparameters
pretrain_length = batch_size #num of experiences stored in the memory when initialized for the first time
memory_size = 1000000 #num of experience that memory can keep

training = True
episode_render = False

'''
Create Deep Q-Learning Neural Network Model
'''

class DQNetwork:
	def __init__(self, state_size, action_size, learning_rate, name = 'DQNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			# Define placeholders
			# state_size: we take each elements of state_size in tuple and like [None, 84,84,4]
			self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
			self.actions = tf.placeholder(tf.float32, [None, 3], name="action")

			self.target_Q = tf.placeholder(tf.float32, [None], name="target")

			# Conv_layer1
			self.conv1 = tf.layers.conv2d(intput = self.inputs, filters=32, kernel_size=[8,8], strides=[4,4], padding="VALID",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv1')
			self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1, training = True, epsilon = 1e-5, name="batch_norm1")
			self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
			# output: [20,20,32]

			# Conv_layer2
			self.conv2 = tf.layers.conv2d(intput = self.inputs, filters=64, kernel_size=[4,4], strides=[2,2], padding="VALID",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv2')
			self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2, training = True, epsilon = 1e-5, name="batch_norm2")
			self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
			# output: [9,9,64]

			# Conv_layer3
			self.conv3 = tf.layers.conv2d(intput = self.inputs, filters=128, kernel_size=[4,4], strides=[2,2], padding="VALID",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), name = 'conv3')
			self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3, training = True, epsilon = 1e-5, name="batch_norm3")
			self.conv3_out = tf.nn.elu(self.conv2_batchnorm, name="conv3_out")
			# output: [3,3,128]

			# Flatten
			self.flatten = tf.layers.flatten(self.conv3_out)

			# Fully Connected
			self.fc = tf.layers.dense(inputs = self.flatten, units = 512, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc1')

			# Output layer (output: 3. One Q-value for each actions)
			self.output = tf.layers.dense(input = self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(), unit=3, activation=None)

			# Q-Value
			self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)

			# Loss : Sum(Qtarget - Q)^2
			self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

			# Optimizer
			self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)












