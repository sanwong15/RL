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

		





