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
	# Step 1: Grayscale
	# Grayscale can be achieve 



