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

# Reset Graph
tf.reset_default_graph()

DQNetwork = DQNetwork(state_size, action_size, learning_rate)

class Memory():
	def __init__(self, max_size):
		self.buffer = deque(maxlen= max_size)

	def add(self, experience):
		self.buffer.append(experience)

	def sample(self, batch_size):
		buffer_size = len(self.buffer)
		index = np.random.choice(np.arrange(buffer_size), size=batch_size, replace=False)

		return [self.buffer[i] for i in index]

memory = Memory(max_size= memory_size)

# Render new Env
game.new_episode()

for i in range(pretrain_length):
	if i == 0:
		# For the first step
		# Need a state
		state = game.get_state().screen_buffer
		state, stacked_frames = stack_frames(stacked_frames, state, True)

	# Random Action
	action = random.choice(possible_actions)

	# Get Rewards
	reward = game.make_action(action)

	# Check if episode is finished
	done = game.is_episode_finished

	if done:
		# Episode finished
		next_state = np.zeros(state.shape)

		# Add experience to memory
		memory.add((state, action, reward, next_state, done))

		# Start a new episode
		game.new_episode()

		# get state
		state = game.get_state().screen_buffer

		state, stacked_frames = stack_frames(stacked_frames, state, True)
	else:
		# Episode not finished
		next_state = game.get_state().screen_buffer
		next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

		# Add experience to memory
		memory.add((state, action, reward, next_state, done))

		# Update state
		state = next_state


# Set up Tensorboard
# set up Tensorboard writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

# Losses
tf.summary.scalar("Loss", DQNetwork.loss)

# write output
write_op = tf.summary.merge_all()


# Train Agent
'''
(1) Init weight
(2) Init Env
(3) Init Decay rate

FOR-LOOP (Episode) for each episode
Make new episode
Set step to ZERO
Observe the first state s_0

While-Loop (while below max_steps):
Increase decay_rate
With prob epsilon: select a random action a_t, with prob (1-epsilon) select a_t = argmax_a Q(s_t, a)
Execute action a_t, observe reward r_t+1 and get new state s_t+1
Store Transition (to Experience Buffer)
Sample random mini-batch
Set Predicted_next_state_Q_value = r (terminate state: episode ends) OR Predicted_next_state_Q_value = r + Decay_rate(max_a: Q(next_state, next_state_all_possible_action)
Make Gradient Descent Step with Loss: (Predict_next_state_Q_value - Current_state_Q_value) power to 2


END-WHILE

END-FOR
'''

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
	# Epsilon Greedy
	# Random Number
	exp_exp_tradeoff = np.random.rand()

	explore_prob = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*decay_step)

	if (explore_prob > exp_exp_tradeoff):
		# Make random action (exploration)
		action = random.choice(possible_actions)

	else:
		# Get action from Q-network (exploitation)
		# Estimate the Qs value state
		Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))})
		# Take Biggest Q value -> Best action
		choice = np.argmax(Qs)
		action = possible_actions[int(choice)]


	return action, explore_prob

# Training and Saving
# Saver will help us to save our model
saver = tf.train.Saver()


if training == True:
	with tf.Session() as sess:
		# Init the variables
		sess.run(tf.global_variables_initializer())

		# Init decay rate
		decay_step = 0

		game.init()


		for episode in range(total_episodes):
			# set step to 0
			step = 0

			episode_rewards = [] # Init rewards of the episode

			# New episode
			game.new_episode()
			# Get state
			state = game.get_state().screen_buffer

			# Stack Frame
			state, stacked_frames = stack_frames(stacked_frames, state, True)

			while step < max_steps:
				# Update step
				step += 1

				# Increase decay_step
				decay_step += 1


				# Predict action to take and exe action
				action, explore_prob = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

				# Exe the action
				reward = game.make_action(action)

				# Check if episode is finished
				done = game.is_episode_finished()

				# Append rewards
				episode_rewards.append(reward)

				# If game finished
				if done:
					next_state = np.zeros((84,84), dtype=np.int)
					next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

					# set step to max_steps to end episode
					step = max_steps

					# Get total_reward
					total_reward = np.sum(episode_rewards)

					print('Episode: {}'.format(episode), 'Total reward: {}'.format(total_reward), 'Training loss: {:.4f}'.format(loss), 'Explore P: {:.4f}'.format(explore_prob))

					memory.add((state, action, reward, next_state, done))

				else:
					














