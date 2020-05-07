# Import the gym module
import gym
import numpy as np
import tensorflow as tf
import random

# TODO Change environment to Breakout-v0 and implement frame skipping
# TODO Research Hubber Loss
# TODO read up on eps-greedy

'''Constants'''
ACTIONS = [0, 3, 4]
N_ACTIONS = 3

'''Training params'''
iterations = 100000
eps = 1
eps_subtract = 1e-6

################################################################
'''Pre-processing Functions'''


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


###############################################################

def transform_reward(reward):
    return np.sign(reward)


def fit_batch(model, gamma, start_states, actions, rewards, next_states, is_terminal):
    """Do one deep Q learning iteration.

    Params:
    - model: The DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions corresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminal: numpy boolean array of whether the resulting state is terminal

    """
    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    next_Q_values = model.predict([next_states, np.ones(actions.shape)])
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    model.fit(
        [start_states, actions], actions * Q_values[:, None], nb_epoch=1, batch_size=len(start_states), verbose=0)


def atari_model(n_actions):
    ATARI_SHAPE = (4, 105, 80)

    frames_input = tf.keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = tf.keras.layers.Input((n_actions), name='mask')

    normalized = tf.keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv1 = tf.keras.layers.convolutional.Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(normalized)

    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv2 = tf.keras.layers.convolutional.Convolution2D(32, 4, 4, subsample=(2, 2), actifation='relu')(conv1)

    # Flattening the second convolutional layer.
    conv_flattened = tf.keras.core.Flatten()(conv2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = tf.keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = tf.keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = tf.keras.layers.merge([output, actions_input], mode='mul')

    model = tf.keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = optimizer = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.complie(optimizer, loss='mse')

    return model


def q_iteration(env, model, state, iteration, memory):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)
    # Choose the action
    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_frame, reward, is_done, _ = env.step(action)
    memory.add(state, action, new_frame, reward, is_done)

    # Sample and fit
    batch = memory.sample_batch(32)
    fit_batch(model, batch)


def get_epsilon_for_iteration(iteration):
    if eps > 0.1:
        return eps - eps_subtract
    return eps

def choose_best_action(model, state):
    best_action_index = np.argmax(model.predict([state, state, state], [1, 1, 1]))
    return ACTIONS[best_action_index]


class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == '__main__':
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    #env.render()
    model = atari_model(N_ACTIONS)

    i = 0
    memory = RingBuf(10000)
    while iterations > i:
        q_iteration(env, model, frame, i, memory)

    is_done = False
    while not is_done:
            # Perform a random action, returns the new frame, reward and whether the game is over
            frame, reward, is_done, _ = env.step(choose_best_action(model, frame))
            # Render
            env.render()

    '''is_done = False
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        # Render
        env.render()'''
