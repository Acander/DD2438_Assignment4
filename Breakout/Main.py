# Import the gym module
import gym
import numpy as np
import tensorflow as tf
import keras
import random
import collections

# TODO Change environment to Breakout-v0 and implement frame skipping
# TODO Research Hubber Loss
# TODO read up on eps-greedy

'''Constants'''
ACTIONS = [0, 3, 4]
N_ACTIONS = 3

'''Training params'''
ITERATIONS = 100000
EPS = 1
EPS_SUBTRACT = 1e-6
MEMORY_SIZE = 100000

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


def fit_batch(model, batch):
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
    gamma, start_states, actions, rewards, next_states, is_terminal = batch

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
    # We assume a tensorflow backend here
    ATARI_SHAPE = (105, 80, 4)

    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.convolutional.Convolution2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.convolutional.Convolution2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = keras.layers.concatenate([output, actions_input])
    #filtered_output = keras.layers.merge([output, actions_input], mode='mul')

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model

def q_iteration(env, model, state, iteration, memory):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)
    # Choose the action
    # Choose the action
    #TODO Implement more sophisticated exploitation-exploration trade-off, should maybe be biased towards exploitation later on.
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    new_frame, reward, is_done, _ = env.step(action)
    element = state, action, new_frame, reward, is_done
    memory.append(element)
    # Sample and fit
    batch = memory.sample_batch(32)
    fit_batch(model, batch)


def get_epsilon_for_iteration(iteration):
    if EPS > 0.1:
        return EPS - EPS_SUBTRACT
    return EPS

def choose_best_action(model, state):
    best_action_index = np.argmax(model.predict([state, state, state], [1, 1, 1]))
    return ACTIONS[best_action_index]


'''class RingBuf:
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

    def sample_batch(self, size):


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
'''

def run():
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    # env.render()
    model = atari_model(N_ACTIONS)

    i = 0
    # memory = RingBuf(10000)
    memory = collections.deque(100000)
    while ITERATIONS > i:
        q_iteration(env, model, frame, i, memory)

    is_done = False
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(choose_best_action(model, frame))
        # Render
        env.render()

def run_random():
    '''is_done = False
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        # Render
        env.render()'''

def memory_test():
    # LIFO
    '''memory = collections.deque([], 5)
    memory.append((1, 2, 3))
    print(memory.__len__())
    memory.append((4, 5, 6))
    print(memory.__len__())
    memory.append((7, 8, 9))
    print(memory.__len__())
    print(memory.pop())
    print(memory.__len__())
    print(memory.pop())
    print(memory.__len__())'''

    # FIFO
    memory = collections.deque([], 5)
    memory.append((1, 2, 3))
    print(memory.__len__())
    memory.append((4, 5, 6))
    print(memory.__len__())
    memory.append((7, 8, 9))
    print(memory.__len__())
    memory.append((10, 11, 12))
    print(memory.__len__())
    memory.append((13, 14, 15))
    print(memory.__len__())
    memory.append((16, 17, 18))
    print(memory.__len__())
    print(memory.popleft()) # (4, 5, 6) is expected
    print(memory.pop()) # (16, 17, 18) is expected

if __name__ == '__main__':
    #run()
    #run_random()

    #memory_test()


