# Import the gym module
import gym
import numpy as np
import keras
import random
import collections
import matplotlib.pyplot as plt

# TODO Change environment to Breakout-v0 and implement frame skipping
# TODO Research Hubber Loss
# TODO read up on eps-greedy
# TODO Check how states should be constructed, with or without overlap

'''Constants'''
ACTIONS = [0, 1, 2, 3] # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
ACTIONS_encoded = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                   [0, 0, 0, 1]]
N_ACTIONS = 4
REPLAY_START_SIZE = 20000
STATE_SIZE = 4

'''Training params'''
ITERATIONS = 1000000
EPS = 1
EPS_SUBTRACT = 1e-6
#EPS_SUBTRACT = 0.01
MEMORY_SIZE = 200000
BATCH_SIZE = 32
GAMMA = 0.99

"Game stats"
SLOW_DOWN_RATE = 1000000

"Plot Params"
ITERATIONS_BEFORE_BENCHMARKING = 100000
TEST_STEPS = 10000

################################################################
'''Pre-processing Functions'''


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


###############################################################

'''def transform_reward(reward):
    return np.sign(reward)'''


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
    start_states, actions, rewards, next_states, is_terminal = map(list, zip(*batch))

    # First, predict the Q values of the next states. Note how we are passing ones as the mask.
    #print(np.shape(next_states))
    next_Q_values = model.predict([next_states, np.ones(np.shape(actions))])
    #print(next_Q_values)
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminal] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    Q_values = rewards + GAMMA * np.max(next_Q_values, axis=1)
    # Fit the keras model. Note how we are passing the actions as the mask and multiplying
    # the targets by the actions.
    #print(actions)
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
    filtered_output = keras.layers.multiply([output, actions_input])

    model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    return model

def q_iteration(env, model, start_state, iteration, memory, reward_so_far):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    # Choose the action
    # Choose the action
    #TODO Implement more sophisticated exploitation-exploration trade-off, should maybe be biased towards exploitation later on.
    if random.random() < epsilon:
        action = env.action_space.sample()
        #print("Sampling action")
    else:
        action = choose_best_action(model, start_state)
        #print("Choosing best action")

    # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
    frame, reward, is_done, _ = env.step(action)
    #reward_so_far += reward
    revamp_game(env, is_done)

    start_state_list = list(start_state)
    #start_state_list = np.reshape(start_state_list, (105, 80, 4))
    start_state_list = np.transpose(start_state_list, (1, 2, 0))

    frame = preprocess(frame)
    state = construct_state(start_state, frame)
    #state_list = list(state)
    #print(np.shape(state_list))
    #state_list = np.reshape(state_list, (105, 80, 4))
    state_list = np.transpose(start_state, (1, 2, 0))
    #print(np.shape(state_list))

    action = ACTIONS_encoded[action]
    element = start_state_list, action, reward, state_list, is_done
    memory.append(element)
    # Sample and fit
    batch = memory_sample(memory)
    fit_batch(model, batch)
    return state


def get_epsilon_for_iteration(iteration):
    eps = EPS - iteration*EPS_SUBTRACT
    if eps > 0.1:
        return eps
    return 0.1

def choose_best_action(model, state):
    state_list = list(state)
    #print(np.shape(state_list))
    state_list = np.transpose(state_list, (1, 2, 0))
    #best_action_index = np.argmax(model.predict([np.reshape(state_list, (1, 105, 80, 4)), np.ones((1, N_ACTIONS))]))
    #print(model.predict([np.reshape(state_list, (1, 105, 80, 4)), np.ones((1, N_ACTIONS))]))
    best_action_index = np.argmax(model.predict([np.reshape(state_list, (1, 105, 80, 4)), np.ones((1, N_ACTIONS))]))
    #print(best_action_index)
    return best_action_index

def memory_sample(memory):
    return random.sample(memory, BATCH_SIZE)

def train_model(env, model, state, memory):
    i = 0
    reward_so_far = 0
    reward_averages = []
    print("_____________________________Starting Training________________________________________")
    while ITERATIONS > i:
        state = q_iteration(env, model, state, i, memory, reward_so_far)

        if i % ITERATIONS_BEFORE_BENCHMARKING == 0:
            for _ in range(TEST_STEPS):
                reward, state = test_model(env, model, state)
                reward_so_far += reward

            reward_averages.append(reward_so_far/TEST_STEPS)
            print("Iteration -> ", i)

        #print("Iteration -> ", i)
        i += 1

    return reward_averages

def test_model(env, model, start_state):
    action = choose_best_action(model, start_state)
    frame, reward, is_done, _ = env.step(action)
    revamp_game(env, is_done)
    frame = preprocess(frame)
    state = construct_state(start_state, frame)
    return reward, state

def init_test_environment():
    # Create a breakout environment
    env = gym.make('Breakout-v0')
    # Reset it, returns the starting frame
    frame = env.reset()
    print(env.action_space)
    print(env.unwrapped.get_action_meanings())
    print(env.reward_range)
    # print(np.shape(frame))
    # Render
    #env.render()
    return env

def run_training():
    env = init_test_environment()
    model = atari_model(N_ACTIONS)

    # memory = RingBuf(10000)
    memory = collections.deque([], MEMORY_SIZE)
    state = fill_up_memory(env, memory)
    reward_averages = train_model(env, model, state, memory)
    model.save('BreakoutModel_basic.model')

    random_reward_average = get_random_reward_average()
    plot_reward_per_epoch(reward_averages, random_reward_average)

def plot_reward_per_epoch(reward_averages, random_reward_average):
    print("Number of Epochs:", ITERATIONS/ITERATIONS_BEFORE_BENCHMARKING)
    data_size = np.size(reward_averages)
    random_reward_averages = np.full(data_size, random_reward_average)
    plt.plot(np.arange(data_size), np.array(reward_averages), color='blue', label='Model')
    plt.plot(np.arange(data_size), random_reward_averages, color='red', label='Random')

    xMin = 0
    xMax = data_size

    concat_list = np.concatenate((reward_averages, random_reward_averages))
    yMin = np.min(concat_list)
    yMax = np.max(concat_list)

    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    plt.xlabel('Epoch')
    plt.ylabel('Average Reward per Time Step')
    plt.legend()
    plt.show()
    plt.savefig('RewardPlot.png')


def run_train_existing_model(model_path):
    env = init_test_environment()
    model = atari_model(N_ACTIONS)
    model.load_weights(model_path)

    # memory = RingBuf(10000)
    memory = collections.deque([], MEMORY_SIZE)
    state = fill_up_memory(env, memory)
    train_model(env, model, state, memory)
    model.save('BreakoutModel_basic_200k.model')

def run_model(model_path, slow_down):
    env = init_test_environment()
    model = atari_model(N_ACTIONS)
    model.load_weights(model_path)
    run_game_with_model(env, model, slow_down)

def run_game_with_model(env, model, slow_down):
    nr_games = 1
    tot_reward = 0
    high_score = tot_reward
    state, _, _ = init_state(env, env.action_space.sample())
    # Perform a random action, returns the new frame, reward and whether the game is over
    #print(np.shape(state))
    while True:
        action = choose_best_action(model, state)
        frame, reward, is_done, info = env.step(action)
        print(reward)
        tot_reward += reward
        if is_done:
            tot_reward, high_score, nr_games = note_game(tot_reward, high_score, nr_games)
        revamp_game(env, is_done)
        env.render()
        frame = preprocess(frame)
        state.append(frame)
        if slow_down:
            apply_slow_down_game()


def note_game(tot_reward, high_score, nr_games):
    if tot_reward > high_score:
        high_score = tot_reward
    print("Game nr: ", nr_games)
    print("Score ->", tot_reward)
    print("Current High Score ->", high_score)
    tot_reward = 0
    nr_games += 1
    return tot_reward, high_score, nr_games

def apply_slow_down_game():
    i = 0
    while i < SLOW_DOWN_RATE:
        i += 1

def fill_up_memory(env, memory):
    action = env.action_space.sample()
    start_state, _, _ = init_state(env, action)
    state = 0
    i = 0
    print("_____________________________Prepare Replay Memory________________________________________")
    while i < REPLAY_START_SIZE:
        # Perform a random action, returns the new frame, reward and whether the game is over
        action = env.action_space.sample()
        frame, reward, is_done, _ = env.step(action)
        revamp_game(env, is_done)

        start_state_list = list(start_state)
        #start_state_list = np.reshape(start_state_list, (105, 80, 4))
        start_state_list = np.transpose(start_state_list, (1, 2, 0)) # (4, 105, 80) -> (105, 80, 4)

        frame = preprocess(frame)
        state = construct_state(start_state, frame)

        state_list = list(state)
        #state_list = np.reshape(state_list, (105, 80, 4))
        state_list = np.transpose(start_state, (1, 2, 0))

        action = ACTIONS_encoded[action]
        element = start_state_list, action, reward, state_list, is_done
        memory.append(element)
        i += 1
        start_state = state
    return state

def revamp_game(env, is_done):
    if is_done:
        #print("RESTARTING GAME")
        env.reset()
    #env.render()


def init_state(env, action):
    state = collections.deque([], STATE_SIZE)
    reward = 0
    is_done = False
    while state.__len__() < STATE_SIZE:
        new_frame, reward, is_done, _ = env.step(action)
        revamp_game(env, is_done)
        new_frame = preprocess(new_frame)
        state.append(new_frame)
        #print(np.shape(list(state)))
    #state = np.reshape(state, (105, 80, 4))
    return state, reward, is_done

def construct_state(state, frame):
    state = collections.deque(list(state), STATE_SIZE)
    state.append(frame)
    return state

def get_random_reward_average():
    env = init_test_environment()
    revard_average = 0
    i = 0
    while i < TEST_STEPS:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        revard_average += reward
        revamp_game(env, is_done)
        # Render
        #env.render()
        i += 1
    return revard_average / ITERATIONS_BEFORE_BENCHMARKING

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
    memory.append([1, 2, 3])
    print(memory.__len__())
    memory.append([4, 5, 6])
    print(memory.__len__())
    memory.append([7, 8, 9])
    print(memory.__len__())
    memory.append([10, 11, 12])
    print(memory.__len__())
    memory.append([13, 14, 15])
    print(memory.__len__())
    memory.append([16, 17, 18])
    print(memory.__len__())
    print(memory.popleft()) # (4, 5, 6) is expected
    #print(memory.pop()) # (16, 17, 18) is expected

    print(np.shape(list(memory)))

    '''memoryCopy = copy.copy(memory)
    print("Copy: ", memoryCopy.__len__())
    print("Original: ", memory.__len__())
    memory.popleft()
    print("Copy: ", memoryCopy.__len__())
    print("Original: ", memory.__len__())
    memory.popleft()
    print("Copy: ", memoryCopy.__len__())
    print("Original: ", memory.__len__())'''

    #indeces = random.randint(5, 2)
    #print(memory.__getitem__(indeces))
    #print(random.sample(memory, 2))

    '''batch = random.sample(memory, 2)
    print("Batch: ", batch)
    first, second, last = map(list, zip(*batch))
    print("First: ", first)
    print("Second: ", second)
    print("Last: ", last)'''

def gym_output_test():
    # Create a breakout environment
    env = gym.make('Breakout-v0')
    # Reset it, returns the starting frame
    frame = env.reset()


    while True:
        print(env.action_space.sample())


def test_array_transform():
    array = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

    print(array)
    print(np.reshape(array, (2, 2, 3)))


if __name__ == '__main__':
    run_training()
    #run_train_existing_model("BreakoutModel_basic_200k.model")

    #run_model("BreakoutModel_basic_200k.model", slow_down=False)
    #run_model("BreakoutModel_basic.model", slow_down=False)
    #run_model("BreakoutModel_basic_200000Iterations.model", slow_down=False)


################################################################
    #Tests:

    #run_random()
    #memory_test()
    #gym_output_test()
    #test_array_transform()
