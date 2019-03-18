import tensorflow as tf  
from ai_image_preprocess import preprocess
import ai_state as state_util
import numpy as np
import ai_state as state_util
from ai_logger import Logger
import time
from file_handler import FileHandler
import matplotlib.pyplot as plt
from ai_agent import Agent
import gym
import os 

# Perform step in environment, in this case
# perform the same action twice
def step(env, action, state):
    next_frame_1, reward_1, done_1, _ = env.step(action)
    next_frame_2, reward_2, done_2, _ = env.step(action)
    next_state = state_util.update(state, preprocess(next_frame_1), preprocess(next_frame_2))
    return (next_state, int(reward_1 + reward_2), done_1 or done_2)

# Save model and parameters, is used to be able
# to load a model and continue training it in 
# different steps.
def save(model_name, agent, steps):
    print("Saving backup, please don't interrupt")  
    epsilon = agent.epsilon
    epsilon_decay = agent.epsilon_decay
    file_handler.write_to_file(epsilon_decay, steps, epsilon)
    agent.save_model(model_name)
    print("Model has been saved")

# Calculations dimensions that are used for 
# the input/output of neural network.
# Based upon the dimensions of the states and 
# actions available in the environment
def calc_dimensions(env):
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    height = obs_shape[0]//2; width = obs_shape[1]//2; n_frames = 4
    state_shape = (height, width, n_frames)
    return (state_shape, n_actions)

# Train the model
def train(env, agent, n_episodes=100000, model_name="model.h5", save_interval=25):
    # Used to keep track of progress
    logger = Logger(10, "episode | states | score | step time | epi time | epsilon")
    
    # Backup every 28 episodes
    backup_save_interval = 28

    # If there is a model with model name, load it
    # and load the parameters
    # Otherwise, create a new model and initialize
    # parameters 
    # Parameters:
    #   epsilon, epsilon_decay, number of steps
    if os.path.isfile(model_name):
        agent.load_model(model_name)
        steps = file_handler.steps
        agent.epsilon = file_handler.epsilon
        agent.epsilon_decay = file_handler.decay_rate
        epsilon_decay = file_handler.decay_rate
    else:
        agent.new_model()
        epsilon = agent.epsilon
        epsilon_decay = agent.epsilon_decay
        steps = 0
        file_handler.write_to_file(epsilon_decay, steps, epsilon)

    # Run the environment so data can be gathered
    # to train the model
    for episode in range(n_episodes):
        
        # Reset environment and variables
        # at the start of the episode
        frame = env.reset()
        state = state_util.create(preprocess(frame))
        score = 0
        
        start_time = time.time()
        done = False
        t = 0
        
        # Interacts with environment and saves the data in the agent
        # The data saved is: 
        # state (what did the state look like)
        # action (what action was performed)
        # next_state (what did the state end up as when action was performed)
        # reward (what was the reward for the action)
        # done (did we run out of lives)
        while not done:
            # env.render() uncomment to render the environment
            t+=1
            action = agent.act(state)
            next_state, reward, done = step(env, action, state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
        steps += 1
        
        # Train agent on 128 different steps in the previous episode
        agent.replay(batch_size=128, score=score, epsilon=agent.epsilon)   
        
        # Keep track of how long time it takes to run an episode
        # and train the agent
        duration = time.time() - start_time
        print("{:>7d} | {:>6d} | {:>5d} | {:>9.5f} | {:>8.5f} | {:>7.5f}"
               .format(episode+1, t, score, duration/t, duration, agent.epsilon))
        print(np.min(agent.q), np.max(agent.q))

        # Save current parameters and model      
        if episode % save_interval == 0:
            save(model_name, agent, steps)
        elif episode % backup_save_interval == 0:
        # Added a backup in case main model is corrupted
            save("backup2.h5", agent, steps)

    # Save model after we have run through all the episodes
    # one last time
    save(model_name, agent, steps)
        
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
session = tf.Session(config=config)  

file_handler = FileHandler()
env = gym.make("BreakoutDeterministic-v4")
state_shape, n_actions = calc_dimensions(env)

agent = Agent(state_shape, n_actions)
model_name = "test.h5"
train(env, agent, model_name=model_name)

