import random
import time
from typing import Dict
import numpy as np
import math
import pygame
from utility import play_q_table
from cat_env import make_env, CatChaseEnv 

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################
def choose_action(state: int, q_table: Dict[int, np.ndarray], exploration_rate: float, n_actions: int) -> int:
    if random.uniform(0, 1) < exploration_rate:
        return random.randint(0, n_actions - 1)
    else:
        return int(np.argmax(q_table[state]))
    
def analyze_reward(reward_tuple: tuple, env : CatChaseEnv)-> int:
    done = reward_tuple[2]
    truncated = reward_tuple[3]

    if done:
        return 1000
    elif truncated and env.cat.current_distance < env.cat.prev_distance:
        return -1 * (env.cat.current_distance / 2)
    elif truncated and env.cat.current_distance > env.cat.prev_distance:
        return -5 * (env.cat.current_distance / 2)
    else:
        return -1


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    learning_rate = 0.5
    discount_factor = 0.99
    exploration_rate = 1.0
    min_learning_rate = 0.1
    min_exploration_rate = 0.01
    learning_decay_rate = 0.999
    exploration_decay_rate = 0.998
    max_steps_per_episode = 60 

    # Changed from 60 to 1000, Kaizen
    max_steps_per_episode = 60
    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        initial_state, info = env.reset()
        state = int(initial_state)
        done = False
        truncated = False

        for step in range(max_steps_per_episode):
            action = choose_action(state, q_table, exploration_rate, env.action_space.n)
            next_state, reward, done, truncated, info = env.step(action)

            reward_tuple = tuple([next_state, reward, done, truncated, info])
            # Manually compute reward
            reward = analyze_reward(reward_tuple, env)

            # Update Q-value using the Q-learning formula
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state][action] = new_value

            state = next_state

            if done or truncated:
                break       
        
        learning_rate = max(min_learning_rate, learning_rate * learning_decay_rate)
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay_rate)


        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table