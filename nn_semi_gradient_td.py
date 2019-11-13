# coding=utf-8
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import os, shutil
from tqdm import tqdm

from rl_glue import RLGlue
from environment import BaseEnvironment
from agent import BaseAgent
from optimizer import BaseOptimizer
import plot_script
from randomwalk_environment import RandomWalkEnvironment


# ---------------------------------------------------------------------------------------------------------------------#
# Implementation of the 500-State RandomWalk Environment
# Once the agent chooses which direction to move, the environment determines how far the agent is moved in that
# direction. Assume the agent passes either 0 (indicating left) or 1 (indicating right) to the environment.

class RandomWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.

        Set parameters needed to setup the 500-state random walk environment.

        Assume env_info dict contains:
        {
            num_states: 500 [int],
            start_state: 250 [int],
            left_terminal_state: 0 [int],
            right_terminal_state: 501 [int],
            seed: int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(env_info.get("seed"))

        # set each class attribute
        self.num_states = env_info["num_states"]
        self.start_state = env_info["start_state"]
        self.left_terminal_state = env_info["left_terminal_state"]
        self.right_terminal_state = env_info["right_terminal_state"]

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """

        # set self.reward_state_term tuple
        reward = 0.0
        state = self.start_state
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        # return first state from the environment
        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        last_state = self.reward_state_term[1]

        # set reward, current_state, and is_terminal
        #
        # action: specifies direction of movement - 0 (indicating left) or 1 (indicating right)  [int]
        # current state: next state after taking action from the last state [int]
        # reward: -1 if terminated left, 1 if terminated right, 0 otherwise [float]
        # is_terminal: indicates whether the episode terminated [boolean]
        #
        # Given action (direction of movement), determine how much to move in that direction from last_state
        # All transitions beyond the terminal state are absorbed into the terminal state.

        if action == 0:  # left
            current_state = max(self.left_terminal_state, last_state + self.rand_generator.choice(range(-100, 0)))
        elif action == 1:  # right
            current_state = min(self.right_terminal_state, last_state + self.rand_generator.choice(range(1, 101)))
        else:
            raise ValueError("Wrong action value")

        # terminate left
        if current_state == self.left_terminal_state:
            reward = -1.0
            is_terminal = True

        # terminate right
        elif current_state == self.right_terminal_state:
            reward = 1.0
            is_terminal = True

        else:
            reward = 0.0
            is_terminal = False

        self.reward_state_term = (reward, current_state, is_terminal)

        return self.reward_state_term


# ---------------------------------------------------------------------------------------------------------------------#
# helper functions

def agent_policy(rand_generator, state):
    """
    Given random number generator and state, returns an action according to the agent's policy.

    Args:
        rand_generator: Random number generator

    Returns:
        chosen action [int]
    """

    # set chosen_action as 0 or 1 with equal probability
    # state is unnecessary for this agent policy
    chosen_action = rand_generator.choice([0, 1])

    return chosen_action


def get_state_feature(num_states_in_group, num_groups, state):
    """
    Given state, return the feature of that state

    Args:
        num_states_in_group [int]
        num_groups [int]
        state [int] : 1~500

    Returns:
        one_hot_vector [numpy array]
    """

    ### Generate state feature
    # Create one_hot_vector with size of the num_groups, according to state
    # For simplicity, assume num_states is always perfectly divisible by num_groups
    # Note that states start from index 1, not 0!

    # Example:
    # If num_states = 100, num_states_in_group = 20, num_groups = 5,
    # one_hot_vector would be of size 5.
    # For states 1~20, one_hot_vector would be: [1, 0, 0, 0, 0]

    one_hot_vector = [0] * num_groups
    for k in range(num_groups):
        group_end = num_states_in_group * (k + 1)
        group_start = group_end - num_states_in_group + 1
        if (state <= group_end) & (state >= group_start):
            one_hot_vector[k] = 1
        else:
            one_hot_vector[k] = 0

    return one_hot_vector


def my_matmul(x1, x2):
    """
    Given matrices x1 and x2, return the multiplication of them
    """
    
    result = np.zeros((x1.shape[0], x2.shape[1]))
    x1_non_zero_indices = x1.nonzero()
    if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
        result = x2[x1_non_zero_indices[1], :]
    elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
        result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
    else:
        result = np.matmul(x1, x2)
   
    return result


def get_value(s, weights):
    """
    Compute value of input s given the weights of a neural network
    """
    ### Compute the ouput of the neural network, v, for input s
    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(0, psi)
    v = my_matmul(x, weights[1]["W"]) + weights[1]["b"]    

    return v


# ---------------------------------------------------------------------------------------------------------------------#
# Implement agent methods
# The shape of self.all_state_features numpy array is (num_states, feature_size), with features of states from
# State 1-500. Note that index 0 stores features for State 1 (Features for State 0 does not exist). Use self.all_
# state_features to access each feature vector for a state.
#
# When saving state values in the agent, recall how the state values are represented with linear function approximation.
#
# State Value Representation: v_hat(s,w) = wx^T  where  w  is a weight vector and  x is the feature vector of the state.
#
# When performing TD(0) updates with Linear Function Approximation, recall how we perform semi-gradient TD(0) updates
# using supervised learning.
#
# semi-gradient TD(0) Weight Update Rule:  w_t1 = w_t + alpha * (r_t1 + gamma * v_hat(s_t1, w) − v_hat(s_t, w))
# * delta_v_hat(s_t, w)

# Create TDAgent
class TDAgent(BaseAgent):
    def __init__(self):
        self.num_states = None
        self.num_groups = None
        self.step_size = None
        self.discount_factor = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            num_states: 500 [int],
            num_groups: int,
            step_size: float,
            discount_factor: float,
            seed: int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # set class attributes
        self.num_states = agent_info.get("num_states")
        self.num_groups = agent_info.get("num_groups")
        self.step_size = agent_info.get("step_size")
        self.discount_factor = agent_info.get("discount_factor")

        # pre-compute all observable features
        num_states_in_group = int(self.num_states / self.num_groups)
        self.all_state_features = np.array(
            [get_state_feature(num_states_in_group, self.num_groups, state) for state in range(1, self.num_states + 1)])

        # initialize all weights to zero using numpy array with correct size

        self.weights = np.zeros(self.num_groups)

        self.last_state = None
        self.last_action = None

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            self.last_action [int] : The first action the agent takes.
        """

        # select action given state (using agent_policy), and save current state and action

        self.last_state = state
        self.last_action = agent_policy(self.rand_generator, state)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward [float]: the reward received for taking the last action taken
            state [int]: the state from the environment's step, where the agent ended up after the last step
        Returns:
            self.last_action [int] : The action the agent is taking.
        """

        # get relevant feature
        current_state_feature = self.all_state_features[state - 1]
        last_state_feature = self.all_state_features[self.last_state - 1]

        v_hat_current = np.dot(self.weights, current_state_feature)
        v_hat_last = np.dot(self.weights, last_state_feature)
        self.weights = self.weights + self.step_size * (
                    reward + (self.discount_factor * v_hat_current) - v_hat_last) * last_state_feature
        self.last_state = state
        self.last_action = agent_policy(self.rand_generator, state)

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # get relevant feature
        last_state_feature = self.all_state_features[self.last_state - 1]

        # update weights
        self.weights = self.weights + self.step_size * (
                    reward - np.dot(self.weights, last_state_feature)) * last_state_feature
        return

    def agent_message(self, message):
        # We will implement this method later
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------------------#
# unit tests

# Test Code for agent_policy()
test_rand_generator = np.random.RandomState(99)
state = 250
action_array = []
for i in range(10):
    action_array.append(agent_policy(test_rand_generator, 250))
print('action_array: {}'.format(action_array))

# Test Code for get_state_feature()
# Given that num_states = 10 and num_groups = 5, test get_state_feature()
# There are states 1~10, and the state feature vector would be of size 5.
# Only one element would be active for any state feature vector.
# get_state_feature() should support various values of num_states, num_groups, not just this example
# For simplicity, assume num_states will always be perfectly divisible by num_groups
num_states = 10
num_groups = 5
num_states_in_group = int(num_states / num_groups)

# Test 1st group, state = 1
state = 1
print("1st group: {}".format(get_state_feature(num_states_in_group, num_groups, state)))

# Test 2nd group, state = 3
state = 3
print("2nd group: {}".format(get_state_feature(num_states_in_group, num_groups, state)))

# Test 3rd group, state = 6
state = 6
print("3rd group: {}".format(get_state_feature(num_states_in_group, num_groups, state)))

# Test 4th group, state = 7
state = 7
print("4th group: {}".format(get_state_feature(num_states_in_group, num_groups, state)))

# Test 5th group, state = 10
state = 10
print("5th group: {}".format(get_state_feature(num_states_in_group, num_groups, state)))


# Test Code for get_value()
# Suppose num_states = 5, num_hidden_layer = 1, and num_hidden_units = 10
num_hidden_layer = 1
s = np.array([[0, 0, 0, 1, 0]])

weights_data = np.load("asserts/get_value_weights.npz")
weights = [dict() for i in range(num_hidden_layer+1)]
weights[0]["W"] = weights_data["W0"]
weights[0]["b"] = weights_data["b0"]
weights[1]["W"] = weights_data["W1"]
weights[1]["b"] = weights_data["b1"]

estimated_value = get_value(s, weights)
print ("Estimated value: {}".format(estimated_value))
assert(np.allclose(estimated_value, np.array([[-0.21915705]])))

print ("Passed the assert!")


# Test Code for agent_init()
agent_info = {"num_states": 500,
              "num_groups": 10,
              "step_size": 0.1,
              "discount_factor": 1.0,
              "seed": 1}

test_agent = TDAgent()
test_agent.agent_init(agent_info)

# check attributes
print("num_states: {}".format(test_agent.num_states))
print("num_groups: {}".format(test_agent.num_groups))
print("step_size: {}".format(test_agent.step_size))
print("discount_factor: {}".format(test_agent.discount_factor))

print("weights shape: {}".format(test_agent.weights.shape))
print("weights init. value: {}".format(test_agent.weights))


# Test Code for agent_start() and agent_policy()
agent_info = {"num_states": 500,
              "num_groups": 10,
              "step_size": 0.1,
              "discount_factor": 1.0,
              "seed": 1
             }
state = 250
test_agent = TDAgent()
test_agent.agent_init(agent_info)
test_agent.agent_start(state)

print("Agent state: {}".format(test_agent.last_state))
print("Agent selected action: {}".format(test_agent.last_action))


# Test Code for agent_step()
# Make sure agent_init() and agent_start() are working correctly first.
# agent_step() should work correctly for other arbitrary state transitions in addition to this test case.
agent_info = {"num_states": 500,
              "num_groups": 10,
              "step_size": 0.1,
              "discount_factor": 0.9,
              "seed": 1}
test_agent = TDAgent()
test_agent.agent_init(agent_info)

# Initializing the weights to arbitrary values to verify the correctness of weight update
test_agent.weights = np.array([-1.5, 0.5, 1., -0.5, 1.5, -0.5, 1.5, 0.0, -0.5, -1.0])
print("Initial weights: {}".format(test_agent.weights))

# Assume the agent started at State 50
start_state = 50
action = test_agent.agent_start(start_state)

# Assume the reward was 10.0 and the next state observed was State 120
reward = 10.0
next_state = 120
test_agent.agent_step(reward, next_state)
print("Updated weights: {}".format(test_agent.weights))

if np.allclose(test_agent.weights, np.array([-0.26, 0.5, 1., -0.5, 1.5, -0.5, 1.5, 0., -0.5, -1.])):
    print("weight update is correct!\n")
else:
    print("weight update is incorrect.\n")

print("last state: {}".format(test_agent.last_state))
print("last action: {}".format(test_agent.last_action))


# Test Code for agent_end()
# Make sure agent_init() and agent_start() are working correctly first.

agent_info = {"num_states": 500,
              "num_groups": 10,
              "step_size": 0.1,
              "discount_factor": 0.9,
              "seed": 1}

test_agent = TDAgent()
test_agent.agent_init(agent_info)

# Initializing the weights to arbitrary values to verify the correctness of weight update
test_agent.weights = np.array([-1.5, 0.5, 1., -0.5, 1.5, -0.5, 1.5, 0.0, -0.5, -1.0])
print("Initial weights: {}".format(test_agent.weights))

# Assume the agent started at State 50
start_state = 50
test_agent.agent_start(start_state)

# Assume the reward was 10.0 and reached the terminal state
test_agent.agent_end(10.0)
print("Updated weights: {}".format(test_agent.weights))

if np.allclose(test_agent.weights, np.array([-0.35, 0.5, 1., -0.5, 1.5, -0.5, 1.5, 0., -0.5, -1.])):
    print("weight update is correct!\n")
else:
    print("weight update is incorrect.\n")


