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


def get_gradient(s, weights):
    """
    Given inputs s and weights, return the gradient of v with respect to the weights
    """

    ### Compute the gradient of the value function with respect to W0, b0, W1, b1 for input s
    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(0, psi)
    grads[0]['W'] = my_matmul(s.T, weights[1]["W"].T * (x > 0))
    grads[0]['b'] = weights[1]["W"].T * (x > 0)
    grads[1]['W'] = x.T
    grads[1]['b'] = 1


# ---------------------------------------------------------------------------------------------------------------------#
# stochastic gradient descent method for state-value prediction
#
# At each time step, we update the weights in the direction g_t = D_t * Del * v_hat(S_t, w_t) using a fixed setp-size alpha.
# D_t = R_t+1 + gamma * v_hat(S_t+1, w_t) - v_hat(S_t, w_t) is the TD-error.
# Del * v_hat(S_t, w_t) is the gradient of the value function with respect to the weights.
#
# The weights are structured as an array of dictionaries. Note that the updates g_t, in the case of TD, is
# D_t * Del * v_hat(S_t, w_t).
# As a result, g_t has the same structure as Del * v_hat(S_t, w_t) which is also an array of dictionaries.

class SGD(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """ Setup for the opterator.

        Set parameters needed to setup the stochastid gradient descent metod.

        Assume optimizer_info dict contains:
        {
            step_size: float
        }
        """
        self.step_size = optimizer_info.get("step_szie")

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                weights[i][param] += self.step_size * g[i][param]

        return weights


# ---------------------------------------------------------------------------------------------------------------------#
# Adam Algoritmn
#
# Instead of using SGD for updating the weights, we use a more advanced algorithm called Adam.
# The Adam algorithm improves the SGD update with two concepts: adaptive vecotr step-sizes and momentum.
# It keeps estimates of the mean and second moment of the updates, denoted by m and v respectively:
# m_t = beta_m * m_t-1 + (1 - beta_m) * g_t
# v_t = beta_v * v_t-1 + (1 - beta_v) * (g_t)^2
#
# Given that m and v are initialized to zero, they are biased toward zero. To get unbiased estiamtes of the mean and
# second moment, Adam defines m_hat and v_hat as:
# m_t_hat = m_t / (1 - beta_t_m)
# v_t_hat = v_t / (1 - beta_t_v)
# The weights are then updated as follows:
# w_t = w_t-1 + ( alpha / (sqrt(v_t_hat) + epsilon) ) * m_t_hat
#
# When implmenting the agent we use the Adam algorithm instead of SGD because it is more efficient.

class Adam(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """Setup for the optimizer.

        Set parameters needed to setup the Adam algorithm.

        Assume optimizer_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
        }
        """

        self.num_states = optimizer_info.get("num_states")
        self.num_hidden_layer = optimizer_info.get("num_hidden_layer")
        self.num_hidden_units = optimizer_info.get("num_hidden_units")

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(self.num_hidden_layer+1)]
        self.v = [dict() for i in range(self.num_hidden_layer+1)]

        for i in range(self.num_hidden_layer+1):

            # Initialize self.m[i]["W"], self.m[i]["b"], self.v[i]["W"], self.v[i]["b"] to zero
            self.m[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i+1]))
            self.v[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i+1]))

        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """

        for i in range(len(weights)):
            for param in weights[i].keys():

                ### update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * g[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * (g[i][param] * g[i][param])

                ### compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                ### update weights
                weights[i][param] += self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights



# ---------------------------------------------------------------------------------------------------------------------#
# Implement agent methods
# In agent_init(), the following is initialized:
# - specify the neural network structure by filling self.layer_size with the size of the input layer, hidden layer, and output layer.
# - initialize the network's parameters. We show the parameters as an array of dictionaries, self.weights, where each
#   dictionary corresponds to weights from one layer to the next. Each dictionary includes w and b. To intialize the paramters,
#   we use a normal distribution with mean 0 and standard deviation sqrt( 2 / #_of_each_node ). This initialization heuristic
#   is commonly used when using ReLU gates and helps keep the output of a neuron from getting too big or too small.
#   to intialize the network's parameters, use self.rand_generator.normal() which draws random samples from a normla distribution.
#   The parameters of self.rand_generator.normal are mean of the distribution, standard deviation of the distribution, and
#   output shape in the form of tuple of integers.
#
# In agent_start() we specify self.last_state and self.last_action.
#
# In agent_step() and agent_end(), we:
# - compute the TD error using v(S_t) and v(S_t+1). To compute the value function for S_t and S_t+1, we get their one-hot
#   encoding using one_hot() method. We feed the one-hot encoded state number to the neural networks using get_value() method
#   implemented above. Note that one_hot() method returns the one-hot encoding of a state as a numpy array of shape (1, num_states).
# - retrieve the gradients using get_gradient() function.
# - use Adam_algorithm to update the neural network parameters, self.weights.
# - use agent_policy() method to select actions with agent_step().

def one_hot(state, num_states):
    """
    Given num_state and a state, return the one-hot encoding of the state
    """
    # Create the one-hot encoding of state
    # one_hot_vector is a numpy array of shape (1, num_states)

    one_hot_vector = np.zeros((1, num_states))
    one_hot_vector[0, int((state - 1))] = 1

    return one_hot_vector

class TDAgent(BaseAgent):
    def __init__(self):
        self.name = 'td_agent'
        pass

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD with a Neural Network.

        Assume agent_info dict contain:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            discount_factor: float,
            self.beta_m: float,
            self.beta_v: float,
            self.epsilon: float,
            seed: integer
        }
        """

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get('seed'))

        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get('seed'))

        # Set attributes according to agent_info
        self.num_states = agent_info.get("num_states")
        self.num_hidden_layer = agent_info.get("num_hidden_layer")
        self.num_hidden_units = agent_info.get("num_hidden_units")
        self.discount_factor = agent_info.get("discount_factor")

        # Define the neural netork's structure
        # Specify self.layer_size which shows the number of nodes in each layer
        self.layer_size = np.array([self.num_states, self.num_hidden_layer, self.num_hidden_units])

        # Intialize the neural network's parameters
        self.weights = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):
            # Intialize self.weights[i]['W'] and self.weights[i]['b'] using self.rand_generator.normal()
            self.weights[i]['W'] = self.rand_generator.normal(0, np.sqrt(2/self.layer_size[i]), self.layer_size[i+1])
            self.weights[i]['b'] = self.rand_generator.normal(0, np.sqrt(2/self.layer_size[i]), self.layer_size[i+1])

        # Specify the optimizer
        self.optimizer = Adam()
        optimizer_info = {"num_states": agent_info["num_states"],
                          "num_hidden_layer": agent_info["num_hidden_layer"],
                          "num_hidden_units": agent_info["num_hidden_units"],
                          "step_size": agent_info["step_size"],
                          "beta_m": agent_info["beta_m"],
                          "beta_v": agent_info["beta_v"],
                          "epsilon": agent_info["epsilon"]}
        self.optimizer.optimizer_init(optimizer_info)

        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):
        # set chosen_action as 0 or 1 with equal probability
        chosen_action = self.policy_rand_generator.choice([0, 1])
        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        # select action given state (using self.agent_policy()), and save current state and action
        self.last_state = state
        self.last_action = self.agent_policy()

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        ### Compute TD error (5 lines)
        # delta = None

        ### START CODE HERE ###
        s_t = one_hot(self.last_state, self.num_states)
        s_t1 = one_hot(state, self.num_states)
        v_hat_t = get_value(s_t, self.weights)
        v_hat_t1 = get_value(s_t1, self.weights)
        delta = reward + (self.discount_factor * v_hat_t1) - v_hat_t
        ### END CODE HERE ###

        ### Retrieve gradients (1 line)
        # grads = None

        ### START CODE HERE ###
        grads = get_gradient(one_hot(self.last_state, self.num_states), self.weights)
        ### END CODE HERE ###

        ### Compute g (1 line)
        g = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):
            for param in self.weights[i].keys():

                # g[i][param] = None
                ### START CODE HERE ###
                g[i][param] = delta * grads[i][param]
                ### END CODE HERE ###

        ### update the weights using self.optimizer (1 line)
        # self.weights = None

        ### START CODE HERE ###
        self.weights = self.update_weights(self.weights, g)
        ### END CODE HERE ###

        ### update self.last_state and self.last_action (2 lines)

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = self.agent_policy()
        ### END CODE HERE ###

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        ### compute TD error (3 lines)
        # delta = None

        ### START CODE HERE ###
        s_t = one_hot(self.last_state, self.num_states)
        v_hat_t = get_value(s_t, self.weights)
        delta = reward - v_hat_t
        ### END CODE HERE ###

        ### Retrieve gradients (1 line)
        # grads = None

        ### START CODE HERE ###
        grads = get_gradient(one_hot(self.last_state, self.num_states), self.weights)
        ### END CODE HERE ###

        ### Compute g (1 line)
        g = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):
            for param in self.weights[i].keys():

                # g[i][param] = None
                ### START CODE HERE ###
                g[i][param] = delta * grads[i][param]
                ### END CODE HERE ###

        ### update the weights using self.optimizer (1 line)
        # self.weights = None

        ### START CODE HERE ###
        self.weights = self.update_weights(self.weights, g)
        ### END CODE HERE ###

    def agent_message(self, message):
        if message == 'get state value':
            state_value = np.zeros(self.num_states)
            for state in range(1, self.num_states + 1):
                s = one_hot(state, self.num_states)
                state_value[state - 1] = get_value(s, self.weights)
            return state_value


# ---------------------------------------------------------------------------------------------------------------------#
# unit tests
## Test Code for get_value() ##

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


## Test Code for get_gradient() ##

# Suppose num_states = 5, num_hidden_layer = 1, and num_hidden_units = 2
num_hidden_layer = 1
s = np.array([[0, 0, 0, 1, 0]])

weights_data = np.load("asserts/get_gradient_weights.npz")
weights = [dict() for i in range(num_hidden_layer+1)]
weights[0]["W"] = weights_data["W0"]
weights[0]["b"] = weights_data["b0"]
weights[1]["W"] = weights_data["W1"]
weights[1]["b"] = weights_data["b1"]

grads = get_gradient(s, weights)

grads_answer = np.load("asserts/get_gradient_grads.npz")

print("grads[0][\"W\"]\n", grads[0]["W"], "\n")
print("grads[0][\"b\"]\n", grads[0]["b"], "\n")
print("grads[1][\"W\"]\n", grads[1]["W"], "\n")
print("grads[1][\"b\"]\n", grads[1]["b"], "\n")

assert(np.allclose(grads[0]["W"], grads_answer["W0"]))
assert(np.allclose(grads[0]["b"], grads_answer["b0"]))
assert(np.allclose(grads[1]["W"], grads_answer["W1"]))
assert(np.allclose(grads[1]["b"], grads_answer["b1"]))

print("Passed the asserts!")


## Test Code for update_weights() ##

# Suppose num_states = 5, num_hidden_layer = 1, and num_hidden_units = 2
num_hidden_layer = 1

weights_data = np.load("asserts/update_weights_weights.npz")
weights = [dict() for i in range(num_hidden_layer+1)]
weights[0]["W"] = weights_data["W0"]
weights[0]["b"] = weights_data["b0"]
weights[1]["W"] = weights_data["W1"]
weights[1]["b"] = weights_data["b1"]

g_data = np.load("asserts/update_weights_g.npz")
g = [dict() for i in range(num_hidden_layer+1)]
g[0]["W"] = g_data["W0"]
g[0]["b"] = g_data["b0"]
g[1]["W"] = g_data["W1"]
g[1]["b"] = g_data["b1"]

test_sgd = SGD()
optimizer_info = {"step_size": 0.3}
test_sgd.optimizer_init(optimizer_info)
updated_weights = test_sgd.update_weights(weights, g)

# updated weights asserts
updated_weights_answer = np.load("asserts/update_weights_updated_weights.npz")

print("updated_weights[0][\"W\"]\n", updated_weights[0]["W"], "\n")
print("updated_weights[0][\"b\"]\n", updated_weights[0]["b"], "\n")
print("updated_weights[1][\"W\"]\n", updated_weights[1]["W"], "\n")
print("updated_weights[1][\"b\"]\n", updated_weights[1]["b"], "\n")

assert(np.allclose(updated_weights[0]["W"], updated_weights_answer["W0"]))
assert(np.allclose(updated_weights[0]["b"], updated_weights_answer["b0"]))
assert(np.allclose(updated_weights[1]["W"], updated_weights_answer["W1"]))
assert(np.allclose(updated_weights[1]["b"], updated_weights_answer["b1"]))

print("Passed the asserts!")


## Test Code for agent_init() ##

agent_info = {"num_states": 5,
              "num_hidden_layer": 1,
              "num_hidden_units": 2,
              "step_size": 0.25,
              "discount_factor": 0.9,
              "beta_m": 0.9,
              "beta_v": 0.99,
              "epsilon": 0.0001,
              "seed": 0
             }

test_agent = TDAgent()
test_agent.agent_init(agent_info)

print("layer_size: {}".format(test_agent.layer_size))
assert(np.allclose(test_agent.layer_size, np.array([agent_info["num_states"],
                                                    agent_info["num_hidden_units"],
                                                    1])))

print("weights[0][\"W\"] shape: {}".format(test_agent.weights[0]["W"].shape))
print("weights[0][\"b\"] shape: {}".format(test_agent.weights[0]["b"].shape))
print("weights[1][\"W\"] shape: {}".format(test_agent.weights[1]["W"].shape))
print("weights[1][\"b\"] shape: {}".format(test_agent.weights[1]["b"].shape), "\n")

assert(test_agent.weights[0]["W"].shape == (agent_info["num_states"], agent_info["num_hidden_units"]))
assert(test_agent.weights[0]["b"].shape == (1, agent_info["num_hidden_units"]))
assert(test_agent.weights[1]["W"].shape == (agent_info["num_hidden_units"], 1))
assert(test_agent.weights[1]["b"].shape == (1, 1))

print("weights[0][\"W\"]\n", (test_agent.weights[0]["W"]), "\n")
print("weights[0][\"b\"]\n", (test_agent.weights[0]["b"]), "\n")
print("weights[1][\"W\"]\n", (test_agent.weights[1]["W"]), "\n")
print("weights[1][\"b\"]\n", (test_agent.weights[1]["b"]), "\n")


agent_weight_answer = np.load("asserts/agent_init_weights_1.npz")
assert(np.allclose(test_agent.weights[0]["W"], agent_weight_answer["W0"]))
assert(np.allclose(test_agent.weights[0]["b"], agent_weight_answer["b0"]))
assert(np.allclose(test_agent.weights[1]["W"], agent_weight_answer["W1"]))
assert(np.allclose(test_agent.weights[1]["b"], agent_weight_answer["b1"]))

print("Passed the asserts!")
