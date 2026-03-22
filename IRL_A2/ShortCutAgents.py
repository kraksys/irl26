import numpy as np
from tqdm import tqdm



class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((self.n_states, self.n_actions))

        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, done, next_state): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        if done:
            next_r = 0
        else:
            next_r = self.gamma * np.max(self.Q[next_state])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + next_r - self.Q[state][action])
    
    def train(self, n_episodes, env):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the cumulative reward (=return) per episode

        episode_returns = []
        for episode in tqdm(range(n_episodes), desc="Training Agent"):
            total_reward = 0
            env.reset()

            while not env.done():
                action = self.select_action(env.state())

                cur_state = env.state()

                reward = env.step(action)
                total_reward += reward

                next_state = env.state()
                self.update(cur_state, action, reward, env.done(), next_state)

            episode_returns.append(total_reward)

        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((self.n_states, self.n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, done, next_state, next_action): # Augment arguments if necessary
        # TO DO: Implement SARSA update
        if done:
            next_r = 0
        else:
            next_r = self.gamma * self.Q[next_state][next_action]

        self.Q[state][action] = self.Q[state][action] +  self.alpha * (reward + next_r - self.Q[state][action])

    def train(self, n_episodes, env):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            total_reward = 0
            env.reset()
            cur_state = env.state()
            action = self.select_action(cur_state)

            while not env.done():

                reward = env.step(action)
                total_reward += reward

                next_state = env.state()
                next_action = self.select_action(next_state)
                self.update(cur_state, action, reward, env.done(), next_state, next_action)

                cur_state = next_state
                action = next_action

            episode_returns.append(total_reward)

        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        if done:
            next_r = 0
        else:
            # this is the array of rewards for each action
            next = self.Q[next_state]
            best_action = np.argmax(next)

            # There is a different implementation of teh expected sarsa where we
            # need to distribute the epsilon among the rest of available actions that are not the best one,
            # but here we can just take the average which is basically we give the best action a bit of extra
            expected_reward = np.mean(next) * self.epsilon

            # adding the rest (1 - epsilon) to the best action
            expected_reward += next[best_action] * (1 - self.epsilon)

            next_r = self.gamma * expected_reward

        self.Q[state][action] += self.alpha * (reward + next_r - self.Q[state][action])

    def train(self, n_episodes, env):
        # TO DO: Implement the agent loop that trains for n_episodes.
        # Return a vector with the cumulative reward (=return) per episode


        #same as the Q one I just copied it
        # only one pair of arguments are switched
        episode_returns = []
        for episode in range(n_episodes):
            total_reward = 0
            env.reset()

            while not env.done():
                action = self.select_action(env.state())

                cur_state = env.state()

                reward = env.step(action)
                total_reward += reward

                next_state = env.state()
                self.update(cur_state, action, reward, next_state, env.done())

            episode_returns.append(total_reward)

        return episode_returns


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # TO DO: Implement policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns  
    
    
    