import numpy as np 

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # Initializing a state, action np matrix (to zeroes) for storing Q-values 
        self.Q = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state):
        # Implementing epsilon-greedy policy 
        if np.random.rand() < self.epsilon: 
            # explore 
            action = np.random.randint(self.n_actions) 
        else: 
            # exploit 
            action = np.argmax(self.Q[state]) 

        return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        # Implement Q-learning update

        if done: 
            next_r = 0.0 
        else: 
            next_r = self.gamma * np.max(self.Q[next_state])

        td_target = reward + next_r 
        
        td_error = td_target - self.Q[state, action] 

        self.Q[state, action] += self.alpha * td_error 
    
    def train(self, n_episodes, env):
        # Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for episode in range(n_episodes):
            env.reset() 
            total_reward = 0.0 

            while not env.done(): 
                state = env.state() 

                action = self.select_action(state) 
                reward = env.step(action) 

                next_state = env.state() 
                
                done = env.done() 

                self.update(state, action, reward, next_state, done) 
                total_reward += reward 

            episode_returns.append(total_reward) 

        return episode_returns


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # Initialize variables if necessary
        self.Q = np.zeros((self.n_states, self.n_actions)) 

    def select_action(self, state):
        # Implementing epsilon-greedy policy 
        if np.random.rand() < self.epsilon: 
            # explore 
            action = np.random.randint(self.n_actions) 
        else: 
            # exploit 
            action = np.argmax(self.Q[state]) 

        return action

        
    def update(self, state, action, reward, next_state, next_action, done): # Augmenting with next state, next action
        # Implement SARSA update
        if done: 
            next_r = 0.0 
        else: 
            next_r = self.Q[next_state, next_action]  

        td_target = reward + self.gamma * next_r 
        td_error = td_target - self.Q[state, action] 

        self.Q[state, action] += self.alpha * td_error 


    def train(self, n_episodes, env):
        # Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for episode in range(n_episodes):
            env.reset() 
            total_reward = 0.0 

            state = env.state() 
            action = self.select_action(state)

            while not env.done():
                reward = env.step(action) 
                next_state = env.state() 

                done = env.done() 

                if done: 
                    next_action = None 
                else: 
                    next_action = self.select_action(next_state) 

                total_reward += reward 

                self.update(state, action, reward, next_state, next_action, done) 
                state = next_state 
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
        # Initialize variables if necessary
        self.Q = np.zeros((self.n_states, self.n_actions)) 
        
    def select_action(self, state):
        # Implementing epsilon-greedy policy 
        if np.random.rand() < self.epsilon: 
            # explore 
            action = np.random.randint(self.n_actions) 
        else: 
            # exploit 
            action = np.argmax(self.Q[state]) 

        return action

    def update(self, state, action, reward, next_state, done): # Augmenting with next state 
        if done: 
            expected_next = 0.0 
        else: 
            q_next = self.Q[next_state] 
            probabilities = np.ones(self.n_actions) * self.epsilon / self.n_actions 

            best_action = np.argmax(q_next) 
            probabilities[best_action] += 1.0 - self.epsilon 

            expected_next = np.dot(probabilities, q_next) 


        td_target = reward + self.gamma * expected_next 

        td_error = td_target - self.Q[state, action] 

        self.Q[state, action] += self.alpha * td_error 

    def train(self, n_episodes, env):
        # Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for episode in range(n_episodes):
            env.reset() 
            total_reward = 0.0 

            while not env.done(): 
                state = env.state() 
                action = self.select_action(state) 
                
                reward = env.step(action) 
                
                total_reward += reward 
                next_state = env.state() 
                done = env.done() 

                self.update(state, action, reward, next_state, done) 
            
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
        # Initialize variables if necessary
        self.Q = np.zeros((self.n_states, self.n_actions))

     
    def select_action(self, state):
        # Implementing epsilon-greedy policy 
        if np.random.rand() < self.epsilon: 
            # explore 
            action = np.random.randint(self.n_actions) 
        else: 
            # exploit 
            action = np.argmax(self.Q[state]) 

        return action

    def update(self, states, actions, rewards, tau, done): # Augmenting with tau to match with pseudocode. T = done 
        # Implement n-step SARSA update
        # according to Sutton & Barto (p. 147) pseudocode example for n-step SARSA 
        # tau is the time index of the state-action pair we're updating 
        # done = T meaning the timestep of the terminal state
        G = 0.0 

        for reward_idx in range(tau + 1, min(tau + self.n, done) + 1):
            G += (self.gamma ** (reward_idx - tau - 1)) * rewards[reward_idx] 

        if tau + self.n < done: 
            G += (self.gamma ** self.n) * self.Q[states[tau + self.n], actions[tau + self.n]]

        state_tau = states[tau] 
        action_tau = actions[tau] 
        self.Q[state_tau, action_tau] += self.alpha * (G - self.Q[state_tau, action_tau]) 


    def train(self, n_episodes, env):
        # Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []

        for episode in range(n_episodes):
            env.reset()
            total_reward = 0.0 

            states = [env.state()] 
            actions = [self.select_action(states[0])] 

            # placeholder reward at R_0 
            rewards = [0.0] 

            t = 0 
            done = float("inf") 

            while True: 
                if t < done: 
                    reward = env.step(actions[t]) 
                    total_reward += reward 
                    rewards.append(reward)

                    next_state = env.state() 
                    states.append(next_state) 

                    if env.done():
                        done = t + 1 
                    else: 
                        next_action = self.select_action(next_state) 
                        actions.append(next_action) 

                # delay tau update to ensure we have n rewards to update according to the n-step process
                tau = t - self.n + 1 

                if tau >= 0: 
                    self.update(states, actions, rewards, tau, done) 

                # break when all state-action pairs have been done 
                if tau == done - 1: 
                    break 

                t += 1 

            episode_returns.append(total_reward) 

        return episode_returns  
    
    
    