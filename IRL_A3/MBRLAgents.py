#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""

from queue import PriorityQueue

import numpy as np
from MBRLEnvironment import WindyGridworld


class DynaAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize relevant elements
        self.Q_sa = np.zeros((n_states, n_actions))
        self.transition_counts = np.zeros(
            (n_states, n_actions, n_states)
        )  # counts of (s,a,s')
        self.rewards = np.zeros((n_states, n_actions, n_states))

        self.observed_sa = np.zeros((n_states, n_actions), dtype=bool)
        self.observed_pairs = []

    def select_action(self, s, epsilon):
        # e-greedy action selection
        eps = np.random.random()
        if eps < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):

        self.transition_counts[s, a, s_next] += 1
        self.rewards[s, a, s_next] += r

        if not self.observed_sa[s, a]:
            self.observed_sa[s, a] = True
            self.observed_pairs.append((s, a))

        if done:
            target_r = r
        else:
            target_r = r + self.gamma * np.max(self.Q_sa[s_next])

        self.Q_sa[s, a] += self.learning_rate * (target_r - self.Q_sa[s, a])

        # removing this to prevent full table scanning every time we update
        # plus making it more clear with observed_sa / pairs vars
        # visited = np.argwhere(self.transition_counts.sum(axis=2) > 0)

        # this is like the iner loop with the simulations
        for i in range(n_planning_updates):
            # if we don't have anything to base our simulations on we break
            if len(self.observed_pairs) == 0:
                break

            # Sample a random previously observed (s, a)
            idx = np.random.randint(len(self.observed_pairs))
            s_simulated, a_simulated = self.observed_pairs[idx]

            counts = self.transition_counts[s_simulated, a_simulated]
            # we need it to be [0:1]
            probs = counts / np.sum(counts)

            s_simulated_next = np.random.choice(self.n_states, p=probs)
            r_simulated = (
                self.rewards[s_simulated, a_simulated, s_simulated_next]
                / self.transition_counts[s_simulated, a_simulated, s_simulated_next]
            )

            # to make the final expression a bit clearer and avoid recomputing on terminal state
            if r_simulated == 100:
                target = r_simulated
            else:
                target = r_simulated + self.gamma * np.max(self.Q_sa[s_simulated_next])
            # the update based on the simulation is kinda the same but with the simulated values
            self.Q_sa[s_simulated, a_simulated] += self.learning_rate * (
                target - self.Q_sa[s_simulated, a_simulated]
            )

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class PrioritizedSweepingAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        # Initialize relevant elements
        self.Q_sa = np.zeros((n_states, n_actions))
        self.transition_counts = np.zeros((n_states, n_actions, n_states))
        self.rewards = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        # e-greedy action selection
        if np.random.random() < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        # self.queue.put((-p,(s,a)))
        # Retrieve the top (s,a) from the queue
        # _,(s,a) = self.queue.get() # get the top (s,a) for the queue

        self.transition_counts[s, a, s_next] += 1
        self.rewards[s, a, s_next] += r

        # add to queue
        if done:
            td_error = abs(r - self.Q_sa[s, a])
        else:
            td_error = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])

        if td_error > self.priority_cutoff:
            self.queue.put((-td_error, (s, a)))

        for i in range(n_planning_updates):
            if self.queue.empty():
                break

            _, (s_simulated, a_simulated) = self.queue.get()

            # same simulation as dyna
            counts = self.transition_counts[s_simulated, a_simulated]
            probs = counts / counts.sum()

            s_simulated_next = np.random.choice(self.n_states, p=probs)
            r_simulated = (
                self.rewards[s_simulated, a_simulated, s_simulated_next]
                / self.transition_counts[s_simulated, a_simulated, s_simulated_next]
            )

            # adding both conditional checks on whether we reached target
            # to avoid recomputing both target and td_error on terminal states
            if r_simulated == 100:
                target = r_simulated
            else:
                target = r_simulated + self.gamma * np.max(self.Q_sa[s_simulated_next])
            # update
            self.Q_sa[s_simulated, a_simulated] += self.learning_rate * (
                target - self.Q_sa[s_simulated, a_simulated]
            )

            # This looks a bit cluttered but it just looks at all the pairs (state, action) that can lead to the just
            # simulated state and checks whether their estimate for it needs to change.

            # everything that can lead to s_simulated
            predecessors = np.argwhere(self.transition_counts[:, :, s_simulated] > 0)

            for s_pred, a_pred in predecessors:
                r_new_est = (
                    self.rewards[s_pred, a_pred, s_simulated]
                    / self.transition_counts[s_pred, a_pred, s_simulated]
                )

                if r_new_est == 100:
                    td_error_new_est = abs(r_new_est - self.Q_sa[s_pred, a_pred])
                else:
                    td_error_new_est = abs(
                        r_new_est
                        + self.gamma * np.max(self.Q_sa[s_simulated])
                        - self.Q_sa[s_pred, a_pred]
                    )

                if td_error_new_est > self.priority_cutoff:
                    self.queue.put((-td_error_new_est, (s_pred, a_pred)))

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = "dyna"  # or 'ps'
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    # Initialize environment and policy
    env = WindyGridworld()
    if policy == "dyna":
        pi = DynaAgent(
            env.n_states, env.n_actions, learning_rate, gamma
        )  # Initialize Dyna policy
    elif policy == "ps":
        pi = PrioritizedSweepingAgent(
            env.n_states, env.n_actions, learning_rate, gamma
        )  # Initialize PS policy
    else:
        raise KeyError("Policy {} not implemented".format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = False

    for t in range(n_timesteps):
        # Select action, transition, update policy
        a = pi.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        pi.update(
            s=s,
            a=a,
            r=r,
            done=done,
            s_next=s_next,
            n_planning_updates=n_planning_updates,
        )

        # Render environment
        if plot:
            env.render(
                Q_sa=pi.Q_sa,
                plot_optimal_policy=plot_optimal_policy,
                step_pause=step_pause,
            )

        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input(
                "Press 'Enter' to execute next step, press 'c' to run full algorithm"
            )
            continuous_mode = True if key_input == "c" else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next


if __name__ == "__main__":
    test()
