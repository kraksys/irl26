# THIS IS JUST FOR TESTING RANDOM PIECES OF CODE


import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.psOperators import ps_literal
from tqdm import tqdm
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment

def run_alpha_experiment(alphas, n_rep=100, n_episodes=1000):
    plt.figure(figsize=(10, 6))

    for a in alphas:
        all_runs = []

        for n in range(n_rep):
            env = ShortcutEnvironment()
            agent = QLearningAgent(env.action_size(), env.state_size(), alpha=a)

            rewards = agent.train(n_episodes, env)
            all_runs.append(rewards)

        # Average the 100 runs
        avg_rewards = np.mean(all_runs, axis=0)

        # We use a window of 20
        smoothed_rewards = np.convolve(avg_rewards, np.ones(20) / 20, mode='valid')

        plt.plot(smoothed_rewards, label=f'alpha = {a}')

    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward (Smoothed)')
    plt.title('Q-Learning: Impact of Learning Rate (Alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()

env = WindyShortcutEnvironment()
# Run the experiment
agent = ExpectedSARSAAgent(env.action_size(), env.state_size())
# agaent1 = QLearningAgent(env.action_size(), env.state_size())

data = agent.train(10000, env)
# data1 = agaent1.train(10, env)

env.render_greedy(agent.Q)
# plt.figure(figsize=(10, 6))
# plt.plot(data, label=f'SARSA')
# plt.plot(data1, label=f'Q-Learning')
# plt.xlabel('Episodes')
# plt.legend()
# plt.grid(True)
# plt.show()