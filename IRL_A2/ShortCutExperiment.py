# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib
# easier to load a window than opening an image for every plot
matplotlib.use('TkAgg') # Or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt
from ShortCutAgents import QLearningAgent
from ShortCutEnvironment import ShortcutEnvironment
# I think I'll move the progress tracking into these functions

def single_run(n_episodes=10000):
    env = ShortcutEnvironment()
    agent = QLearningAgent(env.action_size(), env.state_size())
    reward_vector = agent.train(n_episodes, env)
    print("Final Greedy Policy Map:")
    env.render_greedy(agent.Q)

    plt.figure(figsize=(10, 6))
    plt.plot(reward_vector)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Q-Learning Performance')
    plt.grid(True)
    plt.show()

# single run
single_run(10000)



def run_repetitions(n_rep=100, n_episodes=1000):
    all_results = []

    for rep in range(n_rep):
        env = ShortcutEnvironment()
        agent = QLearningAgent(env.action_size(), env.state_size())

        rewards = agent.train(n_episodes, env)
        all_results.append(rewards)

    data_grid = np.array(all_results)
    avg_rewards = np.mean(data_grid, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    plt.title("Q-Learning: Average Learning Curve (100 Repetitions)")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.grid(True)
    plt.show()


# multiple runs
run_repetitions(100, 1000)


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

test_alphas = [0.01, 0.1, 0.5, 0.9]
run_alpha_experiment(test_alphas)

