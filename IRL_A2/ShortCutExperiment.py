# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib
# easier to load a window than opening an image for every plot
matplotlib.use('TkAgg') # Or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt
from ShortCutAgents import QLearningAgent
from ShortCutEnvironment import ShortcutEnvironment


#NOT FINISHED

def run_repetitions(n_episodes=10000):
    env = ShortcutEnvironment()
    agent = QLearningAgent(env.action_size(), env.state_size())
    reward_vector = agent.train(n_episodes, env)
    print("Final Greedy Policy Map:")
    env.render_greedy(agent.Q)
    return reward_vector

data = run_repetitions()
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Q-Learning Performance')
plt.grid(True)
plt.show()


def run_repetitions(n_rep=100, n_episodes=1000):
    all_results = []

    for rep in range(n_rep):

        env = ShortcutEnvironment()
        agent = QLearningAgent(env.action_size(), env.state_size())


        rewards = agent.train(n_episodes, env)
        all_results.append(rewards)

    # Turn the list into a NumPy "grid" so we can calculate the average
    # This turns 100 lists of 1000 numbers into a 100x1000 table
    data_grid = np.array(all_results)
    avg_rewards = np.mean(data_grid, axis=0)

    return avg_rewards



average_learning_curve = run_repetitions(n_rep=100, n_episodes=1000)

plt.figure(figsize=(10, 5))
plt.plot(average_learning_curve)
plt.title("Q-Learning: Average Learning Curve (100 Repetitions)")
plt.xlabel("Episode")
plt.ylabel("Average Cumulative Reward")
plt.grid(True)
plt.show()


