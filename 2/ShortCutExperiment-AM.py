# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib
# easier to load a window than opening an image for every plot
matplotlib.use('TkAgg') # Or 'Qt5Agg' if you have PyQt installed
import matplotlib.pyplot as plt
from tqdm import tqdm
from ShortCutAgents import QLearningAgent, SARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
# I think I'll move the progress tracking into these functions


#This function basically agent.train() with map render
def single_run(n_episodes=10000, environment=ShortcutEnvironment):
    # agent, env = single_run(10000, WindyShortcutEnvironment)
    env = environment()
    agent = QLearningAgent(env.action_size(), env.state_size())
    reward_vector = agent.train(n_episodes, env)


    window_size = 100
    smoothed_rewards = np.convolve(reward_vector, np.ones(window_size) / window_size, mode='valid')
    print("Final Greedy Policy Map:")
    env.render_greedy(agent.Q)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label='Smoothed Reward', linewidth=2)
    plt.xlabel('Episodes', fontsize=20)
    plt.ylabel('Cumulative Reward', fontsize=20)
    plt.title(f'Q-Learning Performance (Smoothed window={window_size})', fontsize=20)
    plt.ylim(-300, 10)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return agent, env


# single run
a, e = single_run(10000, ShortcutEnvironment)



def run_repetitions(n_rep=100, n_episodes=1000, agent_type="qlearning"):
    # run_repetitions(100, 1000, "sarsa")
    all_results = []

    for rep in tqdm(range(n_rep), desc="Training Agent"):
        env = ShortcutEnvironment()
        if agent_type == "qlearning":
            agent = QLearningAgent(env.action_size(), env.state_size())
        elif agent_type == "sarsa":
            agent = SARSAAgent(env.action_size(), env.state_size())

        rewards = agent.train(n_episodes, env)
        all_results.append(rewards)

    data_grid = np.array(all_results)
    avg_rewards = np.mean(data_grid, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    if agent_type == "qlearning":
        plot_text = "Q-Learning"
    elif agent_type == "sarsa":
        plot_text = "SARSA"
    else:
        plot_text = "AGENT"
    plt.title(f"{plot_text}: Average Learning Curve (100 Repetitions)")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.grid(True)
    plt.show()


# multiple runs
# run_repetitions(100, 1000, "sarsa")


def run_alpha_experiment(alphas, n_rep=100, n_episodes=1000, agent_type="qlearning"):
    # test_alphas = [0.01, 0.1, 0.5, 0.9]
    # run_alpha_experiment(test_alphas, agent_type="sarsa")
    plt.figure(figsize=(10, 6))

    for a in alphas:
        all_runs = []

        for n in tqdm(range(n_rep), desc=f"Training Agent (alpha = {a})"):
            env = ShortcutEnvironment()
            if agent_type == "qlearning":
                agent = QLearningAgent(env.action_size(), env.state_size(), alpha=a)
            elif agent_type == "sarsa":
                agent = SARSAAgent(env.action_size(), env.state_size(), alpha=a)

            rewards = agent.train(n_episodes, env)
            all_runs.append(rewards)


        avg_rewards = np.mean(all_runs, axis=0)

        # We use a window of 20
        smoothed_rewards = np.convolve(avg_rewards, np.ones(20) / 20, mode='valid')

        plt.plot(smoothed_rewards, label=f'alpha = {a}')

    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward (Smoothed)')
    if agent_type == "qlearning":
        plot_text = "Q-Learning"
    elif agent_type == "sarsa":
        plot_text = "SARSA"
    else:
        plot_text = "AGENT"
    plt.title(f'{plot_text}: Impact of Learning Rate (Alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()

# test_alphas = [0.01, 0.1, 0.5, 0.9]
# run_alpha_experiment(test_alphas, agent_type="sarsa")


def get_path(env, agent, start_row="top"):
    # path = get_path(env, agent, "bottom")
    env.reset()
    if start_row == "top":
        start_row = 2
    elif start_row == "bottom":
        start_row = 9

    env.y = start_row
    env.x = 2
    env.starty = start_row  # Required so cliff respawn works correctly

    path = []
    while not env.done():
        state = env.state()
        path.append((env.x, env.y))

        action = np.argmax(agent.Q[state])
        env.step(action)

    return path


def run_windy_comparison():
    # run_windy_comparison()
    # Env
    env_q = WindyShortcutEnvironment()
    env_sarsa = WindyShortcutEnvironment()

    # Agents
    q = QLearningAgent(env_q.action_size(), env_q.state_size())
    sarsa = SARSAAgent(env_sarsa.action_size(), env_sarsa.state_size())

    q.train(10000, env_q)
    sarsa.train(10000, env_sarsa)

    print("Windy Q-Learning Policy")
    env_q.render_greedy(q.Q)

    print("Windy SARSA Policy")
    env_sarsa.render_greedy(sarsa.Q)
    print("We can see that SARSA is a pussy.")


#run_windy_comparison()

