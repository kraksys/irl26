from pathlib import Path 

# imports necessary to export the greedy terminal policy output as an SVG to present in the report
from contextlib import redirect_stdout 
from rich.console import Console 
from rich.text import Text 

# multiprocessing because we don't like waiting
from concurrent.futures import ProcessPoolExecutor 
from itertools import repeat 


import io 
import os 

import numpy as np 
import matplotlib 

matplotlib.use("Agg") 
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16, 
    "ytick.labelsize": 16, 
    "legend.fontsize": 16,
    "axes.grid": True, 
    "grid.alpha": 0.25, 
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

plot_colors = {
    "qlearning": "#1f77b4",
    "sarsa": "#d55e00",
    "expectedsarsa": "#2ca02c",
    "nstepsarsa": "#cc79a7",
}

from tqdm import tqdm 

from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment  

# smoothing curve function
def smooth_curve(values, window=100):
    values = np.asarray(values, dtype=float)

    if window <= 1: 
        episodes = np.arange(1, len(values) + 1)
        return episodes, values 

    window = min(window, len(values)) 

    smoothed = np.convolve(values, np.ones(window) / window, mode="valid") 

    episodes = np.arange(window, len(values) + 1) 
    return episodes, smoothed

# a helper to make agents with different paramters 
def make_agent(agent_type, env, alpha=0.1, epsilon=0.1, gamma=1.0, n=1):
    if agent_type == "qlearning": 
        return QLearningAgent(env.action_size(), env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma) 
    elif agent_type == "sarsa": 
        return SARSAAgent(env.action_size(), env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma) 
    elif agent_type == "expectedsarsa": 
        return ExpectedSARSAAgent(env.action_size(), env.state_size(), epsilon=epsilon, alpha=alpha, gamma=gamma) 
    elif agent_type == "nstepsarsa": 
        return nStepSARSAAgent(env.action_size(), env.state_size(), n=n, epsilon=epsilon, alpha=alpha, gamma=gamma)

def single_run(agent_type="qlearning", n_episodes=10000, environment=ShortcutEnvironment,
               alpha=0.1, epsilon=0.1, gamma=1.0, n=1):
    env = environment() 
    agent = make_agent(agent_type, env, alpha=alpha, epsilon=epsilon, gamma=gamma, n=n) 
    rewards = agent.train(n_episodes, env) 
    return np.asarray(rewards), agent, env

def one_repetition(agent_type, n_episodes, environment, alpha, epsilon, gamma, n):
    # reset the rng 
    np.random.seed(None)

    rewards = single_run(agent_type, n_episodes, environment, alpha, epsilon, gamma, n)[0]
    return rewards

def run_repetitions(n_rep=100, n_episodes=1000, agent_type="qlearning", environment=ShortcutEnvironment, alpha=0.1, epsilon=0.1, gamma=1.0, n=1, n_jobs=None): 

    # creating a parallel worker pool 
    if n_jobs is None: 
        n_jobs = max(1, (os.cpu_count() or 1) - 1) 

    # 1 job <-> 1 repetition 
    if n_jobs == 1: 
        all_results = [
            one_repetition(agent_type, n_episodes, environment, alpha, epsilon, gamma, n)
            for rep in tqdm(range(n_rep), desc=f"{agent_type} repetitions")
        ]
    else: 
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            all_results = list(tqdm(
                executor.map(one_repetition, 
                             repeat(agent_type, n_rep), 
                             repeat(n_episodes, n_rep), 
                             repeat(environment, n_rep), 
                             repeat(alpha, n_rep), 
                             repeat(epsilon, n_rep),
                             repeat(gamma, n_rep), 
                             repeat(n, n_rep),
                             ),
                             total=n_rep,
                             desc=f"{agent_type} repetitions",
            )
        )
    # ensure averaging of rewards is in place 
    data_grid = np.array(all_results) 
    avg_rewards = np.mean(data_grid, axis=0)
    return avg_rewards, data_grid 


# Helper for single experiment curve plotting 
def plot_curve(rewards, title, filename, window=100, ylabel="Average Cumulative Reward", color=None):
    Path("results").mkdir(exist_ok=True) 
    x,y = smooth_curve(rewards, window)

    plt.figure(figsize=(10, 6))
    plt.plot(x,y , label=f"Moving average, window={window}", linewidth=1.4, linestyle="-", color=color)
    plt.xlabel("Episodes", fontsize=20) 
    plt.ylabel(ylabel, fontsize=20) 
    plt.title(title, fontsize=20) 
    plt.ylim(-300, 10)
    plt.legend()
    plt.grid(True) 
    plt.tight_layout()
    plt.savefig(Path("results") / filename, dpi=300)
    plt.close()


# Helper for multi-curve plotting 
def plot_many_curves(curves, title, filename, window=100, colors=None, ylim=(-800, 10)):
    Path("results").mkdir(exist_ok=True) 

    plt.figure(figsize=(10,6))
    for label, rewards in curves.items():
        x,y = smooth_curve(rewards, window) 
        plt.plot(x,y , label=label, linewidth=1.4, linestyle="-", color= None if colors is None else colors.get(label)) 

    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Average Episode Return", fontsize=20)
    plt.title(title, fontsize=20)
    plt.ylim(*ylim) 
    plt.legend() 
    plt.grid(True) 
    plt.tight_layout() 
    plt.savefig(Path("results") / filename, dpi=300) 
    plt.close() 

# Helper to save greedy policy .svg's for nice presentation in the report 
def save_greedy_policy(agent, env, filename, title): 
    Path("results").mkdir(exist_ok=True)
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        env.render_greedy(agent.Q)

    text = buffer.getvalue()

    console = Console(record=True, width=25) 
    console.print(Text.from_ansi(text), soft_wrap=False) 
    console.save_svg(str(Path("results") / filename), title=title)

# iterate through all alpha values using repetitions parallel worker 

def run_alpha_experiment(alphas, n_rep=100, n_episodes=1000, agent_type="qlearning"): 
    results = {}

    for a in alphas: 
        avg_rewards = run_repetitions(
            n_rep=n_rep, 
            n_episodes=n_episodes, 
            agent_type=agent_type, 
            alpha=a,
        )[0]
        results[f"alpha={a}"] = avg_rewards 

    return results 

# iterate through all n-step values using repetitions parallel worker

def run_n_experiment(n_values, n_rep=100, n_episodes=1000, alpha=0.1):
    results = {}

    for n in n_values: 
        avg_rewards = run_repetitions(
            n_rep=n_rep,
            n_episodes=n_episodes,
            agent_type="nstepsarsa",
            alpha=alpha,
            n=n,
        )[0]
        results[f"n={n}"] = avg_rewards 
    
    return results 


# helper to find the best curve based on cumulative average reward across 1000 episodes
def best_curve(curves):
    best_label = None 
    best_score = -np.inf 

    for label, rewards in curves.items():
        score = np.mean(rewards[-1000:])
        if score > best_score: 
            best_score = score 
            best_label = label 

    return best_label, curves[best_label] 


# function to wrap all experiments into one 

def run_all_experiments():
    alphas = [0.01, 0.1, 0.5, 0.9]
    n_values = [1,2,5,10,25]

    q_rewards, q_agent, q_env = single_run("qlearning")
    plot_curve(q_rewards, "Q-Learning Performance", "qlearning_single.png", window=100, ylabel="Episode Return", color=plot_colors["qlearning"])
    save_greedy_policy(q_agent, q_env, "qlearning_greedy.svg", "Q-Learning Greedy Policy")
    
    q_avg = run_repetitions(agent_type="qlearning")[0]
    plot_curve(q_avg, "Q-Learning Average Learning Curve", "qlearning_mean.png", window=100, ylabel="Average Episode Return", color=plot_colors["qlearning"])

    q_alphas = run_alpha_experiment(alphas, agent_type="qlearning")
    plot_many_curves(q_alphas, "Q-Learning: Impact of Learning Rate", "qlearning_alpha_sweep.png", window=100) 

    s_rewards, s_agent, s_env = single_run("sarsa")
    plot_curve(s_rewards, "SARSA Performance", "sarsa_single.png", window=100, ylabel="Episode Return", color=plot_colors["sarsa"]) 
    save_greedy_policy(s_agent, s_env, "sarsa_greedy.svg", "SARSA Greedy Policy") 

    s_avg = run_repetitions(agent_type="sarsa")[0]
    plot_curve(s_avg, "SARSA Average Learning Curve", "sarsa_mean.png", ylabel="Average Episode Returns", color=plot_colors["sarsa"])

    s_alphas = run_alpha_experiment(alphas, agent_type="sarsa")
    plot_many_curves(s_alphas, "SARSA: Impact of Learning Rate", "sarsa_alpha_sweep.png")

    windy_q, windy_q_env = single_run("qlearning", environment=WindyShortcutEnvironment)[1:]
    save_greedy_policy(windy_q, windy_q_env, "windy_qlearning_greedy.svg", "Windy Q-Learning Greedy Policy")

    windy_s, windy_s_env = single_run("sarsa", environment=WindyShortcutEnvironment)[1:]
    save_greedy_policy(windy_s, windy_s_env, "windy_sarsa_greedy.svg", "Windy SARSA Greedy Policy")

    e_rewards, e_agent, e_env = single_run("expectedsarsa")[0:3]
    plot_curve(e_rewards, "Expected SARSA Performance", "expected_sarsa_single.png", window=100, ylabel="Episode Return", color=plot_colors["expectedsarsa"])
    save_greedy_policy(e_agent, e_env, "expected_sarsa_greedy.svg", "Expected SARSA Greedy Policy") 

    e_avg = run_repetitions(agent_type="expectedsarsa")[0]
    plot_curve(e_avg, "Expected SARSA Average Learning Curve", "expected_sarsa_mean.png", window=100, color=plot_colors["expectedsarsa"]) 

    e_alphas = run_alpha_experiment(alphas, agent_type="expectedsarsa") 
    plot_many_curves(e_alphas, "Expected SARSA: Impact of Learning Rate", "expected_sarsa_alpha_sweep.png", window=100)

    n_rewards = single_run("nstepsarsa", n=1)[0]
    plot_curve(n_rewards, "n-step SARSA Performance", "nstepsarsa_single.png", window=100, ylabel="Episode Return", color=plot_colors["nstepsarsa"])

    n_results = run_n_experiment(n_values) 
    plot_many_curves(n_results, "n-step SARSA: Impact of n", "nstepsarsa_n_sweep.png", window=100)

    q_label, q_best = best_curve(q_alphas) 
    s_label, s_best = best_curve(s_alphas) 
    e_label, e_best = best_curve(e_alphas) 
    n_label, n_best = best_curve(n_results) 

    best_curves = {
        f"Q-Learning ({q_label})": q_best, 
        f"SARSA ({s_label})": s_best, 
        f"Expected SARSA ({e_label})": e_best, 
        f"n-step SARSA ({n_label})": n_best, 
    }

    best_colors = {
        f"Q-Learning ({q_label})": plot_colors["qlearning"],
        f"SARSA ({s_label})": plot_colors["sarsa"],
        f"Expected SARSA ({e_label})": plot_colors["expectedsarsa"],
        f"n-step SARSA ({n_label})": plot_colors["nstepsarsa"],
    }

    plot_many_curves(best_curves, "Best Parameters Across All Experiments", "best_param_comparison.png", window=100, colors=best_colors, ylim=(-200, 10))

if __name__ == "__main__":
    run_all_experiments() 
