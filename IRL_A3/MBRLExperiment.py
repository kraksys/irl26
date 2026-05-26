#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""

import csv
import time

import numpy as np
from Helper import LearningCurvePlot, smooth
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from MBRLEnvironment import WindyGridworld
from tqdm import tqdm


def run_repetitions(
    agent_class,
    n_timesteps,
    n_repetitions,
    eval_interval,
    gamma,
    learning_rate,
    epsilon,
    n_planning_updates,
    wind_proportion,
    default_reward_per_timestep=-1.0 # to test our reflection hypothesis
):
    all_curves = []
    runtimes = []
    eval_timesteps = np.arange(0, n_timesteps, eval_interval)

    for rep in tqdm(
        range(n_repetitions),
        desc=f"{agent_class.__name__} k={n_planning_updates}, wind={wind_proportion}",
    ):
        # initialize a new environment and agent from scratch each repetition
        env = WindyGridworld(wind_proportion=wind_proportion, default_reward_per_timestep=default_reward_per_timestep)
        eval_env = WindyGridworld(wind_proportion=wind_proportion, default_reward_per_timestep=default_reward_per_timestep)
        agent = agent_class(env.n_states, env.n_actions, learning_rate, gamma)

        s = env.reset()
        curve = []

        start = time.perf_counter()
        for t in range(n_timesteps):
            # evaluate every eval_interval steps
            if t % eval_interval == 0:
                mean_return = agent.evaluate(
                    eval_env, n_eval_episodes=30, max_episode_length=100
                )
                curve.append(mean_return)

            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)

            agent.update(
                s=s,
                a=a,
                r=r,
                done=done,
                s_next=s_next,
                n_planning_updates=n_planning_updates,
            )

            if done:
                s = env.reset()
            else:
                s = s_next

        runtimes.append(time.perf_counter() - start)

        all_curves.append(curve)

    # average the curves over all repetitions
    mean_curve = np.mean(all_curves, axis=0)
    return eval_timesteps, mean_curve, np.array(runtimes)


# select the best of the curves to enable comparison
def select_best(curves):
    best_plan = None
    best_curve = None
    best_return = -np.inf

    for k, curve in curves.items():
        score = np.mean(curve[int(len(curve) * 0.8):])
        if score > best_return:
            best_return = score
            best_plan = k 
            best_curve = curve

    return best_plan, best_curve


# plot the comparison of the different curves
def plot_comparison(
    title,
    filename,
    eval_timesteps,
    q_curve,
    best_dyna,
    best_dyna_curve,
    best_ps,
    best_ps_curve,
    smoothing_window,
):
    plot = LearningCurvePlot(title=title)

    plot.add_curve(
        eval_timesteps, smooth(q_curve, smoothing_window), label="Q-Learning"
    )

    plot.add_curve(
        eval_timesteps,
        smooth(best_dyna_curve, smoothing_window),
        label=f"Best Dyna K={best_dyna}",
    )

    plot.add_curve(
        eval_timesteps,
        smooth(best_ps_curve, smoothing_window),
        label=f"Best PS K={best_ps}",
    )

    plot.save(filename)
    print(f"Saved plot {filename}")


# save a csv file of the runtimes, to enable their usage in the paper
def save_runtime_table(runtime_rows):
    with open("runtime_table.csv", "w", newline="") as f:
        fields = [
            "algorithm",
            "wind_proportion",
            "n_planning_updates",
            "mean_runtime",
            "std_runtime",
            "final_return",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in runtime_rows:
            writer.writerow(row)

    print("Saved runtime table...")

# to test hypothesis of varying default rewards
def experiment_default_reward_effect():
    np.random.seed(70)
    n_timesteps = 10001 
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportion = 0.9
    n_planning_updates = 1
    reward_values = [-1.0, -0.1, 0.0]
    smoothing_window = 11

    colors = {
        -1.0: "tab:blue",
        -0.1: "tab:orange",
        0.0: "tab:green"
    }

    linestyles = {
        -1.0: "-",
        -0.1: "--",
        0.0: ":"
    }

    dyna_plot = LearningCurvePlot(title="Default Reward Comparison Dyna (K=1), Stochastic WGW")

    for reward in reward_values: 
      
        ts, curve, _ = run_repetitions(
            DynaAgent,
            n_timesteps,
            n_repetitions,
            eval_interval,
            gamma, 
            learning_rate,
            epsilon,
            n_planning_updates=n_planning_updates,
            wind_proportion=wind_proportion,
            default_reward_per_timestep=reward
        )

        label = f"Reward = {reward}, K=1"
        dyna_plot.ax.plot(
            ts,
            smooth(curve, smoothing_window),
            label=f"Dyna, Reward = {reward}",
            color=colors[reward],
            linestyle=linestyles[reward],
            linewidth=2.0
        )
    
    dyna_plot.save("default_reward_effect_dyna.png")

    # Prioritized Sweeping comparison 

    ps_plot = LearningCurvePlot(
        title="Default Reward Comparison PS (K=1), Stochastic WGW"
    )

    for reward in reward_values:
        ts, curve, _ = run_repetitions(
            PrioritizedSweepingAgent,
            n_timesteps,
            n_repetitions,
            eval_interval,
            gamma,
            learning_rate,
            epsilon,
            n_planning_updates=n_planning_updates,
            wind_proportion=wind_proportion, 
            default_reward_per_timestep=reward
        )
    
        ps_plot.ax.plot(
            ts, 
            smooth(curve, smoothing_window),
            label=f"PS, Reward = {reward}",
            color=colors[reward],
            linestyle=linestyles[reward],
            linewidth=2.0
        )

    ps_plot.save("default_reward_effect_ps.png")

def experiment():
    np.random.seed(70)
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportions = [0.9, 1.0]
    n_planning_updates = [1, 3, 5]
    smoothing_window = 11

    q_results = {}
    dyna_results = {}
    ps_results = {}

    q_runtimes = {}
    dyna_runtimes = {}
    ps_runtimes = {}

    runtime_rows = []
    # Dyna experiments
    for wind_proportion in wind_proportions:
        if wind_proportion == 0.9:
            title = "Dyna - Stochastic WGW (wind=0.9)"
            fname = "dyna_stochastic.png"
        else:
            title = "Dyna - Deterministic WGW (wind=1.0)"
            fname = "dyna_deterministic.png"

        plot = LearningCurvePlot(title=title)

        # Q-learning baseline: n_planning_updates=0 means no planning, just Q-learning
        ts, curve, runtimes = run_repetitions(
            DynaAgent,
            n_timesteps,
            n_repetitions,
            eval_interval,
            gamma,
            learning_rate,
            epsilon,
            n_planning_updates=0,
            wind_proportion=wind_proportion,
        )

        q_results[wind_proportion] = curve
        q_runtimes[wind_proportion] = runtimes

        plot.add_curve(ts, smooth(curve, smoothing_window), label="Q-Learning (K=0)")

        dyna_results[wind_proportion] = {}
        dyna_runtimes[wind_proportion] = {}
        # Dyna with different planning budgets
        for k in n_planning_updates:
            ts, curve, runtimes = run_repetitions(
                DynaAgent,
                n_timesteps,
                n_repetitions,
                eval_interval,
                gamma,
                learning_rate,
                epsilon,
                n_planning_updates=k,
                wind_proportion=wind_proportion,
            )
            plot.add_curve(
                ts, smooth(curve, smoothing_window), label=f"Dyna K={k}"
            )

            dyna_results[wind_proportion][k] = curve
            dyna_runtimes[wind_proportion][k] = runtimes

        plot.save(fname)
        print(f"Saved {fname}")

    # Prioritized Sweeping experiments
    for wind_proportion in wind_proportions:
        if wind_proportion == 0.9:
            title = "Prioritized Sweeping - Stochastic WGW (wind=0.9)"
            fname = "ps_stochastic.png"
        else:
            title = "Prioritized Sweeping - Deterministic WGW (wind=1.0)"
            fname = "ps_deterministic.png"

        plot = LearningCurvePlot(title=title)

        # Reusing Q-Learning baseline from DynaAgent with n_planning_updates = 0
        plot.add_curve(
            ts,
            smooth(q_results[wind_proportion], smoothing_window),
            label="Q-Learning (K=0)",
        )

        ps_results[wind_proportion] = {}
        ps_runtimes[wind_proportion] = {}
        # Prioritized Sweeping with different planning budgets
        for k in n_planning_updates:
            ts, curve, runtimes = run_repetitions(
                PrioritizedSweepingAgent,
                n_timesteps,
                n_repetitions,
                eval_interval,
                gamma,
                learning_rate,
                epsilon,
                n_planning_updates=k,
                wind_proportion=wind_proportion,
            )
            plot.add_curve(ts, smooth(curve, smoothing_window), label=f"PS K={k}")

            ps_results[wind_proportion][k] = curve
            ps_runtimes[wind_proportion][k] = runtimes

        plot.save(fname)
        print(f"Saved {fname}")

        best_dyna, best_dyna_curve = select_best(dyna_results[wind_proportion])

        best_ps, best_ps_curve = select_best(ps_results[wind_proportion])

        if wind_proportion == 0.9:
            comparison_title = "Comparison - Stochastic WGW (wind=0.9)"
            comparison_fname = "comparison_stochastic.png"
        else:
            comparison_title = "Comparison - Deterministic WGW (wind=1.0)"
            comparison_fname = "comparison_deterministic.png"

        plot_comparison(
            title=comparison_title,
            filename=comparison_fname,
            eval_timesteps=ts,
            q_curve=q_results[wind_proportion],
            best_dyna=best_dyna,
            best_dyna_curve=best_dyna_curve,
            best_ps=best_ps,
            best_ps_curve=best_ps_curve,
            smoothing_window=smoothing_window,
        )

        
        runtime_rows.append(
            {
                "algorithm": "Q-Learning",
                "wind_proportion": wind_proportion,
                "n_planning_updates": 0,
                "mean_runtime": np.mean(q_runtimes[wind_proportion]),
                "std_runtime": np.std(q_runtimes[wind_proportion]),
                "final_return": np.mean(q_results[wind_proportion][int(len(q_results[wind_proportion]) * 0.8):]),
            }
        )

        runtime_rows.append(
            {
                "algorithm": "Dyna",
                "wind_proportion": wind_proportion,
                "n_planning_updates": best_dyna,
                "mean_runtime": np.mean(dyna_runtimes[wind_proportion][best_dyna]),
                "std_runtime": np.std(dyna_runtimes[wind_proportion][best_dyna]),
                "final_return": np.mean(best_dyna_curve[int(len(best_dyna_curve) * 0.8):]),
            }
        )

        runtime_rows.append(
            {
                "algorithm": "Prioritized Sweeping",
                "wind_proportion": wind_proportion,
                "n_planning_updates": best_ps,
                "mean_runtime": np.mean(ps_runtimes[wind_proportion][best_ps]),
                "std_runtime": np.std(ps_runtimes[wind_proportion][best_ps]),
                "final_return": np.mean(best_ps_curve[int(len(best_ps_curve) * 0.8):]),
            }
        )

    save_runtime_table(runtime_rows)


def experiment_long():
    np.random.seed(70)
    n_timesteps = 50001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    smoothing_window = 11

    best_dyna_n = {0.9: 5, 1.0: 3}
    best_ps_n = {0.9: 3, 1.0: 1}

    wind_proportions = [0.9, 1.0]

    for wind_proportion in wind_proportions:
        label = "Stochastic WGW" if wind_proportion == 0.9 else "Deterministic WGW"

        ts_q, curve_q, _ = run_repetitions(
            DynaAgent, n_timesteps, n_repetitions, eval_interval,
            gamma, learning_rate, epsilon,
            n_planning_updates=0, wind_proportion=wind_proportion,
        )

        ts_dyna, curve_dyna, _ = run_repetitions(
            DynaAgent, n_timesteps, n_repetitions, eval_interval,
            gamma, learning_rate, epsilon,
            n_planning_updates=best_dyna_n[wind_proportion],
            wind_proportion=wind_proportion,
        )

        ts_ps, curve_ps, _ = run_repetitions(
            PrioritizedSweepingAgent, n_timesteps, n_repetitions, eval_interval,
            gamma, learning_rate, epsilon,
            n_planning_updates=best_ps_n[wind_proportion],
            wind_proportion=wind_proportion,
        )

        plot = LearningCurvePlot(title=f"Long-run Comparison - {label} (wind={wind_proportion})")
        plot.add_curve(ts_q, smooth(curve_q, smoothing_window), label="Q-Learning")
        plot.add_curve(ts_dyna, smooth(curve_dyna, smoothing_window), label=f"Best Dyna K={best_dyna_n[wind_proportion]}")
        plot.add_curve(ts_ps, smooth(curve_ps, smoothing_window), label=f"Best PS K={best_ps_n[wind_proportion]}")
        plot.save(f"long_{label}.png")



if __name__ == "__main__":
    experiment()
    experiment_long()
    experiment_default_reward_effect()
