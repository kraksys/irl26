#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from tqdm import tqdm
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth


def run_repetitions(agent_class, n_timesteps, n_repetitions, eval_interval, gamma,
                    learning_rate, epsilon, n_planning_updates, wind_proportion):
    all_curves = []
    eval_timesteps = np.arange(0, n_timesteps, eval_interval)

    for rep in tqdm(range(n_repetitions),
                    desc=f'{agent_class.__name__} n_plan={n_planning_updates}, wind={wind_proportion}'):
        # initialize a new environment and agent from scratch each repetition
        env = WindyGridworld(wind_proportion=wind_proportion)
        eval_env = WindyGridworld(wind_proportion=wind_proportion)
        agent = agent_class(env.n_states, env.n_actions, learning_rate, gamma)

        s = env.reset()
        curve = []

        for t in range(n_timesteps):
            # evaluate every eval_interval steps
            if t % eval_interval == 0:
                mean_return = agent.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100)
                curve.append(mean_return)

            a = agent.select_action(s, epsilon)
            s_next, r, done = env.step(a)
            agent.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

            if done:
                s = env.reset()
            else:
                s = s_next

        all_curves.append(curve)

    # average the curves over all repetitions
    mean_curve = np.mean(all_curves, axis=0)
    return eval_timesteps, mean_curve


def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 20
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportions = [0.9, 1.0]
    n_planning_updatess = [1, 3, 5]
    smoothing_window = 11

    # Dyna experiments
    for wind_proportion in wind_proportions:

        if wind_proportion == 0.9:
            title = 'Dyna - Stochastic environment (wind=0.9)'
            fname = 'dyna_stochastic.png'
        else:
            title = 'Dyna - Deterministic environment (wind=1.0)'
            fname = 'dyna_deterministic.png'

        plot = LearningCurvePlot(title=title)

        # Q-learning baseline: n_planning_updates=0 means no planning, just Q-learning
        ts, curve = run_repetitions(DynaAgent, n_timesteps, n_repetitions, eval_interval,
                                    gamma, learning_rate, epsilon, n_planning_updates=0,
                                    wind_proportion=wind_proportion)
        plot.add_curve(ts, smooth(curve, smoothing_window), label='Q-learning (n=0)')

        # Dyna with different planning budgets
        for n_plan in n_planning_updatess:
            ts, curve = run_repetitions(DynaAgent, n_timesteps, n_repetitions, eval_interval,
                                        gamma, learning_rate, epsilon, n_planning_updates=n_plan,
                                        wind_proportion=wind_proportion)
            plot.add_curve(ts, smooth(curve, smoothing_window), label=f'Dyna n={n_plan}')

        plot.save(fname)
        print(f'Saved {fname}')

    # Prioritized Sweeping experiments
    for wind_proportion in wind_proportions:

        if wind_proportion == 0.9:
            title = 'Prioritized Sweeping - Stochastic environment (wind=0.9)'
            fname = 'ps_stochastic.png'
        else:
            title = 'Prioritized Sweeping - Deterministic environment (wind=1.0)'
            fname = 'ps_deterministic.png'

        plot = LearningCurvePlot(title=title)

        # Q-learning baseline: reuse DynaAgent with n_planning_updates=0
        ts, curve = run_repetitions(DynaAgent, n_timesteps, n_repetitions, eval_interval,
                                    gamma, learning_rate, epsilon, n_planning_updates=0,
                                    wind_proportion=wind_proportion)
        plot.add_curve(ts, smooth(curve, smoothing_window), label='Q-learning (n=0)')

        # Prioritized Sweeping with different planning budgets
        for n_plan in n_planning_updatess:
            ts, curve = run_repetitions(PrioritizedSweepingAgent, n_timesteps, n_repetitions,
                                        eval_interval, gamma, learning_rate, epsilon,
                                        n_planning_updates=n_plan, wind_proportion=wind_proportion)
            plot.add_curve(ts, smooth(curve, smoothing_window), label=f'PS n={n_plan}')

        plot.save(fname)
        print(f'Saved {fname}')


if __name__ == '__main__':
    experiment()