from stable_baselines3.common.evaluation import evaluate_policy
from gym_wrapper import GymWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from Callback import CustomEvalCallback
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import pickle



class PolicyTrainer:
    def __init__(self, model_name, policy_class=PPO, n_steps=512):
        self.name = model_name
        self.seed = 1
        self.policy_class = policy_class
        self.engel_sim = GymWrapper(self)
        self.engel_sim_test = GymWrapper(self)
        self.env = DummyVecEnv([lambda: self.engel_sim])
        self.eval_env = DummyVecEnv([lambda: Monitor(self.engel_sim_test, f'./current_models/{self.name}/eval_logs')])
        self.best_model_save_path = f'./current_models/best_{self.name}'
        self.model = self.policy_class("MlpPolicy", self.env, verbose=1, tensorboard_log="./tensorboard/",
                                       n_steps=n_steps)

    def train(self, timesteps):
        callback = CustomEvalCallback(self.eval_env, self.best_model_save_path, self.name, safe_freq=999_999_999,
                                      early_stopping_patience=400_000, eval_freq=5_000)
        self.model.learn(total_timesteps=timesteps, callback=callback)


        """self.model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        input("Press Enter to continue...")

        # Continue training for the remaining timesteps
        remaining_timesteps = 4500
        self.model.learn(total_timesteps=remaining_timesteps, callback=callback, reset_num_timesteps=False)

        # Stop tracking and print memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")"""

        valid_job_picked_history = self.engel_sim.valid_job_picked_history
        remaining_prod_time_T4_history = self.engel_sim.remaining_prod_time_T4_history
        finished_jobs_history = self.engel_sim.finished_jobs_history
        reward_history = self.engel_sim.episodes_reward_history

        # Save data
        with open('pickl_dumps/training/valid_job_picked_history.pkl', 'wb') as f:
            pickle.dump(valid_job_picked_history, f)
        with open('pickl_dumps/training/remaining_prod_time_T4_history.pkl', 'wb') as f:
            pickle.dump(remaining_prod_time_T4_history, f)
        with open('pickl_dumps/training/finished_jobs_history.pkl', 'wb') as f:
            pickle.dump(finished_jobs_history, f)
        with open('pickl_dumps/training/reward_history.pkl', 'wb') as f:
            pickle.dump(reward_history, f)
        with open('pickl_dumps/training/truncated_vs_terminated.pkl', 'wb') as f:
            pickle.dump(self.engel_sim.truncated_vs_terminated, f)

        window_size = 100  # episodes for moving average
        moving_avg_rewards = np.convolve(reward_history, np.ones(window_size), 'valid') / window_size
        moving_avg_finished_jobs = np.convolve(finished_jobs_history, np.ones(window_size), 'valid') / window_size

        # Create directories for each plot
        reward_plot_dir = 'reward_plots'
        job_vs_time_plot_dir = 'job_vs_time_plots'
        os.makedirs(reward_plot_dir, exist_ok=True)
        os.makedirs(job_vs_time_plot_dir, exist_ok=True)

        plt.figure(figsize=(30, 11))  # Increase figure size
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Simulation runs')
        ax1.set_ylabel('Reward', color=color)
        ax1.plot(moving_avg_rewards, color=color, linewidth=.5)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Finished Jobs', color=color)
        ax2.plot(moving_avg_finished_jobs, color=color, linewidth=.5)
        ax2.tick_params(axis='y', labelcolor=color)
        # self.visualize_truncated_history(self.engel_sim.truncated_history)

        plt.title(f'Train: Moving avg({window_size}) Rewards, Finished Jobs for total TIMESTEPS:'f'{timesteps}')
        plt.savefig(
            os.path.join(reward_plot_dir, f'{time.strftime("%Y-%m-%d")}Training reward_plot_TIMESTEPS_{timesteps}.png'),
            dpi=600)
        plt.show()

        # Separate job picked history into two lists
        job1_times = [i for i, job in enumerate(valid_job_picked_history) if job == 1]
        print("Valid Job 1 count:", len(job1_times))
        job2_times = [i for i, job in enumerate(valid_job_picked_history) if job == 2]
        print("Valid Job 2 count:", len(job2_times))
        # Create the scatter plot for Job 1,2
        plt.scatter(job1_times, [remaining_prod_time_T4_history[i] for i in job1_times], color='blue', label='Job 1',
                    s=.2)
        plt.scatter(job2_times, [remaining_prod_time_T4_history[i] for i in job2_times], color='red', label='Job 2',
                    s=.3)
        plt.title(f'Training: Job 1, Job 2 vs Remaining Prod Time T4 for {timesteps} TIMESTEPS')
        plt.xlabel('Timesteps')
        plt.ylabel('Remaining Production Time T4')
        plt.legend()
        plt.savefig(os.path.join(job_vs_time_plot_dir,
                                 f'{time.strftime("%Y-%m-%d")}Training Job1_vs_Job2_TIMESTEPS_{timesteps}.png'),
                    dpi=600)
        plt.show()

    def evaluate(self, randomize=False, episoded=None, deterministic=True):
        print(f"-----------------Evaluating---deterministic={deterministic}--------------")
        maximum = 9
        picked_jobs_in_maximum = []
        for i in range(1 if episoded is None else episoded):
            obs, _ = self.engel_sim_test.reset()
            terminated, truncated = False, False
            # print(obs)
            remaining_prod_time_T4_episode_history = []
            while not (terminated or truncated):
                if randomize:
                    action = self.engel_sim_test.action_space.sample()
                else:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    """This parameter corresponds to "Whether to use deterministic or stochastic actions". So the 
                    thing is when you are selecting an action according to given state, the actor_network gives you a 
                    probability distribution. For example for two possible actions a1 and a2: [0.25, 0.75]. If you 
                    use deterministic=True, the result will be action a2 since it has more probability. In the case 
                    of deterministic=False, the result action will be selected with given probabilities [0.25, 0.75]."""
                obs, reward, terminated, truncated, info = self.engel_sim_test.step(action)
                remaining_prod_time_T4_episode_history.append(self.engel_sim_test.simulation.remainingProdTime_T4)
                if terminated or truncated:
                    print(f"truncated: {truncated}, terminate: {terminated}")

            if not truncated:
                if sum(self.engel_sim_test.get_info()['finished_products'].values()) >= maximum:
                    picked_jobs_in_maximum.append(self.engel_sim_test.simulation.decision_list)
                    maximum = sum(self.engel_sim_test.get_info()['finished_products'].values())
                    print("Maximum: ", maximum)
                    print("Average remaining time T4 for max episode: ", np.mean(remaining_prod_time_T4_episode_history))
                    print("Composition: ", self.engel_sim_test.get_info()['finished_products'])
                    time.sleep(5)

        print("Valid Job picked history: ", self.engel_sim_test.valid_job_picked_history)
        finished_jobs_history = self.engel_sim_test.finished_jobs_history
        reward_history = self.engel_sim_test.episodes_reward_history

        # Save data
        with open('pickl_dumps/evaluating/valid_job_picked_history', 'wb') as f:
            pickle.dump(self.engel_sim_test.valid_job_picked_history, f)
        with open('pickl_dumps/evaluating/remaining_prod_time_T4_history.pkl', 'wb') as f:
            pickle.dump(self.engel_sim_test.remaining_prod_time_T4_history, f)
        with open('pickl_dumps/evaluating/truncated_vs_terminated.pkl', 'wb') as f:
            pickle.dump(self.engel_sim_test.truncated_vs_terminated, f)
        with open('pickl_dumps/evaluating/finished_jobs_history.pkl', 'wb') as f:
            pickle.dump(self.engel_sim_test.finished_jobs_history, f)

        # Create directories for each plot
        reward_plot_dir = 'reward_plots'
        job_vs_time_plot_dir = 'job_vs_time_plots'
        os.makedirs(reward_plot_dir, exist_ok=True)
        os.makedirs(job_vs_time_plot_dir, exist_ok=True)

        if episoded is not None:
            window_size = 1  # episodes for moving average
            moving_avg_rewards = np.convolve(reward_history, np.ones(window_size), 'valid') / window_size
            moving_avg_finished_jobs = np.convolve(finished_jobs_history, np.ones(window_size), 'valid') / window_size

            plt.figure(figsize=(30, 11))  # Increase figure size
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('Simulation runs')
            ax1.set_ylabel('Reward', color=color)
            ax1.plot(moving_avg_rewards, color=color, linewidth=.5)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Finished Jobs', color=color)
            ax2.plot(moving_avg_finished_jobs, color=color, linewidth=.5)
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title(f'Evaluation: Moving avg({window_size}) of Rewards and Finished Jobs')
            plt.savefig(os.path.join(reward_plot_dir, f'{time.strftime("%Y-%m-%d")}Evaluation reward_plot.png'),
                        dpi=600)
            plt.show()

        # Separate job picked history into two lists
        job1_times = [i for i, job in enumerate(self.engel_sim_test.valid_job_picked_history) if job == 1]
        print("Valid Job 1 picked count:", len(job1_times))
        job2_times = [i for i, job in enumerate(self.engel_sim_test.valid_job_picked_history) if job == 2]
        print("Valid Job 2 picked count:", len(job2_times))
        # Create the scatter plot for Job 1,2
        plt.scatter(job1_times, [self.engel_sim_test.remaining_prod_time_T4_history[i] for i in job1_times],
                    color='blue', label='Job 1', s=15)
        plt.scatter(job2_times, [self.engel_sim_test.remaining_prod_time_T4_history[i] for i in job2_times],
                    color='red', label='Job 2', s=15)

        plt.title(f'Evaluation: Job 1, Job 2 vs Remaining Production Time T4')
        plt.xlabel('Single Steps')
        plt.ylabel('Remaining Production Time T4')
        plt.legend()
        plt.savefig(os.path.join(job_vs_time_plot_dir, f'{time.strftime("%Y-%m-%d")}Evaluation Job1_vs_Job2_.png'),
                    dpi=600)
        plt.show()

        # self.visualize_truncated_history(self.engel_sim_test.truncated_history)

        return self.engel_sim_test.get_info()

    def evaluate_policy_official(self, model, eval_env):
        print(evaluate_policy(model, eval_env, n_eval_episodes=1, return_episode_rewards=True))
        # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    def evaluate_with_graph(self, randomize=False, episoded=None):
        print("-----------------Evaluating with Graph-----------------")
        job1_means, job2_means = [], []
        for i in range(1, 1 if episoded is None else episoded):
            obs, _ = self.engel_sim_test.reset()
            terminated = False
            truncated = False
            job1_sum, job2_sum = 0, 0
            while not (terminated or truncated):
                if randomize:
                    action = self.engel_sim_test.action_space.sample()
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.engel_sim_test.step(action)
                if not truncated:
                    finished_jobs = self.engel_sim_test.get_info()['finished_products']
                    job1_sum += finished_jobs['Job 1']
                    job2_sum += finished_jobs['Job 2']
                    job1_means.append(job1_sum / i)
                    job2_means.append(job2_sum / i)

        plt.plot(list(range(1, episoded + 1)), (np.array(job1_means) + np.array(job2_means)), label='combined')
        plt.plot(list(range(1, episoded + 1)), job1_means, label='Job 1')
        plt.plot(list(range(1, episoded + 1)), job2_means, label='Job 2')
        plt.xlabel('Evaluation run')
        plt.ylabel('Jobs done')
        plt.legend()
        plt.show()

    def save(self, path: str):
        self.model.save(path)

    def visualize_truncated_history(self, data):
        plt.bar(range(len(data)), data)
        plt.xlabel('Simulation runs')
        plt.ylabel('Value')
        plt.title('Frequency of choosing invalid option')
        plt.show()


# Function for TIMESTEPS
'''
    def evaluate(self, TIMESTEPS, steps_done=0):
        while steps_done < TIMESTEPS:
            obs, _ = self.engel_sim_test.reset()
            done = False
            # print(obs)
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # print(action)
                obs, reward, terminated, truncated, info = self.engel_sim_test.step(action)
                steps_done += 1

                if terminated or truncated:
                    print(f"truncated: {truncated}, terminate: {terminated}")
                    done = True

        return self.engel_sim_test.get_info()
'''
