import os
import random
import gymnasium as gym
from gymnasium import spaces
import logging
from sim.decision_point import DecisionPointExit
from sim.simulation import Simulation, run_until_decision_point
import numpy as np
import simulation_data as sim_data
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(),
                                                     logging.FileHandler('sim_run.log', encoding='utf-8')])
logger = logging.getLogger(__name__)


def run_manual(average_time_t4):
    terminated, truncated = False, False
    wrapper.reset()
    while not (terminated or truncated):
        options = wrapper.get_valid_actions()
        remaining_time_T4 = wrapper.get_remaining_production_time()
        action = optimize_manually(options, average_time_t4, remaining_time_T4)
        action_index = wrapper.actions.index(action)
        print("Action index: ", action_index)
        observation, reward, terminated, truncated, info = wrapper.step(action_index)
    finished_jobs = wrapper.last_valid_info['finished_products']
    # finished_jobs = wrapper.get_info()['finished_products']
    return finished_jobs['Job 1'], finished_jobs['Job 2']


def optimize_manually(options, t4_time_benchmark, actual_remainingt4_time):
    # Select job
    job_options = list(set([o[0] for o in options]))
    if len(job_options) == 1:
        picked_job = job_options[0]
    if len(job_options) == 2:
        # BETTER HEURISTIC!
        # picked_job = 2 if dp_count == 0 or t4_time_benchmark < actual_remainingt4_time else 1
        picked_job = 2 if t4_time_benchmark < actual_remainingt4_time else 1
    # Select path
    if picked_job == 1:
        path_options = set([o[1] for o in options if o[0] == 1])
    if picked_job == 2:
        path_options = set([o[1] for o in options if o[0] == 2])
    chosen_path = min(list(path_options))  # Pick path with lower number, as they are faster
    return picked_job, chosen_path  # random.choice([(1, 1), (1, 2), (2, 4), (2, 1), (2, 5)])


class GymWrapper(gym.Env):
    def __init__(self, policyTrainer, dp_max=12, max_sim_time=480):
        super(GymWrapper, self).__init__()
        self.dp_max = dp_max
        self.max_sim_time = max_sim_time
        self.disturbance_data = sim_data.disturbance_data
        self.simulation = Simulation(num_stations=1, num_cycles=self.dp_max, sim_time=self.max_sim_time, num_pals=3,
                                     disturbance_data=self.disturbance_data)
        self.dp_count = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float64)
        self.action_space = spaces.Discrete(5)
        self.actions = [(1, 1), (1, 2), (2, 4), (2, 1), (2, 5)]
        self.valid_actions_before_step = None
        self.last_valid_info = {"finished_products": {"Job 1": 0, "Job 2": 0}}
        self.reward_episode = 0
        self.episodes_reward_history = []
        self.finished_jobs_history = []
        self.valid_job_picked_history = []
        self.remaining_prod_time_T4_history = []
        self.truncated_history = []
        self.seed = 1
        self.blocked_time_history = []

    def reset(self, seed=None):
        set_random_seed(self.seed)
        print("Seed: ", self.seed)
        # print("Seed: ", self.policyTrainer.seed)
        """
        # super().reset(seed=seed)
        seed = np.random.randint(0, 99999)
        print("Seed: ", seed)
        """
        self.dp_count = 0
        self.simulation = Simulation(num_stations=1, num_cycles=self.dp_max, sim_time=self.max_sim_time, num_pals=3,
                                     disturbance_data=self.disturbance_data)
        self.simulation.env.process(run_until_decision_point(self.simulation, 'AGV 1',
                                                             self.simulation.prod_sys))
        info = {'Info': 'Not on first dp'} if self.dp_count == 0 else self.get_info()
        self.seed += 1
        self.reward_episode = 0
        return self.create_observation(), info

    def step(self, action_index: int):
        print("Valid actions: ", self.get_valid_actions())
        action = self.actions[action_index]
        picked_job, chosen_path = action
        print("Picked job: ", picked_job, "and path: ", chosen_path)
        self.valid_actions_before_step = self.get_valid_actions()
        if action in self.valid_actions_before_step:
            truncated = False
            remaining_prod_time_T4 = self.get_remaining_production_time()

            print("Remaining Production Time T4: ", remaining_prod_time_T4)
            self.remaining_prod_time_T4_history.append(remaining_prod_time_T4)
            self.valid_job_picked_history.append(picked_job)

            logger.info('Chosen routing decision (path) for next sim decision-point-run: %i' % chosen_path)
            logger.info('Chosen job picking decision for next sim decision-point-run: %i' % picked_job)

            # Update path for the next step with the current simulation
            self.simulation.update_decision(picked_job, chosen_path, self.dp_count)
            # Run until a decision point event
            self.simulation.env._simpy_dp_exit_event = DecisionPointExit(self.simulation.env)
            self.simulation.env.run(until=self.simulation.env._simpy_dp_exit_event)
            self.simulation.sim_stats.update_stats(self.simulation)

            # Print finished products if > 9
            info = self.get_info()
            if sum(info['finished_products'].values()) > 9:
                print(f"Finished Products: {info['finished_products']} Sum: {sum(info['finished_products'].values())}")

            reward = self.create_reward(action)
            terminated = True if self.simulation.env.now >= self.max_sim_time else False
            if terminated:
                self.episodes_reward_history.append(self.reward_episode)
                self.finished_jobs_history.append(sum(self.get_info()['finished_products'].values()))
                blocked_time = self.get_info()["total_block_duration_agvs"]
                self.blocked_time_history.append(blocked_time)
                print("Composition decisions: ", self.get_info()['decisions'])
                print("Info: ", self.get_info())
            self.truncated_history.append(0)
            self.dp_count += 1
        else:
            truncated = True
            self.truncated_history.append(1)
            reward = self.create_reward(action)
            self.episodes_reward_history.append(self.reward_episode)
            self.finished_jobs_history.append(sum(self.last_valid_info['finished_products'].values()))
            terminated = False
            info = {'Picked invalid action. No info about non existing state': True}
            blocked_time = self.last_valid_info["total_block_duration_agvs"]
            self.blocked_time_history.append(blocked_time)

        return self.create_observation(), reward, terminated, truncated, info

    def create_reward(self, action: tuple):
        # total_waiting_time = sum(np.sum(times) for times in kpis['agv_idle_times'].values())
        # reward_waiting_time = .5 / total_waiting_time if total_waiting_time > 0 else .5  # check if the value is zero
        if action in self.valid_actions_before_step:
            finished_products = self.simulation.sim_stats.get_kpis()['finished_products']
            finished_job_1, finished_job_2 = finished_products.values()
            print("Finished products: ", finished_job_1 + finished_job_2)  # Insert heuristic MAX number of jobs 11??
            reward = (.5 * finished_job_1 + .5 * finished_job_2) / 10
        else:
            reward = -.4
        print(f"Reward step: {reward:.3f}")
        self.reward_episode += reward  # Reward sim run
        return reward  # Reward step

    def get_info(self):
        self.last_valid_info = self.simulation.sim_stats.get_kpis()
        if self.last_valid_info["total_block_duration_agvs"] is not None:
            print("Blocked time: ", self.last_valid_info["total_block_duration_agvs"])
        return self.simulation.sim_stats.get_kpis()  # for custom callback

    def get_location(self):
        # Check which state the valid actions correspond to
        valid_actions = self.get_valid_actions()
        if valid_actions == [(1, 1), (1, 2)]:  # Second part of job 1, return path
            return 0
        elif valid_actions == [(1, 1), (1, 2), (2, 4), (2, 1)]:
            return 1 / 2
        elif valid_actions == [(1, 1), (1, 2), (2, 5)]:
            return 1

    def create_observation(self):
        location = self.get_location()  # Get the current state
        rem_prod_time_t4 = self.get_remaining_production_time() / 100  # Normalized 0-1

        all_path_disturbance_data = self.disturbance_data  # Get disturbance data for all paths

        # Flatten availabilities and shifts into np arrays
        availabilities_arr = np.array(list(all_path_disturbance_data['availabilities'].values()))
        shifts_arr = np.array(list(all_path_disturbance_data['shifts'].values()))

        # Normalize availabilities and shifts to the range [0, 1]
        availabilities_arr = availabilities_arr / np.max(availabilities_arr)
        shifts_arr = shifts_arr / np.max(shifts_arr)

        observation = np.array((location, rem_prod_time_t4))  # Combine all
        return observation

    def get_remaining_production_time(self):
        return self.simulation.remainingProdTime_T4

    def get_valid_actions(self):
        return self.simulation.options(return_all=False)

    def close(self):
        pass


if __name__ == "__main__":
    wrapper = GymWrapper("Run manually")
    jobs_per_time = {}
    job1_means, job2_means = [], []
    total_waiting_times = []  # List to store waiting times for each episode
    for average_time_t4 in range(0, 110):  # Run the simulation for each T4 time
        job1_sum, job2_sum = 0, 0
        num_average_runs = 10
        for _ in range(num_average_runs):
            job1, job2 = run_manual(average_time_t4)
            job1_sum += job1
            job2_sum += job2
            total_waiting_times.append(
                wrapper.last_valid_info['total_waiting_times_agv'])  # Add waiting time for this episode
        total_jobs = (job1_sum + job2_sum) / num_average_runs  # Calculate the average number of finished jobs
        jobs_per_time[average_time_t4] = total_jobs
        print(f"Switching to job 2 when T4 time is {average_time_t4} or more, achieves {total_jobs} jobs on average")
        job1_means.append(job1_sum / num_average_runs)
        job2_means.append(job2_sum / num_average_runs)
    total_average_job1 = sum(job1_means) / len(job1_means)
    total_average_job2 = sum(job2_means) / len(job2_means)
    print(f"Total average of finished Job 1: {total_average_job1}")
    print(f"Total average of finished Job 2: {total_average_job2}")

    # Calculate average waiting time for each station
    total_waiting_times_per_station = {'T0': 0.0, 'T4': 0.0, 'T5': 0.0, 'Flimo': 0.0}
    for waiting_times in total_waiting_times:
        for station, time in waiting_times.items():
            total_waiting_times_per_station[station] += time
    average_waiting_times_per_station = {station: time / len(total_waiting_times) for station, time in
                                         total_waiting_times_per_station.items()}
    print(f"Average waiting times per station: {average_waiting_times_per_station}")

    truncated_episodes = sum(wrapper.truncated_history)
    percentage_truncated = (truncated_episodes / len(wrapper.truncated_history)) * 100
    print(f"Percentage of truncated episodes: {percentage_truncated}%")

    total_blocked_time = sum(wrapper.blocked_time_history)
    print(f"Total blocked time for all episodes: {total_blocked_time}")

    plt.plot(list(range(0, 110)), (np.array(job1_means) + np.array(job2_means)), label='combined', color='blue')
    plt.plot(list(range(0, 110)), job1_means, label='Job 1', color='darkorange')
    plt.plot(list(range(0, 110)), job2_means, label='Job 2', color='red')
    job1_optimal_boundary = 53.027
    plt.axvline(x=job1_optimal_boundary, color='r', linestyle='--')
    plt.text(job1_optimal_boundary, 1, str(job1_optimal_boundary), color='r', ha='right', va='top')
    plt.title('Tot. production by choosing Job 1 if T4 time is less, else Job 2')
    plt.xlabel('T4 production time')
    plt.ylabel('Jobs done')
    plt.legend()
    average_job_for_t4_time_dir = 'average_job_for_t4_time'
    os.makedirs(average_job_for_t4_time_dir, exist_ok=True)
    plt.savefig(
        os.path.join(average_job_for_t4_time_dir, f'{datetime.now().strftime("%Y-%m-%d")}average_job_for_t4_time.png'),
        dpi=600)
    plt.show()

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(list(jobs_per_time.keys()), list(jobs_per_time.values()), marker='o')
    plt.xlabel('T4 Time')
    plt.ylabel('Jobs')
    plt.title('Jobs per T4 Time')
    plt.grid(True)
    plt.show()

'''
if __name__ == "__main__":
    jobs_per_time = {}
    for i in range(0, 100):
        jobs = run_manual(i)
        jobs_per_time[i] = jobs
        print(f"Jobs: {jobs} for T4 time: {i}")

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(list(jobs_per_time.keys()), list(jobs_per_time.values()), marker='o')
    plt.xlabel('T4 Time')
    plt.ylabel('Jobs')
    plt.title('Jobs per T4 Time')
    plt.grid(True)
    plt.show()
    '''
