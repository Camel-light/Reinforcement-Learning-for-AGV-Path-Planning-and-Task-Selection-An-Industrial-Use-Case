import gymnasium as gym
from gymnasium import spaces
import logging
from sim.decision_point import DecisionPointExit
from sim.simulation import Simulation, run_until_decision_point
import numpy as np
import simulation_data as sim_data
from stable_baselines3.common.utils import set_random_seed

logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(),
                                                     logging.FileHandler('sim_run.log', encoding='utf-8')])
logger = logging.getLogger(__name__)


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
        self.truncated_vs_terminated = []
        self.policyTrainer = policyTrainer  # Fix for engel.sim vs engel.sim_test GymWrapper instances interfering seeds

    def reset(self, seed=None):
        set_random_seed(self.policyTrainer.seed)
        print("Seed: ", self.policyTrainer.seed)
        """
        # super().reset(seed=seed)
        seed = np.random.randint(0, 99999)
        print("Seed: ", seed)
        """
        # print("Seed counter: ", self.seed_counter)
        self.dp_count = 0
        self.simulation = Simulation(num_stations=1, num_cycles=self.dp_max, sim_time=self.max_sim_time, num_pals=3,
                                     disturbance_data=self.disturbance_data)
        self.simulation.env.process(run_until_decision_point(self.simulation, 'AGV 1',
                                                             self.simulation.prod_sys))
        info = {'reset info': 'bla'} if self.dp_count == 0 else self.get_info()
        # self.policyTrainer.seed += 1
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
                self.truncated_vs_terminated.append(0)
            self.dp_count += 1
        else:
            truncated = True
            self.truncated_vs_terminated.append(1)
            reward = self.create_reward(action)
            self.episodes_reward_history.append(self.reward_episode)
            self.finished_jobs_history.append(sum(self.last_valid_info['finished_products'].values()))
            terminated = False
            info = {'Picked invalid action. No info about non existing state': True}

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
