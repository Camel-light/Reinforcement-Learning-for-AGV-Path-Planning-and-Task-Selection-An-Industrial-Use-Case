import random
import logging
from sim.decision_point import DecisionPointExit
from sim.simulation import Simulation, run_until_decision_point
import numpy as np
import matplotlib.pyplot as plt
import simulation_data as sim_data



"""This is the final version of the Q-table built on the Engels simulation. The hyperparameters(lr, gamma, epsilon, 
decay_rate) are not tuned/optimized. To learn about the remaining production timeT4, since the Q-tabel is discrete 
while time continuous, I fragmented/discretized remaining_time_T4 to remaining_time_T4_bin by braking it into 30 
bins. Therefore, each basic state (0, 1, 2), corresponding to one of 3 possible sets of actions, from the get_state 
method, gets extended to 30 possible basic state+bin(0, 1, 2, ....29). In the printed Q-table all the 90 states are 
printed each time, called "Observation (basic_state n°, bin n°)". Further development could remove bad practice of 
having 2 classes in the same file, hyper-tuning,..."""

logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(),
                                                     logging.FileHandler('sim_run.log', encoding='utf-8')])
logger = logging.getLogger(__name__)


# soll lernen küzeren Pfad zu nehmen. T4 Zeit remaining production.
# Reward printen
# Documentation writeup results
#
# Decrease lr over time. In Geron book.
# reduce epsilon as function of reward: https://stackoverflow.com/a/73688714
"""Alternatively, rather than relying only on chance for exploration, another approach is to encourage the 
exploration policy to try actions that it has not tried much before. This can be implemented as a bonus added to the 
Q-value estimates, as shown in Equation 18-6. Equation 18-6. Q-learning using an exploration function Q s, 
a α r + γ · max f Q s′, a′ , N s′, a′ a′ In this equation: • N(s′, a′) counts the number of times the action a′ was 
chosen in state s′. • f(Q, N) is an exploration function, such as f(Q, N) = Q + κ/(1 + N), where κ is a curiosity 
hyperparameter that measures how much the agent is attracted to the unknown."""


class QLearningAgent:
    def __init__(self, lr=.4, gamma=.7, num_states=90, num_actions=5):
        self.lr = lr
        self.gamma = gamma
        self.q_table = np.zeros((num_states, num_actions))
        self.updates_table = np.zeros((num_states, num_actions))
        self.action_index_map = {(1, 1): 0, (1, 2): 1, (2, 4): 2, (2, 1): 3, (2, 5): 4}
        self.state_index_map = {(0, i): i for i in range(30)}
        self.state_index_map.update({(1, i): i + 30 for i in range(30)})
        self.state_index_map.update({(2, i): i + 60 for i in range(30)})
        self.index_state_map = {v: k for k, v in self.state_index_map.items()}

    def update_q_value(self, observation, action, reward, new_observation, print_table=False):
        action_index = self.get_action_index(action)
        observation_index = self.get_state_index(observation)
        new_observation_index = self.get_state_index(new_observation)
        self.q_table[observation_index, action_index] = self.q_table[observation_index, action_index] + self.lr * (
                reward + self.gamma * np.max(self.q_table[new_observation_index, :]) - self.q_table[
            observation_index, action_index])
        self.updates_table[observation_index, action_index] = 1
        if print_table:
            self.print_q_table()

    def print_q_table(self):
        # Reverse the action_index_map
        index_action_map = {v: k for k, v in self.action_index_map.items()}

        # Determine the maximum width needed for the numbers
        max_width = max(len(f"{value:.2f}") for value in self.q_table.flatten())
        column_width = max(7, max_width)  # Ensure minimum width of 7

        # Determine the maximum length of the observation labels
        max_obs_label_len = max(len(f"Observation {k}") for k in self.index_state_map.values())

        print("\nObservation/Action    |",
              " | ".join(
                  [f"{str(index_action_map[action]):>{column_width}}" for action in self.action_index_map.values()]),
              "|")
        print("-" * (max_obs_label_len + 3 + len(self.action_index_map) * (column_width + 3)))

        for k, state_values in enumerate(self.q_table):
            state_actions_list = []
            for l, value in enumerate(state_values):
                if self.updates_table[k][l] == 1:  # Check if value was updated
                    state_actions_list.append(f"\033[41m{value:>{column_width}.2f}\033[0m")  # Highlight in dark red
                else:
                    state_actions_list.append(f"{value:>{column_width}.2f}")

            state_actions = " | ".join(state_actions_list)
            state_tuple = str(self.index_state_map[k])  # Get the state tuple from the index and convert it to a string
            print(f"Observation {state_tuple:<{max_obs_label_len - 10}} |", state_actions, "|")
        print("\n")
        self.updates_table.fill(0)  # Reset updates_table for the next episode

    def get_best_action(self, valid_options, epsilon, random_choice=False):
        if random_choice:
            action = random.choice(valid_options)
            return action
        # With probability epsilon, select a random action
        random_vs_epsilon = np.random.rand()
        print("Random vs epsilon: ", random_vs_epsilon, "vs. ", epsilon)
        if random_vs_epsilon < epsilon:
            action = random.choice(valid_options)
            print(f"Random action: {action} Action index: {self.get_action_index(action)}")
            job_types['random'] += 1
            return action
        # Otherwise, select the action with the highest Q-value
        else:
            observation_index = self.get_state_index(observation)
            q_values = [self.q_table[observation_index, self.get_action_index(action)] for action in valid_options]
            max_q_value_index = np.argmax(q_values)
            action = valid_options[max_q_value_index]
            print("Greedy action: ", action, "Action index: ", self.get_action_index(action), "Q-values: ", q_values)
            job_types['greedy'] += 1
            return action

    def get_action_index(self, action):
        return self.action_index_map[action]  # Map the action space to a single integer

    def get_state_index(self, state):
        return self.state_index_map[state]  # Map the state space to a single integer


agent = QLearningAgent()  # Initialize Q-learning agent


class EngelAGVSimulationEnv:
    def __init__(self, dp_max=12, max_sim_time=480):
        super(EngelAGVSimulationEnv, self).__init__()
        self.dp_max = dp_max
        self.max_sim_time = max_sim_time
        self.disturbance_data = sim_data.disturbance_data
        self.metrics = sim_data.metrics
        self.simulation = Simulation(num_stations=1, num_cycles=self.dp_max, sim_time=self.max_sim_time, num_pals=3,
                                     disturbance_data=self.disturbance_data)
        self.dp_count = 0
        self.total_waiting_times = {}
        self.seed = 1

    def reset(self):
        # Consider that setting seed at top means randomness is deterministic across runs.
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed += 1
        self.dp_count = 0
        self.simulation = Simulation(num_stations=1, num_cycles=self.dp_max, sim_time=self.max_sim_time, num_pals=3,
                                     disturbance_data=self.disturbance_data)
        self.simulation.env.process(run_until_decision_point(self.simulation, 'AGV 1', self.simulation.prod_sys))
        terminated = False
        return self.dp_count, terminated



    def get_state(self):
        # Check which state the valid options correspond to
        valid_options = self.simulation.options(return_all=False)
        remaining_time_T4 = self.simulation.remainingProdTime_T4
        print("Remaining production time T4: ", remaining_time_T4)
        remaining_time_T4_bin = int(remaining_time_T4 / 30)  # Discretize the remaining prod time  T4
        if valid_options == [(1, 1), (1, 2)]:  # Second part of job 1, return path
            return 0, remaining_time_T4_bin
        elif valid_options == [(1, 1), (1, 2), (2, 4), (2, 1)]:
            return 1, remaining_time_T4_bin
        elif valid_options == [(1, 1), (1, 2), (2, 5)]:
            return 2, remaining_time_T4_bin

    def create_reward(self):
        """function to build reward based on observation"""
        kpis = self.simulation.sim_stats.get_kpis()
        if kpis["total_block_duration_agvs"] is not None:
            print("Blocked time: ", kpis["total_block_duration_agvs"])

        # total_waiting_time = sum(np.sum(times) for times in kpis['agv_idle_times'].values())
        # reward_waiting_time = .5 / total_waiting_time if total_waiting_time > 0 else .5  # check if the value is zero

        finished_products = kpis['finished_products']
        finished_job_1, finished_job_2 = finished_products.values()
        reward = (.5 * finished_job_1 + .5 * finished_job_2) / 10
        #reward = (.719 * finished_job_1 + .281 * finished_job_2) / 10

        print("Reward finished products: ", reward)
        return reward

    def step(self, action):

        # check to make sure that every time a decision point comes, the action from gym is taken

        picked_job, chosen_path = action
        print("Picked job: ", picked_job, "and path: ", chosen_path)

        logger.info('Chosen routing decision (path) for next sim decision-point-run: %i' % chosen_path)
        logger.info('Chosen job picking decision for next sim decision-point-run: %i' % picked_job)

        # Update path for the next step with the current simulation
        self.simulation.update_decision(picked_job, chosen_path, self.dp_count)
        # Run until a decision point event
        self.simulation.env._simpy_dp_exit_event = DecisionPointExit(self.simulation.env)
        self.simulation.env.run(until=self.simulation.env._simpy_dp_exit_event)
        self.simulation.sim_stats.update_stats(self.simulation)

        self.dp_count += 1
        terminated = True if self.simulation.env.now >= self.max_sim_time else False
        reward = self.create_reward()

        # Get the new state
        new_observation = self.get_state()

        return new_observation, terminated, reward

    def close(self):
        pass


if __name__ == "__main__":

    env = EngelAGVSimulationEnv(dp_max=12, max_sim_time=480)

    num_episodes = 1000
    job_counter = {}
    job_types = {'random': 0, 'greedy': 0}
    finished_jobs_run = {}
    epsilons = []
    decay_rate = 0.04  # for epsilon

    for i in range(num_episodes):
        epsilon = max(0, np.exp(-decay_rate * i))
        epsilons.append(epsilon)
        terminated = env.reset()[0]
        print(f"Simulation run: {i + 1}")
        print("Epsilon: ", epsilon)

        reward_run = 0
        num_steps = 0
        relative_reward = 0
        max_q_value = float('-inf')

        while not terminated:
            print("Decision point n°: ", env.dp_count, "Simulation time: ", env.simulation.env.now, "Cycle finished: ",
                  env.simulation.cycle_finished, "Decision list length: ", len(env.simulation.decision_list),
                  "AGV location: ", env.simulation.agv.location)
            print('All options: ', env.simulation.options(return_all=True))
            valid_options = env.simulation.options(return_all=False)
            print('Valid options: ', valid_options)

            observation = env.get_state()
            action = agent.get_best_action(valid_options, epsilon, random_choice=False)
            chosen_job = action[0]
            job_counter[chosen_job] = job_counter.get(chosen_job, 0) + 1
            print("Observation before step: ", observation)
            new_observation, terminated, reward = env.step(action)
            print_table = terminated
            print("Observation after step: ", new_observation, "Terminated: ", terminated, "Reward: ", reward)
            agent.update_q_value(observation, action, reward, new_observation, print_table)
            # Update metrics after each step
            reward_run += reward
            num_steps += 1
            relative_reward = reward_run / num_steps if num_steps != 0 else reward_run
            print("Reward DP: ", reward, "Reward run: ", reward_run, "Num steps: ", num_steps, "Relative reward: ",
                  relative_reward, "Finished jobs: ", env.simulation.prod_sys.finished_products)
            max_q_value = max(max_q_value, np.max(agent.q_table))

        finished_jobs_run[i] = env.simulation.prod_sys.finished_products
        env.total_waiting_times[i] = env.simulation.sim_stats.get_kpis()['total_waiting_times_agv']

        # Store metrics for this episode
        env.metrics['total_rewards'].append(reward_run)
        env.metrics['num_steps'].append(num_steps)
        env.metrics['max_q_values'].append(max_q_value)
        env.metrics['relative_rewards'].append(relative_reward)

        env.close()

        # Total random experiment
        # Calculate the total counts of job1 and job2 across all runs
        total_job1 = sum(val.get('Job 1', 0) for val in finished_jobs_run.values())
        total_job2 = sum(val.get('Job 2', 0) for val in finished_jobs_run.values())
        total_jobs = total_job1 + total_job2
        average_jobs = total_jobs / num_episodes if num_episodes != 0 else 0

        # Average counts of job1 and job2
        average_job1 = total_job1 / num_episodes if num_episodes != 0 else 0
        average_job2 = total_job2 / num_episodes if num_episodes != 0 else 0

        print("=" * 100)
    print("=" * 100)
    print("=" * 100)
    # Calculate the total waiting times for each station across all runs
    total_waiting_times = {station: sum(times.get(station, 0) for times in env.total_waiting_times.values()) for
                           station in ['T0', 'T4', 'T5', 'Flimo']}
    # Average waiting times for each station
    average_waiting_times = {station: time / num_episodes if num_episodes != 0 else 0 for station, time in
                             total_waiting_times.items()}

    print(f"Average waiting times: {average_waiting_times}")
    # print(f"Average number of jobs when choosing actions randomly: {average_jobs}")
    print(f"Average number of job1: {average_job1}")
    print(f"Average number of job2: {average_job2}")

    print("Finished jobs| Tot. :", job_counter, "Random: ", job_types['random'], "Greedy: ", job_types['greedy'])
    # Get the run with the maximum finished jobs
    max_finished_jobs_run = max(finished_jobs_run.items(), key=lambda x: sum(x[1].values()))
    print(f"The run with the maximum finished jobs is run {max_finished_jobs_run[0]} with {sum(max_finished_jobs_run[1].values())} finished jobs,"
          f" composed of {max_finished_jobs_run[1]}")
    print(f"Waiting times for the optimal run: {env.total_waiting_times[max_finished_jobs_run[0]]}")
    # print("Total rewards: ", metrics['total_rewards'], "Relative rewards: ", metrics['relative_rewards'],
    # "Number of steps: ", metrics['num_steps'], "Max q values: ", metrics['max_q_values'])

    finished_jobs_values = [sum(val.values()) for val in finished_jobs_run.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(list(finished_jobs_run.keys()), finished_jobs_values, linewidth=.9)
    plt.xlabel('Episode')
    plt.ylabel('Number of Finished Jobs')
    plt.title('Number of Finished Jobs Over Time')
    plt.savefig('finished_jobs_over_time.png', dpi=300)
    plt.show()

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Relative Reward', color=color)
    ax1.plot(env.metrics['relative_rewards'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)  # already handled the x-label with ax1
    ax2.plot(epsilons, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('relative_reward.png', dpi=300)  # Save the plot
    plt.show()
