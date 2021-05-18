import numpy as np
from abc import ABC
import gym
from gym import spaces
from utils_main import save_files
import time
from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils
import config
from run_3D_simulation import main, init_data, quad_sim


class CustomEnv(gym.Env, ABC):
    def __init__(self):
        self.default_range = default_range = (-10, 10)
        self.velocity_range = velocity_range = (-10, 10)
        self.ang_velocity_range = ang_velocity_range = (-1, 1)
        self.q_range = qrange = (-0.3, 0.3)
        self.motor_range = motor_range = (400, 700)
        self.motor_d_range = motor_d_range = (-2000, 2000)
        self.observation_space_domain = {
            "x": default_range,
            "y": default_range,
            "z": default_range,
            "q0": (0.9, 1),
            "q1": qrange,
            "q2": qrange,
            "q3": qrange,
            "u": velocity_range,
            "v": velocity_range,
            "w": velocity_range,
            "p": ang_velocity_range,
            "q": ang_velocity_range,
            "r": ang_velocity_range,
            "wM1": motor_range,
            "wdotM1": motor_d_range,
            "wM2": motor_range,
            "wdotM2": motor_d_range,
            "wM3": motor_range,
            "wdotM3": motor_d_range,
            "wM4": motor_range,
            "wdotM4": motor_d_range,
        }
        self.states_str = list(self.observation_space_domain.keys())
        self.low_obs_space = np.array(list(zip(*self.observation_space_domain.values()))[0], dtype=float)
        self.high_obs_space = np.array(list(zip(*self.observation_space_domain.values()))[1], dtype=float)
        self.observation_space = spaces.Box(low=self.low_obs_space, high=self.high_obs_space, dtype=float)
        self.default_act_range = default_act_range = (-100, 100)
        self.action_space_domain = {
            "deltacol": default_act_range,
            "deltalat": default_act_range,
            "deltalon": default_act_range,
            "deltaped": default_act_range,
            # "f1": (0.1, 5), "f2": (0.5, 20), "f3": (0.5, 20), "f4": (0.5, 10),
            # "lambda1": (0.5, 10), "lambda2": (0.1, 5), "lambda3": (0.1, 5), "lambda4": (0.1, 5),
            # "eta1": (0.2, 5), "eta2": (0.1, 5), "eta3": (0.1, 5), "eta4": (0.1, 5),
        }
        self.low_action_space = np.array(list(zip(*self.action_space_domain.values()))[0], dtype=float)
        self.high_action_space = np.array(list(zip(*self.action_space_domain.values()))[1], dtype=float)
        self.action_space = spaces.Box(low=self.low_action_space, high=self.high_action_space, dtype=float)
        self.min_reward = -13 
        self.high_action_diff = 0.2
        obs_header = str(list(self.observation_space_domain.keys()))[1:-1]
        act_header = str(list(self.action_space_domain.keys()))[1:-1]
        self.header = "time, " + act_header + ", " + obs_header + ", reward" + ", control reward"
        self.saver = save_files()
        self.reward_array = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=float)
        self.constant_dict = {
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 0.0,
            "q": 0.0,
            "r": 0.0,
            "fi": 0.0,
            "theta": 0.0,
            "si": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
        }
        self.start_time = time.time()
        self.Ti = 0
        self.Ts = 0.005
        self.Tf = 2
        self.reward_check_time = 0.7 * self.Tf
        self.ifsave = 0
        self.ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
        self.trajSelect = np.zeros(3)
        self.ctrlType = self.ctrlOptions[0]
        self.trajSelect[0] = 0
        self.trajSelect[1] = 4
        self.trajSelect[2] = 0
        self.quad = Quadcopter(self.Ti)
        self.traj = Trajectory(self.quad, self.ctrlType, self.trajSelect)
        self.ctrl = Control(self.quad, self.traj.yawType)
        self.wind = Wind("None", 2.0, 90, -15)
        self.sDes = self.traj.desiredState(0, self.Ts, self.quad)  # np.zeros(19)
        self.ctrl.controller(self.traj, self.quad, self.sDes, self.Ts)
        self.numTimeStep = int(self.Tf / self.Ts + 1)
        self.longest_num_step = 0
        self.best_reward = float("-inf")
        self.diverge_counter = 0
        self.diverge_list = []

    def reset(self):
        #  initialization
        self.t = 0
        sDes = self.traj.desiredState(self.t, self.Ts, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.Ts)
        (
            self.t_all,
            self.s_all,
            self.pos_all,
            self.vel_all,
            self.quat_all,
            self.omega_all,
            self.euler_all,
            self.sDes_traj_all,
            self.sDes_calc_all,
            self.w_cmd_all,
            self.wMotor_all,
            self.thr_all,
            self.tor_all,
        ) = init_data(self.quad, self.traj, self.ctrl, self.numTimeStep)
        self.all_actions = np.zeros((self.numTimeStep, len(self.high_action_space)))
        self.control_rewards = np.zeros((self.numTimeStep, 1))
        self.all_rewards = np.zeros((self.numTimeStep, 1))
        self.t_all[0] = self.Ti
        observation = self.quad.state.copy()
        self.pos_all[0, :] = self.quad.pos
        self.vel_all[0, :] = self.quad.vel
        self.quat_all[0, :] = self.quad.quat
        self.omega_all[0, :] = self.quad.omega
        self.euler_all[0, :] = self.quad.euler
        self.sDes_traj_all[0, :] = self.traj.sDes
        self.sDes_calc_all[0, :] = self.ctrl.sDesCalc
        self.w_cmd_all[0, :] = self.ctrl.w_cmd
        self.wMotor_all[0, :] = self.quad.wMotor
        self.thr_all[0, :] = self.quad.thr
        self.tor_all[0, :] = self.quad.tor
        self.counter = 0
        self.save_counter = 0
        self.control_input = np.array((522.984714071469, 522.984714071469, 522.984714071469, 522.984714071469), dtype=float)
        self.find_next_state()
        self.jj = 0
        self.counter = 0


        # self.quad.state[0:12] = self.initial_states = list((np.random.rand(12) * 0.02 - 0.01))
        # self.quad.state[0:12] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.done = False
        for i in range(len(self.high_obs_space)):
            current_range = self.observation_space_domain[self.states_str[i]]
            observation[i] = (observation[i] - sum(current_range) / 2) / current_range[1]
        self.s_all[0, :] = observation
        return observation

    def action_wrapper(self, current_action: list) -> np.array:
        # [-1, 1]
        current_action = np.clip(current_action, -1, 1)
        self.control_input = (
            current_action * (self.high_action_space - self.low_action_space) / 2
            + (self.high_action_space + self.low_action_space) / 2 + 522.984714071469
        )

    def find_next_state(self) -> list:
        self.quad.update(self.t, self.Ts, self.control_input, self.wind)  # self.control_input = self.ctrl.w_cmd
        # self.quad.update(self.t, self.Ts, self.ctrl.w_cmd, self.wind)
        self.t += self.Ts
        # self.control_input = self.ctrl.w_cmd.copy()
        # sDes = self.traj.desiredState(self.t, self.Ts, self.quad)
        # print('state', self.quad.state)
        # print(' ctrl', self.ctrl.w_cmd)
        # self.ctrl.controller(self.traj, self.quad, sDes, self.Ts)

    def observation_function(self) -> list:
        i = self.counter
        quad = self.quad
        self.t_all[i] = self.t
        self.s_all[i, :] = quad.state
        self.pos_all[i, :] = quad.pos
        self.vel_all[i, :] = quad.vel
        self.quat_all[i, :] = quad.quat
        self.omega_all[i, :] = quad.omega
        self.euler_all[i, :] = quad.euler
        self.sDes_traj_all[i, :] = self.traj.sDes
        self.sDes_calc_all[i, :] = self.ctrl.sDesCalc
        self.w_cmd_all[i, :] = self.ctrl.w_cmd
        self.wMotor_all[i, :] = quad.wMotor
        self.thr_all[i, :] = quad.thr
        self.tor_all[i, :] = quad.tor
        observation = list(quad.state.copy())
        for iii in range(len(self.high_obs_space)):
            current_range = self.observation_space_domain[self.states_str[iii]]
            observation[iii] = 2 * (observation[iii] - current_range[0]) / (current_range[1] - current_range[0]) -1
        self.s_all[i, :] = observation
        return observation

    def reward_function(self, observation) -> float:
        # add reward slope to the reward
        # TODO: normalizing reward
        # TODO: adding reward gap
        # error_x = np.linalg.norm((abs(self.quad.state[0:3]).reshape(12)), 1)
        # error_ang = np.linalg.norm((abs(self.quad.state[3:7]).reshape(12)), 1)
        # error_v = np.linalg.norm((abs(self.quad.state[7:10]).reshape(12)), 1)
        # error_vang = np.linalg.norm((abs(self.quad.state[10:12]).reshape(12)), 1)
        # error = -np.linalg.norm((abs(self.s_all[self.counter, 0:12]).reshape(12)), 1) + 0.05
        error = -np.linalg.norm(observation[0:12], 1) + 1
        self.control_rewards[self.counter] = self.all_rewards[self.counter] = reward = error
        # for i in range(12):
        #     self.reward_array[i] = abs(self.current_states[i]) / self.default_range[1]
        # reward = self.all_rewards[self.counter] = -sum(self.reward_array) + 0.17 / self.default_range[1]
        # # control reward
        # reward += 0.05 * float(
        #     self.control_rewards[self.counter] - self.control_rewards[self.counter - 1]
        # )  # control slope
        # reward += -0.005 * sum(abs(self.all_actions[self.counter]))  # input reward
        # for i in (self.high_action_diff - self.all_actions[self.counter] - self.all_actions[self.counter - 1]) ** 2:
        #     reward += -min(0, i)
        return reward

    def check_diverge(self) -> bool:
        for i in range(len(self.high_obs_space)):
            if (abs(self.quad.state[i])) > self.high_obs_space[i]:
                self.diverge_list.append((tuple(self.observation_space_domain.keys())[i], self.quad.state[i]))
                self.saver.diverge_save(tuple(self.observation_space_domain.keys())[i], self.quad.state[i])
                self.diverge_counter += 1
                if self.diverge_counter == 2000:
                    self.diverge_counter = 0
                    print((tuple(self.observation_space_domain.keys())[i], self.quad.state[i]))
                self.jj = 1
                return True
        if self.counter >= self.numTimeStep - 1:  # number of timesteps
            print("successful")
            return True
        # after self.reward_check_time it checks whether or not the reward is decreasing
        # if self.counter > self.reward_check_time / self.Ts:
        #     prev_time = int(self.counter - self.reward_check_time / self.Ts)
        #     diverge_criteria = (
        #         self.all_rewards[self.counter] - sum(self.all_rewards[prev_time:prev_time - 10]) / 10
        #     )
        #     if diverge_criteria < -1:
        #         print("reward_diverge")
        #         self.jj = 1
        #         return True
        bool_1 = any(np.isnan(self.quad.state))
        bool_2 = any(np.isinf(self.quad.state))
        if bool_1 or bool_2:
            self.jj = 1
            print("state_inf_nan_diverge")
        return False

    def done_jobs(self) -> None:
        counter = self.counter
        self.save_counter += 1
        current_total_reward = sum(self.all_rewards)
        if self.save_counter >= 100:
            print("current_total_reward: ", current_total_reward)
            self.save_counter = 0
            self.saver.reward_step_save(self.best_reward, self.longest_num_step, current_total_reward, counter)
        if counter >= self.longest_num_step:
            self.longest_num_step = counter
        if current_total_reward >= self.best_reward:
            self.best_reward = sum(self.all_rewards)
            ii = self.counter + 1
            self.saver.best_reward_save(
                self.t_all[0:ii],
                self.w_cmd_all[0:ii],
                self.s_all[0:ii],
                self.all_rewards[0:ii],
                self.control_rewards[0:ii],
                self.header,
            )

    def step(self, current_action):
        self.action_wrapper(current_action)
        try:
            self.find_next_state()
        except OverflowError or ValueError or IndexError:
            self.jj = 1
        observation = self.observation_function()
        reward = self.reward_function(observation)
        self.done = self.check_diverge()
        if self.jj == 1:
            observation = list((self.s_all[self.counter]) - self.default_range[0])
            reward = self.min_reward + self.counter*self.Tf/self.Ts/2
        if self.done:
            if self.counter > 0:
                print(self.counter)
            self.done_jobs()

        self.counter += 1
        # self.make_constant(list(self.constant_dict.values()))
        return observation, reward, self.done, {}

    def make_constant(self, true_list):
        for i in range(len(true_list)):
            if i == 1:
                self.quad.state[i] = self.initial_states[i]
