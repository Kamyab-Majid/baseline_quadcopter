import csv
import gym
import envs
import numpy as np

ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
my_env.trajSelect[0] = 0
my_env.trajSelect[1] = 4
my_env.trajSelect[2] = 0
low_action = my_env.observation_space_domain['deltacol_p'][0]
high_action = my_env.observation_space_domain['deltacol_p'][1]

done = False
my_env.current_states = my_env.reset()
my_env.save_counter = 10000
# hover states
control_input_normilized = np.array((-0.03859327, -0.03859327, -0.03859327, -0.03859327))

while not done:
    # current_action = my_contr.Controller_model(my_env.current_states, my_env.dt * my_env.counter, action=sl_action)
    my_env.current_states, b, done, _ = my_env.step(control_input_normilized)
    control_input_normilized = 2 * (my_env.ctrl.w_cmd - low_action) / (high_action - low_action) - 1
    fields = my_env.ctrl.w_cmd
    with open("ctrl.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
my_env.reset()
# if my_env.best_reward > current_best_rew:
#     current_best_rew = my_env.best_reward
# with open("reward_step.csv", "a") as f:
#     writer = csv.writer(f)
#     writer.writerow(sl_action)
