from stable_baselines3.common.env_checker import check_env
import gym
import envs
ENV_ID = "CustomEnv-v0"
my_env = gym.make(ENV_ID)
# It will check your custom environment and output additional warnings if needed
check_env(my_env)