import gymnasium as gym
import evorobotpy_envs
from evorobotpy_envs import *
from stable_baselines3 import PPO

NSTEPS = 5*1e4

env = gym.make("evorobotpy_envs/GridWorld-v0")#, render_mode="human")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=NSTEPS)
# Save the model
filename = "gridworld_model.zip"
model.save(filename)
"""
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()
"""
env.close()
