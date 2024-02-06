import gymnasium as gym
import evorobotpy_envs
from evorobotpy_envs import *
from stable_baselines3 import PPO

env = gym.make("evorobotpy_envs/GridWorld-v0", render_mode="human")
# Load the trained model
filename = "gridworld_model.zip"
model = PPO.load(filename, env=env)

vec_env = model.get_env()
avg_rew = 0.0
for t in range(10):
    obs = vec_env.reset()
    rews = 0.0
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        rews += reward
        vec_env.render()
        if done:
            break
        # VecEnv resets automatically
        # if done:
        #   obs = env.reset()
    #rews /= 1000.0
    print(f"Episode {t + 1}: reward = {rews}")
    avg_rew += rews
avg_rew /= 10.0
print(f"Average reward over 10 trials: {avg_rew}")
env.close()
