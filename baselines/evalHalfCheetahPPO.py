import gymnasium as gym
import pybullet
import pybullet_envs
from stable_baselines3 import PPO

# Create the environment
env = gym.make("HalfCheetahBulletEnv-v0", render_mode="human")
# Load the trained model
filename = "halfcheetah_model.zip"
model = PPO.load(filename, env=env)

vec_env = model.get_env()
avg_rew = 0.0
for t in range(3):
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
avg_rew /= 3.0
print(f"Average reward over 3 trials: {avg_rew}")
env.close()
