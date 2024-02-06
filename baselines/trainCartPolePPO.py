import gymnasium as gym

from stable_baselines3 import PPO

NSTEPS = 1e5

env = gym.make("CartPole-v1")#, render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=NSTEPS)
# Save the model
filename = "cartpole_model.zip"
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
