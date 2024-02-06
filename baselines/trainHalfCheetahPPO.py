import gymnasium as gym
import pybullet
import pybullet_envs
from stable_baselines3 import PPO
import argparse
import sys
import os

# Total number of steps
NSTEPS = 5*1e7

def trainModel(seed=721, directory=os.getcwd()):
    # Create the environment
    env = gym.make("HalfCheetahBulletEnv-v0")
    env.reset(seed=seed)
    #env.render()

    # Create the model: we used the PPO algorithm with an MLP policy
    model = PPO("MlpPolicy", env, verbose=1)
    # Training of the model
    model.learn(total_timesteps=NSTEPS)
    # Save the model
    filename = "halfcheetahPPO_modelS" + str(seed) + ".zip"
    model.save(os.path.join(directory, filename))
    env.close()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='random generator seed', type=int, default=1)
    parser.add_argument('-d', '--directory', help='directory containing output files', type=str, default=os.getcwd())

    args = parser.parse_args()

    trainModel(seed=args.seed, directory=args.directory)

if __name__ == "__main__":
    main(sys.argv)
