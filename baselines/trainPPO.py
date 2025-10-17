import gymnasium as gym
import pybullet
import pybullet_envs
from stable_baselines3 import PPO
import numpy as np
import argparse
import sys
import os
import configparser

# Total number of steps (default)
NSTEPS = 5e7

def readConfig(filename):
    environment = None
    maxsteps = NSTEPS
    seed = 1
    folder = os.getcwd()
    # Read data from configuration file
    config = configparser.ConfigParser()
    config.read(filename)
    options = config.options("EXP")
    for o in options:
        found = 0
        if o == "environment":
            environment = config.get("EXP","environment")
            found = 1
        if o == "maxmsteps":
            maxsteps = 1e6 * config.getint("EXP","maxmsteps")
            found = 1
        if o == "seed":
            seed = config.getint("EXP","seed")
            found = 1
        if o == "folder":
            folder = config.get("EXP","folder")
            found = 1
        if found == 0:
            print("\033[1mOption %s in section [EXP] of %s file is unknown\033[0m" % (o, filename))
            sys.exit()
    return environment, maxsteps, seed, folder

def trainModel(filename):
    # Read configuration file
    environment, maxsteps, seed, folder = readConfig(filename)
    if environment is None:
        print("Configuration file %s does not contain an environment name, which is mandatory!!!" % filename)
        sys.exit()
    # Create the directory if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create the environment
    env = gym.make(environment)
    # Reset the environment
    env.reset(seed=seed)
    # Create the model: we used the PPO algorithm with an MLP policy
    model = PPO("MlpPolicy", env, verbose=1)
    # Training of the model
    model.learn(total_timesteps=maxsteps)
    # Save the model
    outfile = environment + "_PPO_modelS" + str(seed) + ".zip"
    model.save(os.path.join(folder, outfile))
    env.close()

def main(argv):
    parser = argparse.ArgumentParser(description='Train PPO')
    parser.add_argument('-f', '--filename', help='configuration file', type=str, default='config.ini')

    args = parser.parse_args()

    trainModel(args.filename)

if __name__ == "__main__":
    main(sys.argv)
