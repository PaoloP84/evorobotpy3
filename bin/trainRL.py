import gymnasium as gym
import pybullet
import pybullet_envs
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
import numpy as np
import argparse
import sys
import os
import configparser
import evorobotpy_envs
from evorobotpy_envs import *

def readConfig(filename):
    environment = None
    algo = 'PPO'
    policy = 'MlpPolicy'
    maxsteps = 5e7
    seed = 1
    folder = os.getcwd()
    optdict = dict()
    # Read data from configuration file
    config = configparser.ConfigParser()
    config.read(filename)
    options = config.options("EXP")
    for o in options:
        found = 0
        if o == "environment":
            environment = config.get("EXP","environment")
            found = 1
        if o == "algorithm":
            algo = config.get("EXP","algorithm")
            found = 1
        if o == "policy":
            policy = config.get("EXP","policy")
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
    # Now look for the [ENV] section (if any)
    try:
        options = config.options("ENV")
        for o in options:
            optdict[o] = config.get("ENV",o)
    except:
        print("File %s does not contain section ENV" % filename)
        pass
    return environment, algo, policy, maxsteps, seed, folder, optdict

def trainModel(filename):
    # Read configuration file
    environment, algo, policy, maxsteps, seed, folder, options = readConfig(filename)
    if environment is None:
        print("Configuration file %s does not contain an environment name, which is mandatory!!!" % filename)
        sys.exit()
    # Create the directory if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Create the environment
    env = None
    if "Er" in environment:                   # Er environment (implemented in C++ and wrapped with Cython)
        ErProblem = __import__(environment)
        env = ErProblem.PyErProblem()   
    elif "Bullet" in environment:             # Pybullet environment 
        try:
            env = gym.make(environment, options=optdict)
        except:
            print(f"Environment {environment} does not accept options as parameter...")
            env = gym.make(environment)
    elif "Custom" in environment:              # Custom environment
        customEnv = __import__(environment)
        try:
            env = customEnv.customEnv(options=optdict)
        except:
            print(f"Environment {environment} does not accept options as parameter...")
            env = customEnv.customEnv()     
    else:                         
        try:
            try:
                env = gym.make(environment, options=options)
            except:
                print(f"Environment {environment} does not accept options as parameter...")
                env = gym.make(environment)
        except:
            print(f"Environment {environment} is not registered in gymnasium... Look for it in evorobotpy_envs!")
            try:
                envname = os.path.join("evorobotpy_envs", environment)
                try:
                    env = gym.make(envname, options=optdict)
                except:
                    print(f"Environment {envname} might not accept <options> dict as parameter")
                    env = gym.make(envname)
            except:
                print(f"Environment {environment} (passed to gym.make() with argument {envname}) not found!!!")
    if env is None:
        print("Failure in environment creation!!!")
        sys.exit()
    # Reset the environment
    env.reset(seed=seed)
    # Create the model: we used a RL algorithm with an MLP policy
    if algo == 'A2C':
        model = A2C(policy, env, verbose=1)
    elif algo == 'DDPG':
        model = DDPG(policy, env, verbose=1)
    elif algo == 'DQN':
        model = DQN(policy, env, verbose=1)
    elif algo == 'PPO':
        model = PPO(policy, env, verbose=1)
    elif algo == 'SAC':
        model = SAC(policy, env, verbose=1)
    elif algo == 'TD3':
        model = TD3(policy, env, verbose=1)
    # Training of the model
    model.learn(total_timesteps=maxsteps)
    # Save the model
    outfile = environment + "_" + algo + "_" + policy + "_modelS" + str(seed) + ".zip"
    model.save(os.path.join(folder, outfile))
    env.close()

def main(argv):
    parser = argparse.ArgumentParser(description='Train RL')
    parser.add_argument('-f', '--filename', help='configuration file', type=str, default='config.ini')

    args = parser.parse_args()

    trainModel(args.filename)

if __name__ == "__main__":
    main(sys.argv)
