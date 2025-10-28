import gymnasium as gym
import pybullet
import pybullet_envs
import neat
import numpy as np
import sys
import os
import argparse
import configparser
import evorobotpy_envs
from evorobotpy_envs import *
import multiprocessing

# Wrapper of the NEAT algorithm from Stanley and Miikkulainen (2002). The code calls functions
# from neat implementation (see https://neat-python.readthedocs.io/en/latest/ and 
# https://github.com/CodeReclaimers/neat-python). Neat can be downloaded with pip: pip install neat-python

# Simulation parameters
sim_params = dict()

def readConfig(filename):
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
            sim_params['algo'] = config.get("EXP","algorithm")
            found = 1
        if o == "maxmsteps":
            sim_params['maxsteps'] = 1e6 * config.getint("EXP","maxmsteps")
            found = 1
        if o == "network":
            sim_params['net'] = config.get("EXP","network")
            found = 1
        if o == "ntrials":
            sim_params['ntrials'] = config.getint("EXP","ntrials")
            found = 1
        if o == "nsteps":
            sim_params['nsteps'] = config.getint("EXP","nsteps")
            found = 1
        if o == "folder":
            sim_params['folder'] = config.get("EXP","folder")
            found = 1
        if found == 0:
            print(f"\033[1mOption {o} in section [EXP] of {filename} file is unknown\033[0m")
            sys.exit()
    # Now look for the [ENV] section (if any)
    try:
        optdict = dict()
        options = config.options("ENV")
        for o in options:
            optdict[o] = config.get("ENV",o)
        sim_params['options'] = optdict
    except:
        print(f"File {filename} does not contain section ENV")
        pass
        
    return environment
    
def evalGenome(genome, config):
    #print(sim_params['ntrials'], sim_params['nsteps'])
    # Default parameters to run a simulation
    env = None
    network = 'FeedForward'
    ntrials = 1
    nsteps = 1
    # Try to recover parameters from the simulation parameters dictionary
    try:
        env = sim_params['env']
    except:
        print("Simulation parametrs do not contain any environment, but this is mandatory. Stop training!")
        sys.exit()
    try:
        network = sim_params['net']
        assert network == 'FeedForward' or network == 'Recurrent', f"Invalid network type {network}"
    except:
        print("Simulation parametrs do not contain any network type. We use feed forward as default.")
    try:
        ntrials = sim_params['ntrials']
    except:
        print("Simulation parametrs do not contain ntrials. We set it to 1.")
    try:
        nsteps = sim_params['nsteps']
    except:
        print("Simulation parametrs do not contain nsteps. We set it to 1.")
    if network == 'FeedForward':
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        net = neat.nn.RecurrentNetwork.create(genome, config)
    # Run a simulation
    fitness = 0.0
    # Loop over trials
    for t in range(ntrials):
        fit = 0.0
        ob, _ = env.reset()
        ob = np.append(ob, 0.5) # To be tuned for different types of problems
        # Loop over steps
        for s in range(nsteps):
            ac = net.activate(list(ob))
            ob, reward, terminated, truncated, info = env.step(ac)
            ob = np.append(ob, 0.5) # To be tuned for different types of problems
            fit += reward
            if terminated or truncated:
                break
        fit /= s
        # Update fitness
        fitness += fit
    # Average fitness over the number of trials
    fitness /= ntrials
    return fit

def trainNEAT(filename):
    # Read configuration file
    environment = readConfig(filename)
    # Check whether a valid environment has been specified in the configuration file
    if environment is None:
        print(f"Configuration file {filename} does not contain an environment name, which is mandatory!!!")
        sys.exit()
    # Get the folder
    folder = os.getcwd()
    try:
        folder = sim_params['folder']
    except:
        pass
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
            print(f"Environment {environment} might not accept <options> dict as parameter")
            env = gym.make(environment)
    elif "Custom" in environment:              # Custom environment
        customEnv = __import__(environment)
        try:
            env = customEnv.customEnv(options=optdict)
        except:
            print(f"Environment {environment} might not accept <options> dict as parameter")
            env = customEnv.customEnv()     
    else:                         
        try:
            try:
                env = gym.make(environment, options=options)
            except:
                print(f"Environment {environment} might not accept <options> dict as parameter")
                env = gym.make(environment)
        except:
            print(f"Environment {environment} is not registered in gymnasium... Look for it in evorobotpy_envs!")
            try:
                envname = os.path.join("evorobotpy_envs", environment)
                try:
                    env = gym.make(envname, options=options)
                except:
                    print(f"Environment {envname} might not accept <options> dict as parameter")
                    env = gym.make(envname)
            except:
                print(f"Environment {environment} (passed to gym.make() with argument {envname}) not found!!!")
    if env is None:
        print("Failure in environment creation!!!")
        sys.exit()
    # Copy the environment in the simulation parameters
    sim_params['env'] = env
    # Get the algorithm
    algo = sim_params['algo']
    if algo != 'NEAT':
        print(f"Unknown algorithm {algo}")
        sys.exit()
        
    # Load NEAT specific parameters from local configuration file
    local_dir = os.getcwd()#path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.ini')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
                         
    # Create population
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evalGenome)
    winner = pop.run(pe.evaluate)
    # Print the best performing/performance                     
    print(winner)

def main(argv):
    parser = argparse.ArgumentParser(description='Train NEAT')
    parser.add_argument('-f', '--filename', help='configuration file', type=str, default='config.ini')

    args = parser.parse_args()

    trainNEAT(args.filename)

if __name__ == "__main__":
    main(sys.argv)
         
