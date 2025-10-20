#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it
   es.py runs an evolutionary expriment or post-evaluate an evolved robot/s
   type python3 es.py for help
   Requires policy.py, evoalgo.py, and salimans.py
   Also requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py   
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin 
"""


import numpy as np
import configparser
import argparse
import sys
import os
import subprocess
import stat
from utils import str2bool
import evorobotpy_envs
from evorobotpy_envs import *

environment = None                      # the problem 
algoname = "OpenAI-ES"                  # evolutionary algorithm
nagents = 1
optdict = dict()

# Parse the [ADAPT] section of the configuration file
def parseConfigFile(filename):
    global environment
    global algoname
    global nagents
    global optdict

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        # Section EVAL
        options = config.options("EXP")
        for o in options:
            found = 0
            if o == "environment":
                environment = config.get("EXP","environment")
                found = 1
            if o == "algo":
                algoname = config.get("EXP","algo")
                found = 1
              
            if found == 0:
                print("\033[1mOption %s in section [EXP] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()
        # We need here to load the number of agents from section policy
        options = config.options("POLICY")
        for o in options:
            if o == "nrobots":
                optdict[o] = config.getint("POLICY","nrobots")
        # Here we check any possible optional parameter to be used for the creation of the environment (if ENV section exists)
        try:
            options = config.options("ENV")
            for o in options:
                optdict[o] = config.get("ENV",o)
        except:
            print("File %s does not contain section ENV" % filename)
            pass
    else:
        print("\033[1mERROR: configuration file %s does not exist\033[0m" % (filename))
        sys.exit()

def helper():
    print("Program Arguments: ")
    print("-f [fileini]              : the file containing the hyperparameters(mandatory)")
    print("-s [integer]              : the number used to initialize the seed")
    print("-n [integer]              : the number of replications to be run")
    print("-p                        : the flag to be used to postevaluate an agent (see parameter -g)")
    print("-g [filename]             : the name of teh file containing the parameters of the agent")
    print("-d                        : the flag that permit to show the activation of neurons during postevaluation")    
    print("-o [directory]            : the directory where all output files are stored (default current dir)")
    print("-w [integer]              : the number of workers used by parallel implementations")
    print("")
    print("The .ini file contains the following [EXP], [POLICY] and [ALGO] parameters:")
    print("[EXP]")
    print("environment [string]      : environment (default 'CartPole-v0'")
    print("algo [string]             : adaptive algorithm (default 'OpenAI-ES'), SSS, coevo, coevo2" )
    print("[POLICY]")
    print("nrobots [integer]         : number of robots (default 1)")
    print("heterogeneous [integer]   : whether robots are heterogeneous (default 0)")
    print("episodes [integer]        : number of evaluation episodes (default 1)")
    print("pepisodes [integer]       : number of post-evaluation episodes (default 0)")
    print("maxsteps [integer]        : number of evaluation steps [for EREnvs only] (default 1000)")
    print("nhiddens [integer]        : number of hidden x layer (default 50)")
    print("nlayers [integer]         : number of hidden layers (default 1)")
    print("bias [0/1]                : whether we have biases (default 0)")
    print("out_type [integer]        : type of output: 1=logistic, 2=tanh, 3=linear, 4=binary (default 2)")
    print("nbins [integer]           : number of bins 1=no-beans (default 1)")
    print("architecture [0/1/2/3]    : network architecture 0=feedforward 1=recurrent 2=fullrecurrent 3=lstm recurrent (default 0)")
    print("afunction [1/2/3]         : the activation function of neurons 1=logistic 2=tanh 3=linear (default 2)")
    print("winit [0/1/2]             : weight initialization 0=xavier 1=norm incoming 2=uniform (default 0)")
    print("action_noise [0/1/2]      : action noise 0=none, 1=gaussian 2=diagonal gaussian (default 0)")
    print("action_noise_range        : action noise range (default 0.01)")   
    print("normalized [0/1]          : whether or not the input observations are normalized (default 1)")
    print("clip [0/1]                : whether we clip observation in [-5,5] (default 0)")
    print("[ALGO]   parameters of the OpenAI-ES algorithm, the default algorithm")
    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
    print("stepsize [float]          : learning stepsize (default 0.01)")
    print("samplesize [int]          : popsize/2 (default 20)")
    print("noiseStdDev [float]       : samples noise (default 0.02)")
    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
    print("symseed [0/1]             : same environmental seed to evaluate symmetrical samples [default 1]")
    print("saveeach [integer]        : same data every N minutes [default 60]")
    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
    print("")
    sys.exit()


def main(argv):
    global environment
    global algoname
    global nagents
    global optdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fileini', help='name of ini file', type=str, default="")
    parser.add_argument('-s', '--seed', help='random generator seed', type=int, default=1)
    parser.add_argument('-n', '--nreplications', help='number of replications', type=int, default=1)
    parser.add_argument('-w', '--workers', help='number of workers', type=int, default=1)
    parser.add_argument('-p', '--postevaluate', help='postevaluate an agent', action='store_true', default=False)
    parser.add_argument('-g', '--genofile', help='name of the file containing the parameters of the agent', type=str, default=None)
    parser.add_argument('-d', '--displayneurons', help='show the activation of the neurons', action='store_true', default=False)
    parser.add_argument('-o', '--folder', help='file folder', type=str, default=".")    
    parser.add_argument('-r', '--render', help='rendering option', type=str2bool, nargs='?', const=True, default=False)
    
    args = parser.parse_args()
    
    if (len(args.fileini) == 0):    # if the name of the inifile is not specified display help information
        print("\033[1mERROR: You need to specify an .ini file\033[0m")
        helper()
        sys.exit(-1)

    parseConfigFile(args.fileini)   # load hyperparameters from the ini file

    availableAlgos = ('OpenAI-ES', 'SSS', 'CMA-ES', 'xNES', 'sNES', 'GA', 'generational', 'Hill-Climber', 'evostick', 'generalist', 'coevo2', 'coevo', 'archivestar1', 'coevoarch', 'coevosinglepop', 'archivestar1singlepop', 'coevoarchsinglepop')   # check whether the user specified a valid algorithm
    if algoname not in availableAlgos:
        print("\033[1mAlgorithm %s is unknown\033[0m" % algoname)
        print("Please use one of the following algorithms:")
        for a in availableAlgos:
            print("%s" % a)
        sys.exit(-1)

    test = 0
    if args.postevaluate:
        if args.displayneurons:
            test = 2
        else:
            test = 1
    
    if args.workers > 1:            # create multiple threads
        from fork import FORK
        forc = FORK()
        typ, comm, rank = forc.mpi_fork(args.workers)
        if typ == "parent": 
            sys.exit(-1)
        print("process", rank, "out of total ", comm.Get_size(), "started")

    
    print("Experiment: Environment %s Algo %s nreplications %d " % (environment, algoname, args.nreplications))
    
    render_mode = None
    if args.render or test > 0:
        render_mode = "human"
        
    if "Er" in environment:                   # Er environment (implemented in C++ and wrapped with Cython)
        ErProblem = __import__(environment)
        env = ErProblem.PyErProblem()   
        from policy import ErPolicy
        policy = ErPolicy(env, args.fileini, args.seed, test)
    elif "Bullet" in environment:             # Pybullet environment 
        import gymnasium as gym
        from gymnasium import spaces
        import pybullet
        import pybullet_envs
        env = gym.make(environment, render_mode=render_mode, options=optdict)
        from policy import BulletPolicy
        policy = BulletPolicy(env, args.fileini, args.seed, test)
    elif "Custom" in environment:              # Custom environment
        customEnv = __import__(environment)
        env = customEnv.customEnv(render_mode=render_mode, options=optdict)   
        from policy import GymPolicy
        policy = GymPolicy(env, args.fileini, args.seed, test)      
    else:                                       # OpenAi Gym environment
        import gymnasium as gym
        from gymnasium import spaces
        add_env = False
        no_net_env = False
        try:
            try:
                env = gym.make(environment, render_mode=render_mode, options=optdict)
            except:
                print(f"Environment {environment} might not accept <options> dict as parameter")
                env = gym.make(environment, render_mode=render_mode)
        except:
            print(f"Environment {environment} is not registered in gymnasium... Look for it in evorobotpy_envs!")
            try:
                envname = os.path.join("evorobotpy_envs", environment)
                try:
                    env = gym.make(envname, render_mode=render_mode, options=optdict)
                except:
                    print(f"Environment {envname} might not accept <options> dict as parameter")
                    env = gym.make(envname, render_mode=render_mode)
                add_env = True
                no_net_env = False
                try:
                    no_net_env = env.noNetEnv()
                except:
                    pass
            except:
                print(f"Environment {environment} (passed to gym.make() with argument {envname}) not found!!!")
                sys.exit()
        if no_net_env:
            from policy import PolicyNoNet
            policy = PolicyNoNet(env, args.fileini, args.seed, test)
        elif (isinstance(env.action_space, gym.spaces.box.Box) or add_env):
            from policy import GymPolicy
            policy = GymPolicy(env, args.fileini, args.seed, test)      # with continuous action space
        else:
            from policy import GymPolicyDiscr
            policy = GymPolicyDiscr(env, args.fileini, args.seed, test) # with discrete action space

    policy.environment = environment
    
    # Create the algorithm class
    if (algoname =='OpenAI-ES'):
        if args.workers > 1:
            from openaiesp import Algo
        else:
            from openaies import Algo
    elif (algoname == 'SSS'):
        from sss import Algo
    elif (algoname == 'SSS2'):
        from sss2 import Algo
    elif (algoname == 'CMA-ES'):
        from cmaes import Algo
    elif (algoname == 'xNES'):
        from xnes import Algo
    elif (algoname == 'sNES'):
        from snes import Algo
    elif (algoname == 'GA'):
        from ga import Algo
    elif (algoname == 'generational'):
        from generational import Algo
    elif (algoname == 'Hill-Climber'):
        from hillclimber import Algo
    elif (algoname == 'evostick'):
        from evostick import Algo
    elif (algoname == 'coevo' or algoname == 'generalist'):
        from coevo import Algo
    elif (algoname == 'coevo2'):
        from coevo2 import Algo
    elif (algoname =='archivestar1'):
        from archivestar1 import Algo
    elif (algoname == 'coevoarch'):
        from coevoarch import Algo
    elif (algoname == 'coevosinglepop'):
        from coevosinglepop import Algo
    elif (algoname =='archivestar1singlepop'):
        from archivestar1singlepop import Algo
    elif (algoname == 'coevoarchsinglepop'):
        from coevoarchsinglepop import Algo
        
    algo = Algo(env, policy, args.seed, args.fileini, args.folder)


    if args.workers > 1:
        algo.setProcess(args.workers, comm, rank)   # Initialize the variables

    if (test > 0):
        # test a policy
        print("Run Test: Environment %s " % (environment))
        algo.test(args.genofile)
    else:
        # run evolution
        if (args.seed != 0):
            print("Run Evolve: Environment %s Seed %d Nreplications %d" % (environment, args.seed, args.nreplications))
            for r in range(args.nreplications):
                algo.run()
                algo.seed += 1
                policy.seed += 1
                algo.reset()
                policy.reset()
                if args.workers > 1:
                    algo.setProcess(args.workers, comm, rank)
        else:
            print("\033[1mPlease indicate the seed to run evolution\033[0m")

if __name__ == "__main__":
    main(sys.argv)

