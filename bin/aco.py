#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/paolop84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py

   It implements a continous version of the Ant Colony Optimization (ACO) algorithm introduced in:

   Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains. European journal of operational research, 185(3), 1155-1173.
"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort
import os
import sys
import configparser
import random

# Evolve with Ant Colony Optimization (ACO) algorithm for continuous problems (ACOR)
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popSize = 50
            self.nants = 10
            self.q = 0.1
            self.xi = 0.85
            self.saveeach = 1
            # Parse section [ALGO]
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "popsize":
                    self.popSize = config.getint("ALGO","popsize")
                    found = 1
                if o == "nants":
                    self.nants = config.getint("ALGO","nants")
                    found = 1
                if o == "q":
                    self.g = config.getfloat("ALGO","q")
                    found = 1
                if o == "xi":
                    self.xi = config.getfloat("ALGO","xi")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("Option %s in section [ALGO] of %s file is unknown" % (o, self.fileini))
                    sys.exit()
        else:
            print("ERROR: configuration file %s does not exist" % (self.fileini))
        
    def savedata(self, ceval, cgen, bfit, bgfit, avgfit):
        self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")  
        fp = open(fname, "w")  # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f\n' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avgfit))
        fp.close()
        
    def savePop(self):
        fname = os.path.join(self.filedir, "popS" + str(self.seed))
        np.save(fname, self.pop)
        
    def loadPop(self):
        self.pop = np.load(os.path.join(self.filedir, "popS" + str(self.seed) + ".npy"), allow_pickle=True)

    def initPop(self):
        for i in range(self.popSize):
            self.policy.nn.initWeights()
            self.pop[i] = np.copy(self.policy.get_trainable_flat())
        
    def run(self):

        self.readConfig()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        self.nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        self.rg = np.random.RandomState(self.seed) # create a random generator and initialize the seed
        self.pop = self.rg.randn(self.popSize, self.nparams) # population
        fitness = zeros(self.popSize)          # fitness
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        print("ACO: seed %d maxsteps %d popsize %d nants %d q %lf xi %lf" % (self.seed, self.maxsteps, self.popSize, self.nants, self.q, self.xi))
        
        # Calculate selection weights for archive solutions (higher weight for better rank in maximization)
        i = np.arange(1, self.popSize + 1)
        self.weights = 1.0 / (self.q * self.popSize * np.sqrt(2 * np.pi)) * np.exp(-((i - 1) ** 2) / (2 * (self.q * self.popSize) ** 2))
        self.weights /= np.sum(self.weights)  # Normalize weights to sum to 1

        # If normalize=1 we update the normalization vectors
        if (self.policy.normalize == 1):
            self.policy.nn.updateNormalizationVectors()
            
        self.policy.nn.seed(self.policy.get_seed)  # set the policy seed
            
        # Evaluate the population
        for i in range(self.popSize):
            self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
            fitness[i] = eval_rews                        # store fitness
            ceval += eval_length                          # Update the number of evaluations
            self.updateBest(fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
                
        # Sort population
        sort_idx = np.argsort(fitness)[::-1]
        self.pop = self.pop[sort_idx]
        fitness = fitness[sort_idx]
        
        bind = self.pop[0].copy()
        bfit = fitness[0]
        # main loop
        cgen = 0
        ceval = 0
        while ceval < self.maxsteps:
            # Update generation counter
            cgen += 1
            
            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            #self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Generate and evaluate new candidate ants
            new_ants = []
            new_ants_fit = []
            for _ in range(self.nants):
                selected_idx = self.rg.choice(self.popSize, p=self.weights)
                chosen_solution = self.pop[selected_idx]
                
                # Create the new ant
                ant = np.zeros(self.nparams)
                for j in range(self.nparams):
                    mean_val = chosen_solution[j]
                    sigma = self.xi * np.sum(np.abs(self.pop[:, j] - mean_val)) / (self.popSize - 1)
                    if sigma == 0.0:
                        sigma = 1e-5
                    ant[j] = self.rg.normal(mean_val, sigma)
                
                # Evaluate the ant
                self.policy.set_trainable_flat(ant)        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                cfit = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(cfit, ant)           # Update data if the current offspring is better than current best
                
                new_ants.append(ant)
                new_ants_fit.append(cfit)
            
            # Evaluate the population
            for i in range(self.popSize):
                self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
    
            # Combine and select top solutions
            sols = np.vstack((self.pop, np.array(new_ants)))
            fits = np.concatenate((fitness, np.array(new_ants_fit)))
            
            sort_idx = np.argsort(fits)[::-1]
            self.pop = sols[sort_idx][:self.popSize]
            fitness = fits[sort_idx][:self.popSize]
            
            # Update global best tracked across iterations
            if fitness[0] > bfit:
                bfit = fitness[0]
                bind = self.pop[0].copy()            
                
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(bind)     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, bind)            # eventually update the genotype/fitness of the best post-evaluated individual

            # Compute average fitness
            avgf = np.average(fitness)

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, np.average(fitness)))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, avgf])

            if (cgen % self.saveeach) == 0:
                self.savedata(ceval, cgen, bfit, bgfit, avgf) 

        self.savedata(ceval, cgen, bfit, bgfit, avgf)
        self.savePop()
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

