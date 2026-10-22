#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/paolop84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py

   It implements the Artificial Bee Colony (ABC) algorithm introduced in:

   Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
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

# Evolve with Artificial Bee Colony (ABC) algorithm
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popSize = 20
            self.limit = 15
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
                if o == "limit":
                    self.limit = config.getint("ALGO","limit")
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
        trial_counters = np.zeros(self.popSize, dtype=int)
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        print("ABC: seed %d maxsteps %d popsize %d limit %d" % (self.seed, self.maxsteps, self.popSize, self.limit))
        
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
                
        # Store best idx
        idx = np.argmax(fitness)        
        bind = self.pop[idx].copy()
        bfit = fitness[idx]
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
            
            # Employed Bees Phase
            for i in range(self.popSize):
                # Pick a random dimension to modify and another distinct food source
                k = self.rg.choice([x for x in range(self.popSize) if x != i])
                j = self.rg.randint(0, self.nparams - 1)
                phi = self.rg.uniform(-1, 1)
                
                neighbor = self.pop[i].copy()
                neighbor[j] += phi * (self.pop[i][j] - self.pop[k][j])

                # Evaluate neighbor solution
                self.policy.set_trainable_flat(neighbor) # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                nfit = eval_rews                         # store fitness
                ceval += eval_length                     # Update the number of evaluations
                self.updateBest(nfit, neighbor)          # Update data if the current offspring is better than current best
                
                # Selection
                if nfit > fitness[i]:
                    self.pop[i] = neighbor
                    fitness[i] = nfit
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1
                    
            # Update global best
            idx = np.argmax(fitness)
            if fitness[idx] > bfit:
                bfit = fitness[idx]
                bind = self.pop[idx].copy()

            # Onlooker Bees Phase
            # Calculate selection probabilities
            max_fit = np.max(fitness)
            if max_fit == 0:
                probabilities = np.ones(self.popSize) / self.popSize
            else:
                probabilities = 0.9 * (fitness / max_fit) + 0.1
                
            i = 0
            t = 0
            while t < self.popSize:
                if self.rg.random() < probabilities[i]:
                    t += 1
                    k = self.rg.choice([x for x in range(self.popSize) if x != i])
                    j = self.rg.randint(0, self.nparams - 1)
                    phi = self.rg.uniform(-1, 1)
                    
                    neighbor = self.pop[i].copy()
                    neighbor[j] += phi * (self.pop[i][j] - self.pop[k][j])
                    
                    # Evaluate neighbor solution
                    self.policy.set_trainable_flat(neighbor) # set policy parameters
                    eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                    nfit = eval_rews                         # store fitness
                    ceval += eval_length                     # Update the number of evaluations
                    self.updateBest(nfit, neighbor)          # Update data if the current offspring is better than current best
                
                    # Selection
                    if nfit > fitness[i]:
                        self.pop[i] = neighbor
                        fitness[i] = nfit
                        trial_counters[i] = 0
                    else:
                        trial_counters[i] += 1
                i = (i + 1) % self.popSize

            # Update global best
            idx = np.argmax(fitness)
            if fitness[idx] > bfit:
                bfit = fitness[idx]
                bind = self.pop[idx].copy()

            # Scout Bees Phase
            abandoned_idx = np.argmax(trial_counters)
            if trial_counters[abandoned_idx] > self.limit:
                low = -self.policy.wrange
                high = self.policy.wrange
                self.pop[abandoned_idx] = np.random.uniform(low, high, size=self.nparams)
                self.policy.set_trainable_flat(self.pop[abandoned_idx]) # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[abandoned_idx] = eval_rews                         # store fitness
                ceval += eval_length                     # Update the number of evaluations
                self.updateBest(fitness[abandoned_idx], self.pop[abandoned_idx])          # Update data if the current offspring is better than current best
                if fitness[abandoned_idx] > bfit:
                    bfit = fitness[abandoned_idx]
                    bind = self.pop[abandoned_idx].copy()
                trial_counters[abandoned_idx] = 0      
                
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

