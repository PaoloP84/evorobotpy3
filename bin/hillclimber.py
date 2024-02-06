#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py 
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


# Evolve with Hill-climber
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popSize = 20
            self.mutation = 0.02
            self.saveEveryN = 1
            # Parse section [ALGO]
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "mutation":
                    self.mutation = config.getfloat("ALGO","mutation")
                    found = 1
                if o == "popsize":
                    self.popSize = config.getint("ALGO","popsize")
                    found = 1
                if o == "saveeveryn":
                    self.saveEveryN = config.getint("ALGO","saveeveryn")
                    found = 1

                if found == 0:
                    print("Option %s in section [ALGO] of %s file is unknown" % (o, self.fileini))
                    sys.exit()
        else:
            print("ERROR: configuration file %s does not exist" % (self.fileini))

    def savedata(self, ceval, cgen, bfit, bgfit, avefit):
        self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")  
        fp = open(fname, "w")  # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f\n' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avefit))
        fp.close()

    def initPop(self):
        for i in range(self.popSize):
            self.pop[i] = self.policy.get_trainable_flat()

    def mutate(self, src):
        assert src < self.popSize, "Invalid index %d for mutate()".format(src)
        parent = self.pop[src]
        child = np.zeros(self.nparams, dtype=np.float64)
        for g in range(self.nparams):
            child[g] = parent[g]
            coin = self.rg.uniform(0.0, 1.0)
            if coin <= self.mutation:
                child[g] += self.rg.randn()
        return child

    def run(self):

        self.readConfig()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        self.nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        self.rg = np.random.RandomState(self.seed) # create a random generator and initialize the seed
        self.pop = self.rg.randn(self.popSize, self.nparams) # population
        fitness = zeros(self.popSize * 2)          # fitness
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        print("Hill-Climber: seed %d maxsteps %d popSize %d mutRate %lf" % (self.seed, self.maxsteps, self.popSize, self.mutation))
        
        # Initialize population
        self.initPop()

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
            
            # Evaluate the population
            for i in range(self.popSize):
                # Evaluate parent
                self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
                
                # Create child and evaluate it
                child = self.mutate(i)
                self.policy.set_trainable_flat(child)
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[self.popSize + i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[self.popSize + i], child)           # Update data if the current offspring is better than current best
                # Check whether child is better than parent. If this is the case, replace
                if fitness[self.popSize + i] > fitness[i]:
                    self.pop[i] = child
                    fitness[i] = fitness[self.popSize + i]

            popfit, index = descendent_sort(fitness[0:self.popSize])         # create an index with the ID of the individuals sorted for fitness
            bfit = popfit[0]
            
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(self.pop[index[0]])     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, self.pop[index[0]])            # eventually update the genotype/fitness of the best post-evaluated individual

            avgf = np.average(popfit)

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, np.average(fitness)))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, avgf])

            if (cgen % self.saveEveryN) == 0:
                self.savedata(ceval, cgen, bfit, bgfit, avgf) 

        self.savedata(ceval, cgen, bfit, bgfit, avgf)
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))
