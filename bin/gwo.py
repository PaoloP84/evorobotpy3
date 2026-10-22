#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/paolop84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py 

   It implements the Grey Wolf Optimizer (GWO) algorithm introduced in:

   Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.
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

# Evolve with Grey Wolf Optimizer (GWO) algorithm
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popSize = 20
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

        print("GWO: seed %d maxsteps %d popsize %d" % (self.seed, self.maxsteps, self.popSize))
        
        alpha_pos = np.zeros(self.nparams)
        alpha_score = -float("inf")
        
        beta_pos = np.zeros(self.nparams)
        beta_score = -float("inf")
        
        delta_pos = np.zeros(self.nparams)
        delta_score = -float("inf")

        bfit = -99999999.0
        bind = None
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
                self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
                if fitness[i] > bfit:
                    bfit = fitness[i]
                    bind = self.pop[i].copy()
                
                # Update Alpha, Beta, Delta
                if fitness[i] > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fitness[i], self.pop[i].copy()
                elif fitness[i] > beta_score and fitness[i] < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fitness[i], self.pop[i].copy()
                elif fitness[i] > delta_score and fitness[i] < beta_score and fitness[i] < alpha_score:
                    delta_score, delta_pos = fitness[i], self.pop[i].copy()
                    
            # Parameter 'a' decreases linearly from 2 to 0
            a = 2 - ceval * (2 / self.maxsteps)
            
            # Update positions of all search agents
            for i in range(self.popSize):
                for j in range(self.nparams):
                    r1, r2 = self.rg.rand(), self.rg.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - self.pop[i][j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = self.rg.rand(), self.rg.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - self.pop[i][j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = self.rg.rand(), self.rg.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - self.pop[i][j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    self.pop[i][j] = (X1 + X2 + X3) / 3.0
                
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

