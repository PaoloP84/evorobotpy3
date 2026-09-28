#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/paolop84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
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
import random

# Evolve with (mu, lambda)-ES or (mu + lambda)-ES with adaptive step size
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.mu = 1
            self.lam = 1
            self.selection = 0 # 0: (mu,lambda)-ES; 1: (mu+lambda)-ES
            self.mutation = 0.02
            self.adapt = 0 # Works only for (mu+lambda)-ES
            self.saveeach = 1
            # Parse section [ALGO]
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "mu":
                    self.mu = config.getint("ALGO","mu")
                    found = 1
                if o == "lam":
                    self.lam = config.getint("ALGO","lam")
                    found = 1
                if o == "selection":
                    self.selection = config.getint("ALGO","selection")
                    found = 1
                if o == "mutation":
                    self.mutation = config.getfloat("ALGO","mutation")
                    found = 1
                if o == "adapt":
                    self.adapt = config.getint("ALGO","adapt")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
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
        
    def savePop(self):
        fname = os.path.join(self.filedir, "popS" + str(self.seed))
        np.save(fname, self.pop)
        
    def loadPop(self):
        self.pop = np.load(os.path.join(self.filedir, "popS" + str(self.seed) + ".npy"), allow_pickle=True)

    def initPop(self):
        for i in range(self.mu):
            self.policy.nn.initWeights()
            self.pop[i] = np.copy(self.policy.get_trainable_flat())

    def run(self):

        self.readConfig()                 # initialize hyperparameters
        
        # Check values of mu and lambda (it should be lambda >= mu)
        assert self.lam >= self.mu, "Inconsistent values"
        
        start_time = time.time()                   # start time
        self.nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        self.rg = np.random.RandomState(self.seed) # create a random generator and initialize the seed
        self.pop = self.rg.randn(self.mu, self.nparams) # population
        self.fitness = zeros(self.mu + self.lam)   # fitness
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations
        self.off = zeros((self.lam, self.nparams))
        
        print("ES: seed %d maxsteps %d mu %d lambda %d mutRate %.3f selection %d" % (self.seed, self.maxsteps, self.mu, self.lam, self.mutation, self.selection))
        
        # Initialize population
        self.initPop()
        
        # Initialize mutation rate (for adaptive (mu+lambda)-ES)
        initMut = self.mutation

        # main loop
        cgen = 0
        ceval = 0
        while ceval < self.maxsteps:
            # Update generation counter
            cgen += 1
            
            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Evaluate the population
            for i in range(self.mu):
                self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                self.fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(self.fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
                
                # Generate and evaluate lambda offspring
                
                for j in range(self.lam):
                    self.off[j] = self.pop[i] + self.mutation * self.rg.randn(self.nparams)
                    self.policy.set_trainable_flat(self.off[j])
                    eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                    self.fitness[self.mu + j] = eval_rews                        # store fitness
                    ceval += eval_length                          # Update the number of evaluations
                    self.updateBest(self.fitness[self.mu + j], self.off[j]) # Update data if the current offspring is better than current best
                       
            # Selection
            # Sort fitness
            sfitness, sindex = descendent_sort(self.fitness) # create an index with the ID of the individuals sorted for fitness 
            # Retrieve the best
            bfit = sfitness[0]
            if sindex[0] < self.mu:
                bind = self.pop[sindex[0]]
            else:
                bind = self.off[sindex[0] - self.mu]
            # Now check the type of selection
            if self.selection == 0:
                # (mu,lambda)-ES: retrieve the best offspring
                next_pop = []
                popsize = 0
                i = 0
                while i < (self.mu + self.lam) and popsize < self.mu:
                    idx = sindex[i]
                    if idx >= self.mu:
                        next_pop.append(self.off[idx - self.mu])
                        self.fitness[popsize] = self.fitness[idx]
                        popsize += 1
                    i += 1
                assert popsize == self.mu, "Weird failure!!!"
                # Replace pop
                self.pop = next_pop
            else:
                # (mu+lambda)-ES: retrieve the best between parents and offspring
                next_pop = []
                next_fit = []
                popsize = 0
                i = 0
                while i < (self.mu + self.lam) and popsize < self.mu:
                    idx = sindex[i]
                    if idx >= self.mu:
                        next_pop.append(self.off[idx - self.mu])
                        # Check if adapt is set to 1
                        if self.adapt == 1:
                            self.mutation *= 1.1
                            if self.mutation >= 1.0:
                                self.mutation = 1.0
                    else:
                        next_pop.append(self.pop[idx])
                        if self.adapt == 1:
                            self.mutation *= 0.9
                            if self.mutation <= 0.0:
                                self.mutation = 0.0
                    next_fit.append(self.fitness[idx])
                    popsize += 1
                    i += 1
                assert popsize == self.mu, "Weird failure!!!"
                # Replace pop
                self.pop = next_pop
                # And fitness
                for i in range(self.mu):
                    self.fitness[i] = next_fit[i]
            
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(bind)     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, bind)            # eventually update the genotype/fitness of the best post-evaluated individual   

            # Compute average fitness
            avgf = np.average(self.fitness)

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avgf))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, avgf])

            if (cgen % self.saveeach) == 0:
                self.savedata(ceval, cgen, bfit, bgfit, avgf) 

        self.savedata(ceval, cgen, bfit, bgfit, avgf)
        self.savePop()
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

