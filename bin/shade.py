#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/paolop84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py 

   It implements the Success-History based Adaptive Differential Evolution (SHADE) algorithm introduced in:

   Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation for differential evolution. In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
"""

import numpy as np
from numpy import zeros, ones, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort
import os
import sys
import configparser
import random

# Evolve with Success-History based Adaptive Differential Evolution (SHADE) algorithm
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popSize = 20
            self.memSize = 5
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
                if o == "memsize":
                    self.memSize = config.getint("ALGO","memsize")
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
        archive = [] # Empty archive
        # Initialize historical memories for F and CR (default to 0.5)
        memory_F = ones(self.memSize) * 0.5
        memory_CR = ones(self.memSize) * 0.5
        memory_pos = 0
        fitness = zeros(self.popSize)          # fitness
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations
        
        # initialze the population
        for i in range(self.popSize):
            self.policy.nn.initWeights()
            self.pop[i] = self.policy.get_trainable_flat()

        print("SHADE: seed %d maxsteps %d popsize %d memsize %d" % (self.seed, self.maxsteps, self.popSize, self.memSize))
        
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
    
            # Get current best individual
            bid = np.argmax(fitness)
            if fitness[bid] > bfit:
                bfit = fitness[bid]
                bind = self.pop[bid].copy()
                   
            p_rate = 0.11  # Top p-percent for pbest mutation strategy
            p_num = max(2, int(np.ceil(p_rate * self.popSize)))
            
            successful_F = []
            successful_CR = []
            fitness_improvements = []
            
            new_pop = np.zeros_like(self.pop)
            new_fitness = np.zeros_like(fitness)
            
            # Sort population descending for maximization
            sorted_indices = np.argsort(fitness)[::-1]
            sorted_pop = self.pop[sorted_indices]
            
            for i in range(self.popSize):
                # Select a random memory index
                r_idx = self.rg.randint(0, self.memSize)
                
                # Generate scaling factor F using Cauchy distribution
                F = memory_F[r_idx] + 0.1 * self.rg.standard_cauchy()
                while F <= 0:
                    F = memory_F[r_idx] + 0.1 * self.rg.standard_cauchy()
                # F cannot be > 1
                F = min(F, 1.0)
                
                # Generate crossover rate CR using normal distribution
                CR = self.rg.normal(loc=memory_CR[r_idx], scale=0.1)
                # CR is bounded in the range [0,1]
                CR = np.clip(CR, 0.0, 1.0)
                
                # Mutation (DE/current-to-pbest/1 with archive)
                p_idx = self.rg.randint(0, p_num)
                x_pbest = sorted_pop[p_idx]
                
                r1 = self.rg.randint(0, self.popSize)
                while r1 == i:
                    r1 = self.rg.randint(0, self.popSize)
                x_r1 = self.pop[r1]
                
                pop_archive = np.vstack((self.pop, np.array(archive))) if len(archive) > 0 else self.pop
                r2 = self.rg.randint(0, len(pop_archive))
                x_r2 = pop_archive[r2]
                
                v = self.pop[i] + F * (x_pbest - self.pop[i]) + F * (x_r1 - x_r2)
                
                # Binomial Crossover
                j_rand = self.rg.randint(0, self.nparams)
                u = np.array([v[j] if (self.rg.rand() < CR or j == j_rand) else self.pop[i, j] for j in range(self.nparams)])
                
                # Evaluate trial vector
                self.policy.set_trainable_flat(u)        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                u_fit = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(u_fit, u)           # Update data if the current offspring is better than current best
                
                # Selection
                if u_fit > fitness[i]:
                    new_pop[i] = u
                    new_fitness[i] = u_fit
                    archive.append(self.pop[i].copy())
                    
                    improvement = u_fit - fitness[i]
                    successful_F.append(F)
                    successful_CR.append(CR)
                    fitness_improvements.append(improvement)
                else:
                    new_pop[i] = self.pop[i]
                    new_fitness[i] = fitness[i]
                    
            # Update population
            self.pop = new_pop
            fitness = new_fitness
            
            # Maintain archive size bound
            if len(archive) > self.popSize:
                idx_to_remove = np.random.choice(len(archive), len(archive) - self.popSize, replace=False)
                archive = [archive[idx] for idx in range(len(archive)) if idx not in idx_to_remove]
                
            # Update historical memory using weighted Lehmer mean for successful parameters
            if len(successful_F) > 0:
                weights = np.array(fitness_improvements) / np.sum(fitness_improvements)
                
                memory_F[memory_pos] = np.sum(weights * (np.array(successful_F) ** 2)) / np.sum(weights * np.array(successful_F))
                memory_CR[memory_pos] = np.sum(weights * np.array(successful_CR))
                
                memory_pos = (memory_pos + 1) % self.memSize
                
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

