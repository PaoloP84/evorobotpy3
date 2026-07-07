#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
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
import copy

# Differential Evolution (DE)
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popsize = 20 # Population size
            self.scalefactor = 0.0 # Scale factor for mutations (should be bounded between 0 and 2)
            self.crossrate = 0.0 # Crossover rate (should be bounded between 0 and 1)
            self.saveeach = 60
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "popsize":
                    self.popsize = config.getint("ALGO","popsize")
                    found = 1
                if o == "scalefactor":
                    self.scalefactor = config.getfloat("ALGO","scalefactor")
                    found = 1
                if o == "crossrate":
                    self.crossrate = config.getfloat("ALGO","crossrate")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("popsize [int]             : popsize (20)")
                    print("scalefactor [float]       : scale factor for mutation (bounded in [0,2], default 0)")
                    print("crossrate [float]         : crossover rate (bounded in [0,1], default 0)")
                    print("saveeach [integer]        : save file every N minutes (default 60)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))

    def savedata(self, ceval, cgen, bfit, bgfit, avefit, aveweights):
        self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")  
        fp = open(fname, "w")  # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f \n' % (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avefit, aveweights))
        fp.close()
            
    def mutate(self, x):
        return x[0] + self.scalefactor * (x[1] - x[2])
        
    def crossover(self, mutated, target, rg, nparams):
        # generate a uniform random value for every dimension
        p = rg.uniform(0.0, 1.0, nparams)
        # generate trial vector by binomial crossover
        trial = [mutated[i] if p[i] < self.crossrate else target[i] for i in range(nparams)]
        return trial

    def run(self):

        self.loadhyperparameters()                 # initialize hyperparameters
        assert self.scalefactor >= 0.0 and self.scalefactor <= 2.0, "scalefactor should be bounded in [0,2]!"
        assert self.crossrate >= 0.0 and self.crossrate <= 1.0, "crossrate should be bounded in [0,1]!"
        start_time = time.time()                   # start time
        nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        rg = np.random.RandomState(self.seed)      # create a random generator and initialize the seed
        pop = rg.randn(self.popsize, nparams)      # particles
        fitness = zeros(self.popsize)
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        # initialze the population
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            pop[i] = self.policy.get_trainable_flat()       

        print("Differential Evolution (DE): seed %d maxmsteps %d popSize %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.popsize, nparams))

        # main loop
        elapsed = 0
        while (ceval < self.maxsteps):
            
            cgen += 1

            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            #self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Evaluate the population
            bfit = -9999999.0
            bid = -1
            for i in range(self.popsize):
                # Choose 3 candidates other than i
                candidates = [candidate for candidate in range(self.popsize) if candidate != i]
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
                
                # perform mutation
                mutated = self.mutate([a, b, c])
                # perform crossover
                offspring = self.crossover(mutated, pop[i], rg, nparams)
                                 
                # Evaluate individual
                self.policy.set_trainable_flat(pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], pop[i])           # Update data if the current offspring is better than current best
                
                # Evaluate offspring
                self.policy.set_trainable_flat(offspring)        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                ofit = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(ofit, offspring)           # Update data if the current offspring is better than current best
                
                if ofit > fitness[i]:
                    # Replace individual with the offspring
                    pop[i] = copy.deepcopy(offspring)
                    # Update its fitness
                    fitness[i] = ofit
                
                # Update current best
                if fitness[i] > bfit:
                    bfit = fitness[i]
                    bid = i
                
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(pop[bid])     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, pop[bid])            # eventually update the genotype/fitness of the best post-evaluated individual
            
            #print(bgfit)
            
            avgfit = np.average(fitness[0:self.popsize])

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avgfit, np.average(np.absolute(pop[bid]))))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, avgfit])

            # save data
            if ((time.time() - self.last_save_time) > (self.saveeach * 60)):
                self.savedata(ceval, cgen, bfit, bgfit, avgfit, np.average(np.absolute(pop[bid])))
                self.last_save_time = time.time()  

        self.savedata(ceval, cgen, bfit, bgfit, avgfit, np.average(np.absolute(pop[bid])))
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))#
