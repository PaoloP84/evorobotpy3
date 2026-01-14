#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   It includes an implementation of the SSSHC algorithm described in the following paper:
   Pagliuca P. (2024). Learning and evolution: factors influencing an effective combination. AI, vol. 5, issue 4, pp. 2393-2432.
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

# Evolve with SSSHC
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popsize = 20
            self.mutation = 0.02
            self.crossover = 0.0
            self.iters = 1
            self.scale = 1.0
            self.saveeach = 60
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
                    self.popsize = config.getint("ALGO","popsize")
                    found = 1
                if o == "crossover":
                    self.crossover = config.getfloat("ALGO","crossover")
                    found = 1
                if o == "iters":
                    self.iters = config.getint("ALGO","iters")
                    found = 1
                if o == "scale":
                    self.scale = config.getfloat("ALGO","scale")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("popsize [int]             : popsize (20)")
                    print("mutation [float]          : mutation (default 0.02)")
                    print("saveeach [integer]        : save file every N minutes (default 60)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))

    def savedata(self, ceval, cgen, bfit, bgfit, avefit, aveweights):
            self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
            fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")  
            fp = open(fname, "w")  # save summary
            fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f \n' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avefit, aveweights))
            fp.close()
            
    def mutate(self, ind, size, rg):
        mut = np.zeros(size, dtype=np.float64)
        # Perform crossover (if any)
        if self.crossover > 0.0:
            coin = rg.uniform(0.0,1.0)
            if coin < self.crossover:
                # Extract random point
                point = rg.randint(0, size)
                i = 0
                while i < point:
                    mut[i] = ind[size - point + i]
                    i += 1
                i = point
                while i < size:
                    mut[i] = ind[i - point]
                    i += 1
        # Perform mutations
        for i in range(size):
            coin = rg.uniform(0.0,1.0)
            if coin < self.mutation:
                mut[i] = self.scale * rg.uniform(-self.policy.wrange, self.policy.wrange)#rg.randn()
            else:
                mut[i] = ind[i]
        return mut
        
    def mutate_neutral(self, ind, size, rg):
        mut = np.zeros(size, dtype=np.float64)
        # Extract gene to mutate
        gene = rg.randint(0, size)
        # Perform mutations
        for i in range(size):
            if i == gene:
                mut[i] = self.scale * rg.uniform(-self.policy.wrange, self.policy.wrange)#rg.randn()
            else:
                mut[i] = ind[i]
        return mut

    def run(self):

        self.loadhyperparameters()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        rg = np.random.RandomState(self.seed)      # create a random generator and initialize the seed
        pop = self.scale * rg.randn(self.popsize, nparams)      # population
        offspring = self.scale * rg.randn(self.popsize, nparams)# offspring
        fitness = zeros(self.popsize * 2)
        pfitness = zeros(self.popsize)             # fitness
        cfitness = zeros(self.popsize)
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        assert ((self.popsize % 2) == 0), print("the size of the population should be odd")

        # initialze the population
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            pop[i] = self.policy.get_trainable_flat()       

        print("SSSHC: seed %d maxmsteps %d popSize %d noiseStdDev %lf nparams %d" % (self.seed, self.maxsteps / 1000000, self.popsize, self.mutation, nparams))

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
            for i in range(self.popsize):                           
                self.policy.set_trainable_flat(pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                fitness[i] = pfitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], pop[i])           # Update data if the current offspring is better than current best
                
                # Generate offspring
                offspring[i] = self.mutate(pop[i], nparams, rg) 
                
                self.policy.set_trainable_flat(offspring[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                fitness[self.popsize + i] = cfitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[self.popsize + i], offspring[i])           # Update data if the current offspring is better than current best
                
            #print(fitness[0:self.popsize])
            #print(fitness[self.popsize:])

            pfitness, pindex = descendent_sort(pfitness)         # create an index with the ID of the individuals sorted for fitness
            cfitness, cindex = descendent_sort(cfitness)
            
            #print(pfitness)
            #print(cfitness)
            
            #print(pindex)
            #print(cindex)
            
            #print(fitness[0:self.popsize])
            #print(fitness[self.popsize:])
            
            # Select best popsize individuals as new pop
            pId = 0
            cId = 0
            bfit = None
            bid = -1
            for i in range(self.popsize):
                if pfitness[pId] > cfitness[cId]:
                    if i == 0:
                        bfit = pfitness[pId]
                        bid = pindex[pId]
                    pId += 1
                else:
                    #print("Replacing %d with %d" % (pindex[self.popsize - 1 - cId], cindex[cId]))
                    #print("Fitness: %lf vs %lf" % (pfitness[self.popsize - 1 - cId], cfitness[cId]))
                    #print("Fitness: %lf vs %lf" % (fitness[pindex[self.popsize - 1 - cId]], fitness[self.popsize + cindex[cId]]))
                    pop[pindex[self.popsize - 1 - cId]] = copy.deepcopy(offspring[cindex[cId]])
                    fitness[pindex[self.popsize - 1 - cId]] = fitness[self.popsize + cindex[cId]]
                    #print("New fitness: %lf" % fitness[pindex[self.popsize - 1 - cId]])
                    if i == 0:
                        bfit = cfitness[cId]
                        bid = pindex[self.popsize - 1 - cId]
                    cId += 1
                    
            #print(fitness[0:self.popsize])
                    
            #print(bfit, bid)
                    
            # Refinement
            #candidate = np.arange(nparams, dtype=np.float32)
            bid = -1
            bfit = -999999999.0
            for i in range(self.popsize):
                for it in range(self.iters):
                    candidate = self.mutate_neutral(pop[i], nparams, rg)
                    self.policy.set_trainable_flat(candidate)
                    eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                    self.updateBest(eval_rews, candidate)              # eventually update the genotype/fitness of the best individual so far
                    if eval_rews > fitness[i]:
                        fitness[i] = eval_rews
                        pop[i] = copy.deepcopy(candidate)
                    ceval += eval_length
                if fitness[i] > bfit:
                    bfit = fitness[i]
                    bid = i
                    
            #print(bfit, bid)
            
            #print(fitness[0:self.popsize])
            
            #bfit = fitness[index[0]]
            #self.updateBest(bfit, pop[bid])              # eventually update the genotype/fitness of the best individual so far

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
