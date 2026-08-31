#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   
   It implements the JADE algorithm proposed in:

   Zhang, J., & Sanderson, A. C. (2009). JADE: adaptive differential evolution with optional external archive. 
   IEEE Transactions on evolutionary computation, 13(5), 945-958.
"""

import numpy as np
from numpy import zeros, dot, sqrt
import random
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort
import os
import sys
import configparser
import copy

# Adaptive Differential Evolution (JADE)
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popsize = 20 # Population size
            self.c = 0.1 # Taken from original paper
            self.p = 0.05 # Taken from original paper
            self.archsize = 0 # No archive is default
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
                if o == "c":
                    self.c = config.getfloat("ALGO","c")
                    found = 1
                if o == "p":
                    self.p = config.getfloat("ALGO","p")
                    found = 1
                if o == "archsize":
                    self.archsize = config.getint("ALGO","archsize")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("popsize [int]             : popsize (20)")
                    print("c [float]                 : factor controlling modifications for scaling factor and crossover rate")
                    print("p [float]                 : percentage of top solutions among which extracting for mutations")
                    print("archsize [int]            : size of the archive of outperformed solutions (default 0)")
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
        
    def crossover(self, mutated, target, cr, irand, rg, nparams):
        # generate a uniform random value for every dimension
        p = rg.uniform(0.0, 1.0, nparams)
        # generate trial vector by binomial crossover
        trial = [mutated[i] if (i == irand or p[i] < cr) else target[i] for i in range(nparams)]
        return trial
        
    def mean_l(self, vals):
        nvals = len(vals)
        num = 0.0
        den = 0.0
        for i in range(nvals):
            num += vals[i]**2
            den += vals[i]
        return num / den

    def run(self):

        self.loadhyperparameters()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        rg = np.random.RandomState(self.seed)      # create a random generator and initialize the seed
        pop = rg.randn(self.popsize, nparams)      # population size
        scaling_factors = zeros(self.popsize)
        crossover_rates = zeros(self.popsize)
        topp = int(self.p * self.popsize)
        mu_cr = 0.5
        st_cr = 0.1
        mu_f = 0.5
        st_f = 0.1
        archive = [] # Empty archive
        fitness = zeros(self.popsize)
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        # initialze the population
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            pop[i] = self.policy.get_trainable_flat()       

        print("Adaptive Differential Evolution (JADE): seed %d maxmsteps %d popSize %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.popsize, nparams))

        # main loop
        elapsed = 0
        while (ceval < self.maxsteps):
            
            cgen += 1
            
            # Initialize the lists of successful scaling factors and crossover rates
            successful_f = []
            successful_cr = []

            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            #self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Evaluate the population
            bfit = -9999999.0
            bid = -1
            for i in range(self.popsize):
                # Initialize scaling factors and crossover rates
                # Crossover rate
                crossover_rates[i] = rg.normal(loc=mu_cr, scale=st_cr)
                # Bounding in [0,1]
                if crossover_rates[i] < 0.0:
                    crossover_rates[i] = 0.0
                if crossover_rates[i] > 1.0:
                    crossover_rates[i] = 1.0
                # Scaling factor
                ok = False
                while not ok:
                    ok = True
                    scaling_factors[i] = mu_f + st_f * rg.standard_cauchy()
                    # Scaling factor is truncated to 1 if F >= 1, or regenerated if F <= 0
                    if scaling_factors[i] >= 1.0:
                        scaling_factors[i] = 1.0
                    if scaling_factors[i] <= 0.0:
                        ok = False
                                        
                # Evaluate individual
                self.policy.set_trainable_flat(pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], pop[i])           # Update data if the current offspring is better than current best
                
            sfitness, sindex = descendent_sort(fitness)       # sort the fitness
            
            for i in range(self.popsize):
                # Choose candidate among the 100p% best solutions
                ri = rg.randint(0, topp)
                idx = sindex[ri]
                cbest = pop[idx]
                # Choose random index from the population
                r1 = rg.randint(0, self.popsize)
                sol1 = pop[r1]
                # Choose another random index from P U A (union of population and archive). If no archive, random from population
                if self.archsize == 0:
                    ok = False
                    while not ok:
                        r2 = rg.randint(0, self.popsize)
                        if r2 != r1:
                            ok = True
                    sol2 = pop[r2]
                else:
                    archlen = len(archive)
                    ok = False
                    while not ok:
                        r2 = rg.randint(0, self.popsize + archlen)
                        if r2 != r1:
                            ok = True
                    if r2 < self.popsize:
                        # r2 comes from the population
                        sol2 = pop[r2]
                    else:
                        # r2 comes from the archive
                        sol2 = archive[r2 - self.popsize]
                        
                # Perform mutation
                mutated = pop[i] + scaling_factors[i] * (cbest - pop[i]) + scaling_factors[i] * (sol1 - sol2)

                # Perform crossover
                prand = rg.randint(0, nparams) # Random index to allow at least one selection from mutated
                offspring = self.crossover(mutated, pop[i], crossover_rates[i], prand, rg, nparams)

                # Evaluate offspring
                self.policy.set_trainable_flat(offspring)        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                ofit = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(ofit, offspring)           # Update data if the current offspring is better than current best
                
                if ofit > fitness[i]:
                    if self.archsize > 0:
                        # Individual moved to archive
                        archive.append(pop[i])
                    # Append crossover rate to list of successful crossovers
                    successful_cr.append(crossover_rates[i])
                    # Append scaling factors to list of successful scaling factors
                    successful_f.append(scaling_factors[i])
                    # Replace individual with the offspring
                    pop[i] = copy.deepcopy(offspring)
                    # Update its fitness
                    fitness[i] = ofit
                
                # Update current best
                if fitness[i] > bfit:
                    bfit = fitness[i]
                    bid = i
                    
            # Check archivesize (if enabled)
            if self.archsize > 0:
                archlen = len(archive)
                if archlen > self.popsize:
                    difflen = archlen - self.popsize
                    indices = list(np.arange(archlen))
                    removed = random.sample(indices, difflen)
                    # Delete indices
                    for elem in sorted(removed, reverse=True):
                        archive.pop(elem)
                    assert len(archive) == self.archsize, "Unexpected error!"
                        
            # Update mu_cr and mu_f
            mu_cr = (1.0 - self.c) * mu_cr + self.c * np.mean(successful_cr)
            mu_f = (1.0 - self.c) * mu_f + self.c * self.mean_l(successful_f)
                
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
