#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py 

   It implements a Memetic Algorithm (MA). For further references, readers can refer to:

   Moscato, P. (1989). On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms. Caltech concurrent computation program, C3P Report, 826 (1989), 37.
   Moscato, P., Cotta, C., & Mendes, A. (2004). Memetic algorithms. In New optimization techniques in engineering (pp. 53-85).
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

# Evolve with Memetic Algorithm (MA)
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
            self.tour_size = 2
            self.loc_freq = 0
            self.loc_top = 0.1
            self.loc_iters = 1
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
                if o == "tsize":
                    self.tour_size = config.getint("ALGO", "tsize")
                    found = 1
                if o == "freq":
                    self.loc_freq = config.getint("ALGO","freq")
                    found = 1
                if o == "top":
                    self.loc_top = config.getfloat("ALGO","top")
                    found = 1
                if o == "iters":
                    self.loc_iters = config.getint("ALGO","iters")
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

    def savedata(self, ceval, cgen, bfit, bgfit, avefit):
            self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
            fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")  
            fp = open(fname, "w")  # save summary
            fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f \n' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avefit))
            fp.close()
            
    def initPop(self):
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            self.pop[i] = np.copy(self.policy.get_trainable_flat())

    def mutate(self, individual):
        # Applies Gaussian mutation
        for i in range(self.nparams):
            if self.rg.rand() < self.mutation:
                span = 2.0 * self.policy.wrange
                individual[i] += np.random.normal(0, 0.1 * span)
        return individual

    def runCrossover(self, parent1, parent2):
        # Performs simulated arithmetic crossover.
        if self.rg.rand() < self.crossover:
            alpha = self.rg.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        return parent1.copy(), parent2.copy()

    def tournament_selection(self, fitness, k=2):
        # Selects the best individual from k random candidates.
        selected = self.rg.choice(self.popsize, size=k, replace=False)
        best_idx = selected[np.argmax(fitness[selected])]
        return self.pop[best_idx]
        
    def local_search(self, individual, current_fitness):
        # Stochastic local search (Random Walk / Hill Climber with resampling).
        best_ind = individual.copy()
        best_fit = current_fitness
        span = 2.0 * self.policy.wrange
        step_size = 0.05 * span
        nevals = 0

        for _ in range(self.loc_iters):
            # Perturb
            neighbor = best_ind + self.rg.normal(0, step_size, size=self.nparams)

            # Evaluate with resampling
            self.policy.set_trainable_flat(neighbor)       # set policy parameters
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + self.cgen)  # evaluate the individual
            neighbor_fit = eval_rews                       # store fitness
            nevals += eval_length                          # Update the number of evaluations
            self.updateBest(neighbor_fit, neighbor)        # Update data if the current offspring is better than current best

            if neighbor_fit > best_fit:
                best_fit = neighbor_fit
                best_ind = neighbor

        return best_ind, best_fit, nevals

    def run(self):

        self.loadhyperparameters()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        self.nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        self.cgen = 0                                   # current generation
        self.rg = np.random.RandomState(self.seed)      # create a random generator and initialize the seed
        self.pop = self.rg.randn(self.popsize, self.nparams)      # population
        fitness = zeros(self.popsize)
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        assert ((self.popsize % 2) == 0), print("the size of the population should be odd")

        # initialze the population
        self.initPop()       

        print("MA: seed %d maxmsteps %d popSize %d mutation %lf crossover %lf nparams %d" % (self.seed, self.maxsteps / 1000000, self.popsize, self.mutation, self.crossover, self.nparams))
        
        # If normalize=1 we update the normalization vectors
        if (self.policy.normalize == 1):
            self.policy.nn.updateNormalizationVectors()
            
        self.policy.nn.seed(self.policy.get_seed)  # set the policy seed
            
        # Evaluate the population
        for i in range(self.popsize):
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
        elapsed = 0
        while (ceval < self.maxsteps):
            
            self.cgen += 1

            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            #self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + self.cgen)  # set the policy seed, it changes every generation
            
            # Generate new population
            new_pop = []
            # Elitism (copy the best)
            new_pop.append(bind)
            
            while len(new_pop) < self.popsize:
                # Tournament selection
                p1 = self.tournament_selection(fitness, k=self.tour_size)
                p2 = self.tournament_selection(fitness, k=self.tour_size)
                # Crossover
                c1, c2 = self.runCrossover(p1, p2)
                # Mutation
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                new_pop.append(c1)
                new_pop.append(c2)
            
            # Trim excess if population size is odd
            if len(new_pop) > self.popsize:
               new_pop.pop()
               
            self.pop = np.array(new_pop)
            
            # Evaluate the population
            for i in range(self.popsize):                           
                self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + self.cgen)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
                
            # Apply Memetic Local Search periodically to top individuals
            if self.cgen % self.loc_freq == 0:
                # Sort population by fitness to pick elites for local search
                sidx = np.argsort(fitness)[::-1]
                # Refine top 10% of the population
                num_to_refine = max(1, int(self.loc_top * self.popsize))
                for i in range(num_to_refine):
                    idx = sidx[i]
                    refined_ind, refined_fit, nevals = self.local_search(self.pop[idx], fitness[idx])
                    self.pop[idx] = refined_ind
                    fitness[idx] = refined_fit
                    ceval += nevals

            # Update global best
            bidx = np.argmax(fitness)
            if fitness[bidx] > bfit:
                bfit = fitness[bidx]
                bind = self.pop[bidx].copy()
                    
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(bind)     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, bind)            # eventually update the genotype/fitness of the best post-evaluated individual
            
            avgfit = np.average(fitness)

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, self.cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avgfit))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, avgfit])

            # save data
            if ((time.time() - self.last_save_time) > (self.saveeach * 60)):
                self.savedata(ceval, self.cgen, bfit, bgfit, avgfit)
                self.last_save_time = time.time()  

        self.savedata(ceval, self.cgen, bfit, bgfit, avgfit)
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))#
