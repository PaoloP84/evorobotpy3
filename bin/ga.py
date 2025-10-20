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
import random

# Evolve with genetic algorithm
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)
        self.popSize = 100

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popSize = 100
            self.mutation = 0.02
            self.crossover = False
            self.crossrate = 0.0
            self.crosstype = 0
            self.elitism = False
            self.tournamentSize = 2
            self.saveEveryN = 1
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
                if o == "mutation":
                    self.mutation = config.getfloat("ALGO","mutation")
                    found = 1
                if o == "crossover":
                    cross = config.getint("ALGO","crossover")
                    self.crossover = bool(cross)
                    found = 1
                if o == "crossrate":
                    self.crossrate = config.getfloat("ALGO","crossrate")
                    found = 1
                if o == "crosstype":
                    self.crosstype = config.getint("ALGO","crosstype")
                    found = 1
                if o == "elitism":
                    elitism = config.getint("ALGO","elitism")
                    self.elitism = bool(elitism)
                    found = 1
                if o == "tournamentsize":
                    self.tournamentSize = config.getint("ALGO","tournamentsize")
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
        
    def savePop(self):
        fname = os.path.join(self.filedir, "popS" + str(self.seed))
        np.save(fname, self.pop)
        
    def loadPop(self):
        self.pop = np.load(os.path.join(self.filedir, "popS" + str(self.seed) + ".npy"), allow_pickle=True)

    def initPop(self):
        for i in range(self.popSize):
            self.policy.nn.initWeights()
            self.pop[i] = np.copy(self.policy.get_trainable_flat())

    def mutate(self, parent):
        child = np.zeros(self.nparams, dtype=np.float64)
        for g in range(self.nparams):
            child[g] = parent[g]
            coin = self.rg.uniform(0.0, 1.0)
            if coin < self.mutation:
                child[g] += self.rg.uniform(-self.policy.wrange, self.policy.wrange)
        return child

    def runCrossover(self, father, mother):
        # We get length from smallest individual (in case of individuals with different lengths!!!)
        ngenes = len(father)
        if len(mother) < ngenes:
            ngenes = len(mother)
        # Cross-over point is at half of the length
        point = np.random.randint(0, ngenes)
        child = np.zeros(self.nparams, dtype=np.float64)
        other_child = np.zeros(self.nparams, dtype=np.float64)
        # Fill children
        child[0:point] = father[0:point]
        other_child[0:point] = mother[0:point]
        child[point:] = mother[point:]
        other_child[point:] = father[point:]
        return child, other_child

    def select(self, fitness, tournament_size=2):
        selected = []
        for _ in range(self.popSize):
            tournament = random.sample(list(zip(self.pop, fitness)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

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

        print("Generational: seed %d maxsteps %d popSize %d mutRate %lf" % (self.seed, self.maxsteps, self.popSize, self.mutation))
        
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
                self.policy.set_trainable_flat(self.pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], self.pop[i])           # Update data if the current offspring is better than current best
                
            # Store best individual
            sfitness, sindex = descendent_sort(fitness) # create an index with the ID of the individuals sorted for fitness  
            bfit = sfitness[0]
            bind = self.pop[sindex[0]]
            
            # Next population (void if reproduction does not occur)    
            next_pop = []
            # Perform reproduction
            if ceval < self.maxsteps:
                # Tournament selection
                self.pop = self.select(fitness, tournament_size=self.tournamentSize)
                # Create next population with cross-over and reproduction
                for i in range(0, self.popSize, 2):
                    # Extract parents
                    parent1 = self.pop[i]
                    parent2 = self.pop[i + 1]
                    # Cross-over
                    child1, child2 = self.runCrossover(parent1, parent2)
                    # Mutate children and add to next population
                    next_pop.append(self.mutate(child1))
                    next_pop.append(self.mutate(child2))
                assert len(next_pop) == self.popSize, "Mismatch between lengths"
                # Check whether the best individual must be kept in the population (elitism)
                if self.elitism: 
                    # Keep best individual in the population
                    idx = np.random.randint(0, self.popSize)
                    next_pop[idx] = bind  
                    bidx = idx 
                # Replace population
                self.pop = next_pop  
                
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

            if (cgen % self.saveEveryN) == 0:
                self.savedata(ceval, cgen, bfit, bgfit, avgf) 

        self.savedata(ceval, cgen, bfit, bgfit, avgf)
        self.savePop()
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

