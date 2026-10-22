#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py

   It implements the Simulated Annealing (SA) algorithm proposed in:

   Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680.
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

# Evolve with simulated annealing (SA) algorithm
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def readConfig(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.mutation = 0.02
            self.temp = 10.0
            self.cool_rate = 0.95
            self.saveeach = 1
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
                if o == "temp":
                    self.temp = config.getfloat("ALGO","temp")
                    found = 1
                if o == "cool_rate":
                    self.cool_rate = config.getfloat("ALGO","cool_rate")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("Option %s in section [ALGO] of %s file is unknown" % (o, self.fileini))
                    sys.exit()
        else:
            print("ERROR: configuration file %s does not exist" % (self.fileini))
        
    def savedata(self, ceval, cgen, bfit, bgfit):
        self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")  
        fp = open(fname, "w")  # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f\n' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit))
        fp.close()
        
    def mutate(self):
        var = np.zeros(self.nparams, dtype=np.float64)
        for g in range(self.nparams):
            var[g] = self.center[g]
            coin = self.rg.uniform(0.0, 1.0)
            if coin <= self.mutation:
                var[g] += self.rg.uniform(-self.policy.wrange, self.policy.wrange)
        return var

    def run(self):

        self.readConfig()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        self.nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        self.rg = np.random.RandomState(self.seed) # create a random generator and initialize the seed
        self.center = self.policy.get_trainable_flat() # individual
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations
        
        # Initialize temperature
        temp = self.temp

        print("Simulated Annealing: seed %d maxsteps %d mutRate %lf temperature %lf" % (self.seed, self.maxsteps, self.mutation, temp))

        # Initialize
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
            
            # Evaluate the candidate solution
            self.policy.set_trainable_flat(self.center)        # set policy parameters
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
            bfit = cfit = eval_rews # Center is the current best
            bind = self.center
            self.updateBest(cfit, self.center)           # Update data if the center is better than current best
            
            # Generate a variation through mutation
            neighbor = self.mutate()
            self.policy.set_trainable_flat(neighbor)        # set policy parameters
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
            nfit = eval_rews                        # store fitness
            ceval += eval_length                          # Update the number of evaluations
            self.updateBest(nfit, neighbor)           # Update data if the neighbor is better than current best
                
            # Check which solution must be retained
            accept = False
            # Compute fitness (energy) difference
            diff = nfit - cfit
            
            # If diff > 0, neighbor is better and we accept this as the new center
            if diff > 0:
                accept = True  
            else:
                # Accept the worse solution with a given probability, which depends on the temperature
                # Check den value, so that we avoid numerical issues with math.exp()
                """
                den = max(temp, 1e-10)
                if den <= 1e-3:
                    p = 1.0
                else:
                    p = math.exp(-diff / den)
                """
                p = math.exp(diff / max(temp, 1e-10))
                accept = np.random.random() < p
                
            if accept:
                # Update center with the neighbor
                bind = self.center = neighbor
                bfit = nfit
       
            # Update temperature (Cooling Schedule)
            temp *= self.cool_rate
                
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(bind)     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, bind)            # eventually update the genotype/fitness of the best post-evaluated individual

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit])

            if (cgen % self.saveeach) == 0:
                self.savedata(ceval, cgen, bfit, bgfit) 

        self.savedata(ceval, cgen, bfit, bgfit)
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

