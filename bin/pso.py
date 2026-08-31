#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   
   It implements the Particle Swarm Optimization (PSO) algorithm introduced in:

   Eberhart, R., & Kennedy, J. (1995). Particle swarm optimization. 
   In Proceedings of the IEEE international conference on neural networks (Vol. 4, pp. 1942-1948).
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

# Particle Swarm Optimization (PSO)
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popsize = 20 # Number of particles
            self.w = 0.7 # Coefficient for velocities (called inertia weight)
            self.c1 = 2.0 # Coefficient for personal acceleration
            self.c2 = 2.0 # Coefficient for social acceleration
            self.saveeach = 60
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "w":
                    self.w = config.getfloat("ALGO","w")
                    found = 1
                if o == "popsize":
                    self.popsize = config.getint("ALGO","popsize")
                    found = 1
                if o == "c1":
                    self.c1 = config.getfloat("ALGO","c1")
                    found = 1
                if o == "c2":
                    self.c2 = config.getfloat("ALGO","c2")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("popsize [int]             : popsize (20)")
                    print("w [float]                 : inertia weight")
                    print("c1 [float]                : personal acceleration factor")
                    print("c2 [float]                : social acceleration factor")
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

    def run(self):
        self.loadhyperparameters()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        rg = np.random.RandomState(self.seed)      # create a random generator and initialize the seed
        particles = rg.randn(self.popsize, nparams)      # particles
        velocities = zeros((self.popsize, nparams), dtype=np.float64) # velocities
        fitness = zeros(self.popsize)
        old_fitness = zeros(self.popsize)
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        # initialze the population
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            particles[i] = self.policy.get_trainable_flat()  
            
        print("PSO: seed %d maxmsteps %d popSize %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.popsize, nparams))

        # Evaluate the population
        bfit = -99999999.0
        bid = -1
        for i in range(self.popsize):                           
            self.policy.set_trainable_flat(particles[i])        # set policy parameters
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
            old_fitness[i] = eval_rews                        # store fitness
            ceval += eval_length                          # Update the number of evaluations
            self.updateBest(old_fitness[i], particles[i])     # Update data if the current offspring is better than current best
            if old_fitness[i] > bfit:
                bfit = old_fitness[i]
                bid = i     

        # Initialize best positions
        best_positions = np.copy(particles)
        swarm_best_position = particles[bid]
        swarm_best_fitness = old_fitness[bid]
        
        # Postevaluate the best individual
        self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
        self.policy.set_trainable_flat(particles[bid])     # set the parameters of the policy
        eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
        bgfit = eval_rews
        ceval += eval_length
        self.updateBestg(bgfit, particles[bid])            # eventually update the genotype/fitness of the best post-evaluated individual

        # main loop
        elapsed = 0
        while (ceval < self.maxsteps):
            
            cgen += 1
            
            r1 = rg.uniform(0, 1, self.popsize) #defining a random coefficient for personal behavior
            r2 = rg.uniform(0, 1, self.popsize) #defining a random coefficient for social behavior
            
            for i in range(self.popsize):
                velocities[i] = self.w * velocities[i] + self.c1 * r1[i] * (best_positions[i] - particles[i]) + self.c2 * r2[i] * (swarm_best_position[i] - particles[i])
            #velocities = np.array(self.w * velocities + self.c1 * r1 * (best_positions - particles) + self.c2 * r2 * (swarm_best_position - particles)) # calculating velocities
                particles[i] += velocities[i]

            #particles += velocities # updating position by adding the velocity

            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            #self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Evaluate the population
            bfit = -99999999.0
            bid = -1
            for i in range(self.popsize):                           
                self.policy.set_trainable_flat(particles[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=self.policy.get_seed + cgen)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], particles[i])     # Update data if the current offspring is better than current best
                if fitness[i] > bfit:
                    bfit = fitness[i]
                    bid = i
            
            # Retrieve indices of particles improving their fitness
            idx = np.where(fitness > old_fitness)
            best_positions[idx] = particles[idx] # updating the best positions with the new particles
            old_fitness[idx] = fitness[idx] #updating gains

            if np.max(fitness) > swarm_best_fitness: #if current maxima is greateer than across all previous iters, than assign
                swarm_best_position = particles[np.argmax(fitness)] #assigning the best candidate solution
                swarm_best_fitness = np.max(fitness) #assigning the best gain   
            
            # Postevaluate the best individual
            #self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(particles[bid])     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, seed=self.policy.get_seed + 100000)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, particles[bid])            # eventually update the genotype/fitness of the best post-evaluated individual
            
            #print(bgfit)
            
            avgfit = np.average(fitness[0:self.popsize])

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avgfit, np.average(np.absolute(particles[bid]))))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, avgfit])

            # save data
            if ((time.time() - self.last_save_time) > (self.saveeach * 60)):
                self.savedata(ceval, cgen, bfit, bgfit, avgfit, np.average(np.absolute(particles[bid])))
                self.last_save_time = time.time()  

        self.savedata(ceval, cgen, bfit, bgfit, avgfit, np.average(np.absolute(particles[bid])))
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))#
