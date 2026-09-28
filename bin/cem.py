#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
"""

import numpy as np
from numpy import zeros, ones, dot, sqrt, eye, log
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort
import sys
import os
import configparser
import random
import copy

class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.batchSize = 20
            self.elites = 10
            self.saveeach = 60
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "samplesize":
                    self.batchSize = config.getint("ALGO","samplesize")
                    found = 1
                if o == "elites":
                    self.elites = config.getint("ALGO","elites")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("samplesize [int]          : popsize (default 20)")
                    print("elites [int]              : elites samples to update mean and variance")
                    print("saveeach [integer]        : save file every N minutes (default 60)")
                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))

    def setProcess(self):
        self.loadhyperparameters()               # load hyperparameters
        self.nparams = self.policy.nparams       # number of parameters
        self.mu = zeros(self.nparams)            # mean of the distribution
        self.sigma = ones(self.nparams)          # standard deviation of the distribution
        self.cgen = 0                            # currrent generation
        self.samplefitness = zeros(self.batchSize) # the fitness of the samples
        self.samples = None                      # the random samples
        self.fitness = None                      # sorted fitness of the samples
        self.rs = None
        #self.I = eye(self.nparams)
        self.bestgfit = -99999999                # the best generalization fitness
        self.bfit = 0                            # the fitness of the best sample
        self.gfit = 0                            # the postevaluation fitness of the best sample of last generation
        self.inormepisodes = self.batchSize * 2 * self.policy.ntrials / 100.0 # number of normalization episode for generation (1% of generation episodes)
        self.tnormepisodes = 0.0                 # total epsidoes in which normalization data should be collected so far
        self.normepisodes = 0                    # numer of episodes in which normalization data has been actually collected so far
        self.normalizationdatacollected = False  # whether we collected data for updating the normalization vector

    def savedata(self):
        self.save()             # save the best agent so far, the best postevaluated agent so far, and progress data across generations
        # We save the centroid, the momentum vectors, the best fitnesses found, the generation and the number of steps performed so far
        # Statistics
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")
        fp = open(fname, "w")   # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avgfit %.2f \n' %
             (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit))
        fp.close()
 
    def evaluate(self):
        cseed = self.seed + self.cgen * self.batchSize  # Set the seed for current generation (master and workers have the same seed)
        self.rs = np.random.RandomState(cseed)
        self.samples = zeros((self.batchSize, self.nparams))
        for i in range(self.nparams):
            self.samples[:,i] = self.rs.normal(loc=self.mu[i], scale=self.sigma[i]+1e-17, size=(self.batchSize,))
        self.cgen += 1

        # evaluate samples
        for b in range(self.batchSize):               
            self.policy.set_trainable_flat(self.samples[b,:])
            self.policy.nn.normphase(0) # normalization data is collected during the post-evaluation of the best sample of he previous generation
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=(self.seed + (self.cgen * self.batchSize)))
            self.samplefitness[b] = eval_rews
            self.steps += eval_length

        self.fitness, self.index = descendent_sort(self.samplefitness)       # sort the fitness
        self.avgfit = np.average(self.fitness)                         # compute the average fitness                   

        self.bfit = self.fitness[0]
        bidx = self.index[0]
        self.updateBest(self.bfit, self.samples[bidx])                  # Stored if it is the best obtained so far 
                
        # postevaluate best sample of the last generation
        # in openaiesp.py this is done the next generation, move this section before the section "evaluate samples" to produce identical results
        gfit = 0
        if self.policy.nttrials > 0 and self.bestsol is not None:
            self.policy.set_trainable_flat(self.bestsol)
            self.tnormepisodes += self.inormepisodes
            for t in range(self.policy.nttrials):
                if self.policy.normalize == 1 and self.normepisodes < self.tnormepisodes:
                    self.policy.nn.normphase(1)
                    self.normepisodes += 1  # we collect normalization data
                    self.normalizationdatacollected = True
                else:
                    self.policy.nn.normphase(0)
                eval_rews, eval_length = self.policy.rollout(1, seed=(self.seed + 100000 + t))
                gfit += eval_rews               
                self.steps += eval_length
            gfit /= self.policy.nttrials    
            self.updateBestg(gfit, self.bestsol)

    def optimize(self):
    
        # Select elites samples based on fitness
        x = []
        for i in range(self.elites):
            x.append(self.samples[self.index[i],:])
        self.mu = np.array(x).mean(axis=0)
        self.sigma = np.array(x).std(axis=0)

    def run(self):

        self.setProcess()                           # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        print("Cross-Entropy Method (CEM): seed %d maxmsteps %d batchSize %d elites %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.batchSize, self.elites, self.nparams))

        while (self.steps < self.maxsteps):
                    
            self.evaluate()                           # evaluate samples  
            
            self.optimize()                           # estimate the gradient and move the centroid in the gradient direction

            self.stat = np.append(self.stat, [self.steps, self.bestfit, self.bestgfit, self.bfit, self.avgfit])  # store performance across generations

            if ((time.time() - last_save_time) > (self.saveeach * 60)):
                self.savedata()                       # save data on files
                last_save_time = time.time()

            if self.normalizationdatacollected:
                self.policy.nn.updateNormalizationVectors()  # update the normalization vectors with the new data collected
                self.normalizationdatacollected = False

            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avg %.2f' %
                      (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit))

        self.savedata()                           # save data at the end of evolution

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

