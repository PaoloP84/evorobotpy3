#!/usr/bin/python

# Libraries to be imported
import gym
from gym import spaces
import numpy as np
from numpy import floor, log, eye, zeros, array, sqrt, sum, dot, tile, outer, real
from numpy import exp, diag, power, ravel
from numpy.linalg import eig, norm
from numpy.random import randn
import math
import random
import time
from scipy import zeros, ones
from scipy.linalg import expm
import configparser
import sys
import os
from six import integer_types
import struct
import net
from policy import ErPolicy, GymPolicy
from evoalgo import EvoAlgo
from utils import ascendent_sort

# Evolve with sNES algorithm (Wierstra, Schaul, Peters and Schmidhuber, 2008)
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)
        
    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.batchSize = 20
            self.saveeach = 60
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "batchsize":
                    self.batchSize = config.getint("ALGO","batchSize")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("batchsize [int]           : number of batches (default 20)")
                    print("saveeach [integer]        : save file every N minutes (default 60)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))
            
    def setProcess(self):
        self.loadhyperparameters()               # load hyperparameters
        self.center = np.copy(self.policy.get_trainable_flat())  # the initial centroid
        self.nparams = len(self.center)          # number of adaptive parameters
        # setting parameters
        if self.batchSize == 0:
            # Use default value: 4 + floor(3 * log(N)), where N is the number of parameters
            self.batchSize = int(4 + floor(3 * log(nparams))) # population size, offspring
        self.mu = int(floor(self.batchSize / 2))                       # number of parents/points for recombination
        self.initVar = 1.0
        # setting parameters
        self.centerLearningRate = 1.0
        self.covLearningRate = 0.5 * min(1.0 / self.nparams, 0.25)          # from MATLAB # covLearningRate = 0.6*(3+log(ngenes))/ngenes/sqrt(ngenes)
        self.stepsize = 1.0 / self.mu
        self.weights = zeros(self.batchSize)
        w = self.stepsize
        for i in range(self.mu):
            self.weights[self.batchSize - self.mu + i] = w
            w += self.stepsize
        self.weights /= sum(self.weights)
        # initialize variance array
        self._sigmas = ones(self.nparams) * self.initVar
        
        self.cgen = 0                            # currrent generation
        self.samplefitness = zeros(self.batchSize) # the fitness of the samples
        self.samples = None                      # the random samples
        self.bestgfit = -99999999                # the best generalization fitness
        self.bfit = 0                            # the fitness of the best sample
        self.gfit = 0                            # the postevaluation fitness of the best sample of last generation
        self.rs = None                           # random number generator
        self.inormepisodes = self.batchSize * 2 * self.policy.ntrials / 100.0 # number of normalization episode for generation (1% of generation episodes)
        self.tnormepisodes = 0.0                 # total epsidoes in which normalization data should be collected so far
        self.normepisodes = 0                    # numer of episodes in which normalization data has been actually collected so far
        self.normalizationdatacollected = False  # whether we collected data for updating the normalization vector
        
    def savedata(self):
        self.save()             # save the best agent so far, the best postevaluated agent so far, and progress data across generations
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")
        fp = open(fname, "w")   # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avgfit %.2f paramsize %.2f \n' %
             (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter))
        fp.close()
        
    def evaluate(self):
        cseed = self.seed + self.cgen * self.batchSize  # Set the seed for current generation (master and workers have the same seed)
        self.rs = np.random.RandomState(cseed)
        self.samples = self.rs.randn(self.batchSize, self.nparams)
        self.S = self.samples.transpose()
        self.cgen += 1

        # Generate offspring
        self.offspring = tile(self.center.reshape(1, self.nparams), (self.batchSize, 1)) + tile(self._sigmas.reshape(1, self.nparams), (self.batchSize, 1)) * self.samples
        for k in range(self.batchSize):
            # Set policy parameters (corresponding to the current offspring)
            self.policy.set_trainable_flat(self.offspring[k])
            self.policy.nn.normphase(0) # normalization data is collected during the post-evaluation of the best sample of he previous generation
            # Evaluate the offspring
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=(self.seed + (self.cgen * self.batchSize) + k))
            # Get the fitness
            self.samplefitness[k] = eval_rews
            # Update the number of evaluations
            self.steps += eval_length

        # Sort by fitness and compute weighted mean into center
        self.fitness, self.index = ascendent_sort(self.samplefitness)
        self.avgfit = np.average(self.fitness)                         # compute the average fitness                   

        self.bfit = self.fitness[self.batchSize - 1]
        bidx = self.index[self.batchSize - 1]
        self.updateBest(self.bfit, self.offspring[bidx])                  # Stored if it is the best obtained so far
        
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
    
        self.S = self.S[:, self.index]

        # Update center
        dCenter = dot(self.weights, self.S.transpose())
        self.center += dCenter
        self.avecenter = np.average(np.absolute(self.center))
        
        # Update variances
        Ssq = self.S * self.S
        SsqMinusOne = Ssq - ones((self.nparams, self.batchSize))
        covGrad = dot(self.weights, SsqMinusOne.transpose())
        dSigma = 0.5 * self.covLearningRate * covGrad
        self._sigmas = self._sigmas * exp(dSigma).transpose()

    def run(self):

        self.setProcess()                           # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        print("sNES: seed %d maxmsteps %d batchSize %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.batchSize, self.nparams))

        while (self.steps < self.maxsteps):
            
            self.evaluate()                           # evaluate samples  
            
            self.optimize()                           # estimate the gradient and move the centroid in the gradient direction

            self.stat = np.append(self.stat, [self.steps, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter])  # store performance across generations

            if ((time.time() - last_save_time) > (self.saveeach * 60)):
                self.savedata()                       # save data on files
                last_save_time = time.time()

            if self.normalizationdatacollected:
                self.policy.nn.updateNormalizationVectors()  # update the normalization vectors with the new data collected
                self.normalizationdatacollected = False

            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avg %.2f weightsize %.2f' %
                      (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter))

        self.savedata()                           # save data at the end of evolution

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

