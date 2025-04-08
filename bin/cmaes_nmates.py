#!/usr/bin/python

# Libraries to be imported
import gym
from gym import spaces
import numpy as np
from numpy import floor, ceil, log, eye, zeros, array, sqrt, sum, dot, tile, outer, real
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
from utils import descendent_sort

# Evolve with CMA-ES algorithm (Hansen and Ostermeier, 2001)
# This code comes from PyBrain (Schaul et al., 2010) with
# some custom modifications
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
                if o == "nmates":
                    self.nmates = config.getint("ALGO","nmates")
                    found = 1
                if o == "mating":
                    self.mating = config.getint("ALGO","mating")
                    found = 1
                
                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("batchsize [int]           : number of batches (default 20)")
                    print("saveeach [integer]        : save file every N minutes (default 60)")
                    print("nmates [integer]          : the number of mates each sample is evaluated with")
                    print("mating [0/1]              : types of mating (default 0 -> centroid is the partner), 1 -> random samples with random signs")
                    
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
            self.batchSize = int(4 + floor(3 * log(self.nparams))) # population size, offspring
        self.mu = int(floor(self.batchSize / 2)) # number of parents/points for recombination
        self.weights = log(self.mu + 1) - log(array(range(1, self.mu + 1))) # use array
        self.weights /= sum(self.weights)	# normalize recombination weights array
        self.muEff = sum(self.weights) ** 2 / sum(power(self.weights, 2)) # variance-effective size of mu
        self.cumCov = 4 / float(self.nparams + 4)	# time constant for cumulation for covariance matrix
        self.cumStep = (self.muEff + 2) / (self.nparams + self.muEff + 3) # t-const for cumulation for Size control
        self.muCov = self.muEff # size of mu used for calculating learning rate covLearningRate
        self.covLearningRate = ((1 / self.muCov) * 2 / (self.nparams + 1.4) ** 2 + (1 - 1 / self.muCov) *	   # learning rate for
                        ((2 * self.muEff - 1) / ((self.nparams + 2) ** 2 + 2 * self.muEff)))		   # covariance matrix
        self.dampings = 1 + 2 * max(0, sqrt((self.muEff - 1) / (self.nparams + 1)) - 1) + self.cumStep	   
                        # damping for stepSize usually close to 1 former damp == dampings/cumStep
        # Initialize dynamic (internal) strategy parameters and constants
        self.covPath = zeros(self.nparams)
        self.stepPath = zeros(self.nparams)               # evolution paths for C and stepSize
        self.B = eye(self.nparams, self.nparams)               # B defines the coordinate system
        self.D = eye(self.nparams, self.nparams)               # diagonal matrix D defines the scaling
        self.C = dot(dot(self.B, self.D), dot(self.B, self.D).T)         # covariance matrix
        self.chiN = self.nparams ** 0.5 * (1 - 1. / (4. * self.nparams) + 1 / (21. * self.nparams ** 2))
                                                # expectation of ||numParameters(0,I)|| == norm(randn(numParameters,1))
        self.stepsize = 0.5
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
        # Data for sequential fitness
        self.bestfitPost = -99999999
        self.bestsolPost = None
        self.bestgfitPost = -99999999
        self.bestgsolPost = None
        
    def savedata(self):
        self.save()             # save the best agent so far, the best postevaluated agent so far, and progress data across generations
        # We save the centroid, the momentum vectors, the best fitnesses found, the generation and the number of steps performed so far
        # Centroid
        fname = os.path.join(self.filedir, "centerS" + str(self.seed))
        np.save(fname, self.center)
        # Centroid with normalization
        fname = os.path.join(self.filedir, "centerNormS" + str(self.seed))
        np.save(fname, np.append(self.center, self.policy.normvector))
        # Best fitnesses
        fname = os.path.join(self.filedir, "bestFitS" + str(self.seed) + ".txt")
        fp = open(fname, "w")
        fp.write("%lf\t%lf\n" % (self.bestfit, self.bestgfit))
        fp.close()
        # Generation and steps
        fname = os.path.join(self.filedir, "genAndStepsS" + str(self.seed) + ".txt")
        fp = open(fname, "w")
        fp.write("%d\t%d\n" % (self.cgen, self.steps))
        fp.close()
        # Statistics
        fname = os.path.join(self.filedir, "S" + str(self.seed) + ".fit")
        fp = open(fname, "w")   # save summary
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avgfit %.2f paramsize %.2f \n' %
             (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter))
        fp.close()
        
    def loaddata(self):
        # We load the centroid, the momentum vectors, the best fitnesses found, the generation and the number of steps performed so far
        # Centroid
        fname = os.path.join(self.filedir, "centerS" + str(self.seed) + ".npy")
        self.center = np.load(fname, allow_pickle=True)
        self.avecenter = np.average(np.absolute(self.center))
        # Best solutions
        fname = os.path.join(self.filedir, "bestS" + str(self.seed) + ".npy")
        self.bestsol = np.load(fname, allow_pickle=True)
        fname = os.path.join(self.filedir, "bestgS" + str(self.seed) + ".npy")
        self.bestgsol = np.load(fname, allow_pickle=True)
        # Best fitnesses
        fname = os.path.join(self.filedir, "bestFitS" + str(self.seed) + ".txt")
        fit = np.loadtxt(fname)
        assert len(fit) == 2, "Invalid number of fitness values!!!"
        self.bestfit = fit[0]
        self.bestgfit = fit[1]
        # Generation and steps
        fname = os.path.join(self.filedir, "genAndStepsS" + str(self.seed) + ".txt")
        genAndSteps = np.loadtxt(fname)
        assert len(genAndSteps) == 2, "Inconsistent number of data!!!"
        self.cgen = int(genAndSteps[0])
        self.steps = int(genAndSteps[1])
        
    def generateMates(self, ind):
        mates = []
        while len(mates) < self.nmates:
            idx = np.random.randint(0, self.batchSize)
            if idx != ind:
                mates.append(idx)
        assert len(mates) == self.nmates, "Something went wrong during the generation of mates!!!"
        return mates
 
    def evaluate(self):
        cseed = self.seed + self.cgen * self.batchSize  # Set the seed for current generation (master and workers have the same seed)
        self.rs = np.random.RandomState(cseed)
        self.samples = self.rs.randn(self.nparams, self.batchSize)
        self.cgen += 1
        
        # Check that the task is multi-agent and requires heterogeneity, otherwise we cannot use N-mates method
        heterogeneous = self.policy.is_heterogeneous
        assert heterogeneous == 1, "Cannot use N-mates method with homogeneous agents"

        # Generate offspring
        self.offspring = tile(self.center.reshape(self.nparams, 1), (1, self.batchSize)) + self.stepsize * dot(dot(self.B, self.D), self.samples)
        for k in range(self.batchSize):
            # Extract <nmates> random partners
            mates = self.generateMates(k)
            self.samplefitness[k] = 0.0
            for i in range(self.nmates):
                # Set policy parameters (corresponding to the current offspring)
                self.policy.set_trainable_flat(np.concatenate((self.offspring[:,k], self.offspring[:,mates[i]])))
                self.policy.nn.normphase(0) # normalization data is collected during the post-evaluation of the best sample of he previous generation
                # Evaluate the offspring
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=(self.seed + (self.cgen * self.batchSize) + k))
                # Get the fitness
                self.samplefitness[k] += eval_rews
                # Update the number of evaluations
                self.steps += eval_length
            self.samplefitness[k] /= self.nmates

        # Sort by fitness and compute weighted mean into center
        self.fitness, self.index = descendent_sort(self.samplefitness)
        self.avgfit = np.average(self.fitness)                         # compute the average fitness                   

        self.bfit = self.fitness[0]
        bidx = self.index[0]
        self.updateBest(self.bfit, self.offspring[:,bidx])                  # Stored if it is the best obtained so far

        # postevaluate best sample of the last generation
        # in openaiesp.py this is done the next generation, move this section before the section "evaluate samples" to produce identical results
        gfit = 0
        if self.policy.nttrials > 0 and self.bestsol is not None:
            mates = self.generateMates(bidx)
            for i in range(self.nmates):  
                self.policy.set_trainable_flat(np.concatenate((self.bestsol, self.offspring[:,mates[i]])))
                self.tnormepisodes += self.inormepisodes
                cfit = 0.0
                for t in range(self.policy.nttrials):
                    if self.policy.normalize == 1 and self.normepisodes < self.tnormepisodes:
                        self.policy.nn.normphase(1)
                        self.normepisodes += 1  # we collect normalization data
                        self.normalizationdatacollected = True
                    else:
                        self.policy.nn.normphase(0)
                    eval_rews, eval_length = self.policy.rollout(1, seed=(self.seed + 100000 + t))
                    cfit += eval_rews
                    self.steps += eval_length
                cfit /= self.policy.nttrials
                gfit += cfit
            gfit /= self.nmates   
            self.updateBestg(gfit, self.bestsol)
            
    def optimize(self):
        # Re-organize samples according to indices
        self.samples = self.samples[:, self.index]
        # Do the same for offspring
        self.offspring = self.offspring[:, self.index]
        # Select best <mu> samples and offspring for computing new center and cumulation paths
        samsel = self.samples[:, range(self.mu)]
        offsel = self.offspring[:, range(self.mu)]
        offmut = offsel - tile(self.center.reshape(self.nparams, 1), (1, self.mu))

        samplemean = dot(samsel, self.weights)
        self.center = dot(offsel, self.weights)
        self.avecenter = np.average(np.absolute(self.center))

        # Cumulation: Update evolution paths
        self.stepPath = (1 - self.cumStep) * self.stepPath \
                     + sqrt(self.cumStep * (2 - self.cumStep) * self.muEff) * dot(self.B, samplemean)		 # Eq. (4)
        hsig = norm(self.stepPath) / sqrt(1 - (1 - self.cumStep) ** (2 * self.steps / float(self.batchSize))) / self.chiN \
                     < 1.4 + 2. / (self.nparams + 1)
        self.covPath = (1 - self.cumCov) * self.covPath + hsig * \
                     sqrt(self.cumCov * (2 - self.cumCov) * self.muEff) * dot(dot(self.B, self.D), samplemean) # Eq. (2)

        # Adapt covariance matrix C
        self.C = ((1 - self.covLearningRate) * self.C					# regard old matrix   % Eq. (3)
                     + self.covLearningRate * (1 / self.muCov) * (outer(self.covPath, self.covPath) # plus rank one update
                     + (1 - hsig) * self.cumCov * (2 - self.cumCov) * self.C)
                     + self.covLearningRate * (1 - 1 / self.muCov)				 # plus rank mu update
                     * dot(dot(offmut, diag(self.weights)), offmut.T)
                )

        # Adapt step size
        self.stepsize *= exp((self.cumStep / self.dampings) * (norm(self.stepPath) / self.chiN - 1)) # Eq. (5)

        # Update B and D from C
        # This is O(n^3). When strategy internal CPU-time is critical, the
        # next three lines should be executed only every (alpha/covLearningRate/N)-th
        # iteration, where alpha is e.g. between 0.1 and 10
        self.C = (self.C + self.C.T) / 2     # enforce symmetry
        Ev, self.B = eig(self.C)        # eigen decomposition, B==normalized eigenvectors
        Ev = real(Ev)	  # enforce real value
        self.D = diag(sqrt(Ev))    #diag(ravel(sqrt(Ev))) # D contains standard deviations now
        self.B = real(self.B)
        
    def run(self):

        self.setProcess()                           # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        print("CMA-ES: seed %d maxmsteps %d batchSize %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.batchSize, self.nparams))
        
        # Load data (if any)
        try:
            self.loaddata()
        except:
            #print("No data to be loaded or some errors occurred!!!")
            pass

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

