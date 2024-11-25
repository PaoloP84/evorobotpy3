#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it
   salimans.py include an implementation of the OpenAI-ES algorithm described in
   Salimans T., Ho J., Chen X., Sidor S & Sutskever I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864v2
   requires es.py, policy.py, and evoalgo.py 
"""

import numpy as np
from numpy import zeros, ones, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort
import sys
import os
import configparser
import copy

# Parallel implementation of Open-AI-ES algorithm developed by Salimans et al. (2017)
# the workers evaluate a fraction of the population in parallel
# the master post-evaluate the best sample of the last generation and eventually update the input normalization vector

class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)
        self.testmode = 0

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.stepsize = 0.01
            self.batchSize = 20
            self.noiseStdDev = 0.02
            self.wdecay = 0
            self.symseed = 1
            self.saveeach = 60
            self.testmode = 0
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "stepsize":
                    self.stepsize = config.getfloat("ALGO","stepsize")
                    found = 1
                if o == "noisestddev":
                    self.noiseStdDev = config.getfloat("ALGO","noiseStdDev")
                    found = 1
                if o == "samplesize":
                    self.batchSize = config.getint("ALGO","sampleSize")
                    found = 1
                if o == "wdecay":
                    self.wdecay = config.getint("ALGO","wdecay")
                    found = 1
                if o == "symseed":
                    self.symseed = config.getint("ALGO","symseed")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1
                if o == "testmode":
                    self.testmode = config.getint("ALGO","testmode")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("stepsize [float]          : learning stepsize (default 0.01)")
                    print("samplesize [int]          : popsize/2 (default 20)")
                    print("noiseStdDev [float]       : samples noise (default 0.02)")
                    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
                    print("symseed [0/1]             : same environmental seed to evaluate symmetrical samples [default 1]")
                    print("saveeach [integer]        : save file every N minutes (default 60)")
                    print("testmode [0/1]            : test mode (default 0), 1 = functional network test")
                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))
    


    def setProcess(self):
        self.loadhyperparameters()               # load hyperparameters
        self.center = np.copy(self.policy.get_trainable_flat())  # the initial centroid
        # Save initial centroid
        fname = os.path.join(self.filedir, "initCenterS" + str(self.seed))
        np.save(fname, self.center)
        self.avecenter = np.average(np.absolute(self.center))
        self.nparams = len(self.center)          # number of adaptive parameters
        self.cgen = 0                            # currrent generation
        self.samplefitness = zeros(self.batchSize * 2) # the fitness of the samples
        self.samples = None                      # the random samples
        self.m = zeros(self.nparams)             # Adam: momentum vector 
        self.v = zeros(self.nparams)             # Adam: second momentum vector (adam)
        self.epsilon = 1e-08                     # Adam: To avoid numerical issues with division by zero...
        self.beta1 = 0.9                         # Adam: beta1
        self.beta2 = 0.999                       # Adam: beta2
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
        # We save the centroid, the momentum vectors, the best fitnesses found, the generation and the number of steps performed so far
        # Centroid
        fname = os.path.join(self.filedir, "centerS" + str(self.seed))
        np.save(fname, self.center)
        # Momentum vectors
        # m
        fname = os.path.join(self.filedir, "mS" + str(self.seed))
        np.save(fname, self.m)
        # v
        fname = os.path.join(self.filedir, "vS" + str(self.seed))
        np.save(fname, self.v)
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
        # Momentum vectors
        # m
        fname = os.path.join(self.filedir, "mS" + str(self.seed) + ".npy")
        self.m = np.load(fname, allow_pickle=True)
        # v
        fname = os.path.join(self.filedir, "vS" + str(self.seed) + ".npy")
        self.v = np.load(fname, allow_pickle=True)
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
 
    def evaluate(self):
        cseed = self.seed + self.cgen * self.batchSize  # Set the seed for current generation (master and workers have the same seed)
        self.rs = np.random.RandomState(cseed)
        self.samples = self.rs.randn(self.batchSize, self.nparams)
        self.cgen += 1

        # evaluate samples
        candidate = np.arange(self.nparams, dtype=np.float64)
        for b in range(self.batchSize):               
            for bb in range(2):
                if (bb == 0):
                    candidate = self.center + self.samples[b,:] * self.noiseStdDev
                else:
                    candidate = self.center - self.samples[b,:] * self.noiseStdDev
                self.policy.set_trainable_flat(candidate)
                self.policy.nn.normphase(0) # normalization data is collected during the post-evaluation of the best sample of he previous generation
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=(self.seed + (self.cgen * self.batchSize) + b))
                self.samplefitness[b*2+bb] = eval_rews
                self.steps += eval_length

        fitness, self.index = ascendent_sort(self.samplefitness)       # sort the fitness
        self.avgfit = np.average(fitness)                         # compute the average fitness                   

        self.bfit = fitness[(self.batchSize * 2) - 1]
        bidx = self.index[(self.batchSize * 2) - 1]  
        if ((bidx % 2) == 0):                                     # regenerate the genotype of the best samples
            bestid = int(bidx / 2)
            self.bestsol = self.center + self.samples[bestid] * self.noiseStdDev  
        else:
            bestid = int(bidx / 2)
            self.bestsol = self.center - self.samples[bestid] * self.noiseStdDev

        self.updateBest(self.bfit, self.bestsol)                  # Stored if it is the best obtained so far 
                
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
            
        popsize = self.batchSize * 2                              # compute a vector of utilities [-0.5,0.5]
        utilities = zeros(popsize)
        for i in range(popsize):
            utilities[self.index[i]] = i
        utilities /= (popsize - 1)
        utilities -= 0.5
        
        weights = zeros(self.batchSize)                           # Assign the weights (utility) to samples on the basis of their fitness rank
        for i in range(self.batchSize):
            idx = 2 * i
            weights[i] = (utilities[idx] - utilities[idx + 1])    # merge the utility of symmetric samples

        g = 0.0
        i = 0
        while i < self.batchSize:                                 # Compute the gradient (the dot product of the samples for their utilities)
            gsize = -1
            if self.batchSize - i < 500:                          # if the popsize is larger than 500, compute the gradient for multiple sub-populations
                gsize = self.batchSize - i
            else:
                gsize = 500
            g += dot(weights[i:i + gsize], self.samples[i:i + gsize,:]) 
            i += gsize
        g /= popsize                                              # normalize the gradient for the popsize
        
        if self.wdecay == 1:
            globalg = -g + 0.005 * self.center                    # apply weight decay
        else:
            globalg = -g

        # adam stochastic optimizer
        a = self.stepsize * sqrt(1.0 - self.beta2 ** self.cgen) / (1.0 - self.beta1 ** self.cgen)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (globalg * globalg)
        dCenter = -a * self.m / (sqrt(self.v) + self.epsilon)
        
        self.center += dCenter                                    # move the center in the direction of the momentum vectors
        self.avecenter = np.average(np.absolute(self.center))      


    def run(self):

        self.setProcess()                           # initialize class variables
        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        print("OpenAI-ES: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d symseed %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.symseed, self.nparams))
        
        # Load data (if any)
        try:
            self.loaddata()
        except:
            #print("No data to be loaded or some errors occurred!!!")
            pass
            
        # Set the frequency at which centroid will be stored
        saveCenterFreq = int(self.maxsteps / 10) # Can be tuned
        # Moment at which center will be saved
        saveCenterSteps = saveCenterFreq
        # Counter
        saveCenterCnt = 1

        while (self.steps < self.maxsteps):

            
            self.evaluate()                           # evaluate samples  
            
            self.optimize()                           # estimate the gradient and move the centroid in the gradient direction

            self.stat = np.append(self.stat, [self.steps, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter])  # store performance across generations

            if ((time.time() - last_save_time) > (self.saveeach * 60)):
                self.savedata()                       # save data on files
                last_save_time = time.time()
                
            if self.steps >= saveCenterSteps:
                # Save centroid
                fname = os.path.join(self.filedir, "centerS" + str(self.seed) + "_" + str(saveCenterCnt))
                np.save(fname, self.center)
                # Update moment
                saveCenterSteps += saveCenterFreq
                # Update counter
                saveCenterCnt += 1

            if self.normalizationdatacollected:
                self.policy.nn.updateNormalizationVectors()  # update the normalization vectors with the new data collected
                self.normalizationdatacollected = False

            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avg %.2f weightsize %.2f' %
                      (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter))

        self.savedata()                           # save data at the end of evolution

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))
        
    def test(self, testfile):
        self.loadhyperparameters()
        if self.testmode == 0:
            EvoAlgo.test(self, testfile)
        else:
            self.funcTest(testfile)
        
    def funcTest(self, testfile):  # postevaluate an agent 
        print("Functional network test")
        if testfile is not None:
            fname = os.path.join(self.filedir, testfile)
            if (self.policy.normalize == 0):
                bestgeno = np.load(fname)
            else:
                geno = np.load(fname)
                for i in range(self.policy.ninputs * 2):
                    self.policy.normvector[i] = geno[self.policy.nparams + i]
                bestgeno = geno[0:self.policy.nparams]
                self.policy.nn.setNormalizationVectors()
        else:
            print("Cannot run functional network test without passing a genotype file!!!")
            sys.exit()
        if (self.policy.nttrials > 0):
            ntrials = self.policy.nttrials
        else:
            ntrials = self.policy.ntrials
        # Output file
        outfile = os.path.join(self.filedir, "funcTestS" + str(self.policy.get_seed) + ".txt")
        fp = open(outfile, "w")
        # Test the original genotype first
        self.policy.set_trainable_flat(bestgeno)
        fp.write("-1\t")
        avgf = 0.0
        for t in range(ntrials):
            eval_rews, eval_length = self.policy.rollout(1, seed=self.policy.get_seed + 100000 + t)
            fp.write("%lf\t" % eval_rews)
            avgf += eval_rews
        avgf /= float(ntrials)
        fp.write("\n")
        print("Postevauation (original genotype): Average Fitness %.2f" % avgf)
        # Get number of inputs, hiddens and outputs of the network
        ninputs = self.policy.ninputs
        nhiddens = self.policy.nhiddens
        noutputs = self.policy.noutputs
        # Now test the functional network
        for h in range(nhiddens):
            geno = copy.deepcopy(bestgeno)
            # Set bias to 0
            geno[h] = 0.0
            # We need to set all the incoming and outgoing weights to 0
            for i in range(ninputs):
                geno[nhiddens + noutputs + (ninputs * h) + i] = 0.0
            for i in range(noutputs):
                geno[nhiddens + noutputs + (ninputs * nhiddens) + (nhiddens * i)] = 0.0
            self.policy.set_trainable_flat(bestgeno) 
            fp.write("%d\t" % h)
            avgf = 0.0
            for t in range(ntrials):
                eval_rews, eval_length = self.policy.rollout(1, seed=self.policy.get_seed + 100000 + t)
                fp.write("%lf\t" % eval_rews)
                avgf += eval_rews
            avgf /= float(ntrials)
            print("Postevauation (removed hidden %d): Average Fitness %.2f" % (h, avgf))
            fp.write("\n")
        fp.close()
