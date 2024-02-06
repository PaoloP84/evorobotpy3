#!/usr/bin/python
# -*- coding: utf-8 -*-

# archivestar1 (single pop version)
"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   coevo2.py include an implementation of an competitive co-evolutionary algorithm analogous
   to that described in:
   Simione L and Nolfi S. (2019). Long-Term Progress and Behavior Complexification in Competitive Co-Evolution, arXiv:1909.08303.

   Requires es.py policy.py and evoalgo.py
   Also requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py  
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext inplace; cp net*.so ../bin
"""


import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort
import random
import os
import sys
import configparser

# competitive coevolutionary algorithm operating on two populations
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)
        self.ncompetitors = 10

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.popsize = 1
            self.ncompetitors = 10
            self.ngenerations = 0
            self.stepsize = 0.01
            self.batchSize = 20
            self.noiseStdDev = 0.02
            self.wdecay = 0
            self.saveeach = 100
            self.checkoverfitevery = 0
            self.preferential = 1
            self.seldifferentiated = 0
            self.maxsteps = 0
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "ngenerations":
                    self.ngenerations = config.getint("ALGO","ngenerations")
                    found = 1
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "ncompetitors":
                    self.ncompetitors = config.getint("ALGO","ncompetitors")
                    found = 1
                if o == "stepsize":
                    self.stepsize = config.getfloat("ALGO","stepsize")
                    found = 1
                if o == "noisestddev":
                    self.noiseStdDev = config.getfloat("ALGO","noiseStdDev")
                    found = 1
                if o == "samplesize":
                    self.batchSize = config.getint("ALGO","samplesize")
                    found = 1
                if o == "wdecay":
                    self.wdecay = config.getint("ALGO","wdecay")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1
                if o == "addtoarchiveevery":
                    self.addtoarchiveevery = config.getint("ALGO","addtoarchiveevery")
                    found = 1
                if o == "popsize":
                    self.popsize = config.getint("ALGO","popsize")
                    found = 1
                if o == "checkoverfiteveryn":
                    self.checkoverfiteveryn = config.getint("ALGO","checkoverfiteveryn")
                    print("warning: checkoverfit not implemented")
                    found = 1
                if o == "preferential":
                    self.preferential = config.getint("ALGO","preferential")
                    found = 1
                if o == "seldifferentiated":
                    self.seldifferentiated = config.getint("ALGO","seldifferentiated")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("ngenerations [integer]    : max number of generations (default 200)")
                    print("ncompetitors [integer]    : number of competitors (default 10)")
                    print("stepsize [float]          : learning stepsize (default 0.01)")
                    print("samplesize [int]          : samplesize/2 (default 20)")
                    print("noiseStdDev [float]       : samples noise (default 0.02)")
                    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
                    print("saveeach [integer]        : save file every N generations (default 100)")
                    print("addtoarchiveeach [integer]: add the best to the archive every n generations (default 1)")
                    print("popsize [integer]         : number of evolving individuals (default 1)")
                    print("checkoverfiteveryn [int.] : check overfit every n generations (default 0 = never)")
                    print("preferential [int.]       : select the hardest competitors among n samples (default 1 = none)")
                    print("seldifferentiated [int.]  : select competitors among different popopulations (default 0 = none)") 
                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))
   

       

    def run(self):

        self.loadhyperparameters()           # load hyperparameters

        seed = self.seed
        self.rs = np.random.RandomState(self.seed)
        self.steps = 0
        self.gen = 0

        # Extract the number of parameters
        self.nparams = int(self.policy.nparams / 2)                                   # parameters required for a single individual
       
        self.candidate = np.arange(self.nparams, dtype=np.float64)                    # the vector used to store offspring      
       
        # initialize the population (we keep pred, but it is non meaningful)
        # Predators
        self.predpop = []                                                            # the population 
        self.predpopm = []                                                           # the momentum of the populations
        self.predpopv = []                                                           # the squared momentum of the populations
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            randomparams = np.copy(self.policy.get_trainable_flat())
            self.predpop.append(randomparams[:self.nparams])
            self.predpopm.append(zeros(self.nparams))
            self.predpopv.append(zeros(self.nparams))
        self.predpop = np.asarray(self.predpop)
        self.predpopm = np.asarray(self.predpopm)
        self.predpopv = np.asarray(self.predpopv)
        # Preys
        self.preypop = []                                                            # the population 
        self.preypopm = []                                                           # the momentum of the populations
        self.preypopv = []                                                           # the squared momentum of the populations
        for i in range(self.popsize):
            self.policy.nn.initWeights()
            randomparams = np.copy(self.policy.get_trainable_flat())
            self.preypop.append(randomparams[:self.nparams])
            self.preypopm.append(zeros(self.nparams))
            self.preypopv.append(zeros(self.nparams))
        self.preypop = np.asarray(self.preypop)
        self.preypopm = np.asarray(self.preypopm)
        self.preypopv = np.asarray(self.preypopv)

        # initialize the archive
        # Predators
        self.predarchive = [] #(we keep prey, but it is non meaningfull)
        # The archives are initialized with randomly generated competitors 
        for i in range(self.ncompetitors):
            self.policy.nn.initWeights()
            randomparams = np.copy(self.policy.get_trainable_flat())
            self.predarchive.append(randomparams[:self.nparams])
        self.predarchive = np.asarray(self.predarchive)
        # initialize the performance vector for the archives. Not yet evaluated competitors start from 0.5
        self.predarchivep = np.zeros(self.ncompetitors)
        self.predarchivep.fill(0.5)
        # Preys
        self.preyarchive = [] #(we keep prey, but it is non meaningfull)
        # The archives are initialized with randomly generated competitors 
        for i in range(self.ncompetitors):
            self.policy.nn.initWeights()
            randomparams = np.copy(self.policy.get_trainable_flat())
            self.preyarchive.append(randomparams[:self.nparams])
        self.preyarchive = np.asarray(self.preyarchive)
        # initialize the performance vector for the archives. Not yet evaluated competitors start from 0.5
        self.preyarchivep = np.zeros(self.ncompetitors)
        self.preyarchivep.fill(0.5)

        # Initialize the bests
        self.predbest = []
        self.preybest = []
        # Flags the initialization of bests
        self.predbestinit = False
        self.preybestinit = False
        # Initialize the list of competitors for overfit checking
        self.predcompcheck = []
        self.preycompcheck = []

        print("Coevo-archive1 seed %d popsize %d competitors %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d nparams %d" % (self.seed, self.popsize, self.ncompetitors, self.batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.nparams))
        
        # Save initial populations (centroids)
        filename = "S%dG0Pred.npy" % (seed)
        np.save(filename, os.path.join(self.filedir, self.predpop))
        # Save archives
        filename = "S%dG0PredArchive.npy" % (seed)
        np.save(filename, os.path.join(self.filedir, self.predarchive))
        filename = "S%dG0Prey.npy" % (seed)
        np.save(filename, os.path.join(self.filedir, self.preypop))
        # Save archives
        filename = "S%dG0PreyArchive.npy" % (seed)
        np.save(filename, os.path.join(self.filedir, self.preyarchive))

        # Flags whether we are evolving predators (flag set to False) or preys (flag set to True).
        # When preys are evolved, fitness is inverted (i.e., f_new = 1.0 - f)
        avefitpred = 0
        bestfitpred = 0
        avefitprey = 0
        bestfitprey = 0
        
        # main loop
        #for self.gen in range(self.ngenerations):
        while ((self.ngenerations == 0 or self.gen < self.ngenerations) and (self.maxsteps == 0 or self.steps < self.maxsteps)):

            # Check overfitting and eventually restore previous generations
            #if (self.checkoverfiteveryn > 0 and self.gen > 0 and (self.gen % self.checkoverfiteveryn) == 0):
            #    self.checkoverfitting()
                                                                                           
            # evolve the centroids for one generation
            for sind in range(self.popsize):
                avefitpred, bestfitpred, = self.runphase(sind, self.nparams)                
            print("seed %d gen %d msteps %.0f ind %d avefit %.2f bestfit %.2f archive %d weight %.2f" % (seed, self.gen, self.steps / 1000000, sind, avefitpred, bestfitpred, np.shape(self.predarchive)[0], np.average(np.absolute(self.predpop[sind]))))
            # print and save statistics
            self.stat = np.append(self.stat, [0, avefitpred, bestfitpred, 0.0, 0.0, 0])  # store performance across generations

            # save the archive and the statistics
            if (((self.gen + 1) % self.saveeach) == 0):
                # Save archives (predators and preys)
                # Predators
                filename = "S%dPredArchive.npy" % (seed)
                np.save(filename, os.path.join(self.filedir, self.predarchive))
                # Preys
                filename = "S%dPreyArchive.npy" % (seed)
                np.save(filename, os.path.join(self.filedir, self.preyarchive))
                # Save stats
                fname = os.path.join(self.filedir, "statS" + str(self.seed))
                np.save(fname, self.stat)
                for i in self.predarchivep:
                    print("%.2f " % (i), end="")
                print("")
            # increment generation counter    
            self.gen += 1

        # Save all files at the end of the evolutionary process
        # Archives (predators and preys)
        # Predators
        filename = "S%dPredArchive.npy" % (seed)
        np.save(filename, os.path.join(self.filedir, self.predarchive))
        # Centroid and momentum vectors (m and v) for predators)
        filename = "S%dG%dPred.npy" % (seed, self.gen + 1)
        np.save(filename, os.path.join(self.filedir, self.predpop))
        filename = "S%dG%dPredm.npy" % (seed, self.gen + 1)
        np.save(filename, os.path.join(self.filedir, self.predpopm))
        filename = "S%dG%dPredv.npy" % (seed, self.gen + 1)
        np.save(filename, os.path.join(self.filedir, self.predpopv))
        # Preys
        filename = "S%dPreyArchive.npy" % (seed)
        np.save(filename, os.path.join(self.filedir, self.preyarchive))
        # Centroid and momentum vectors (m and v) for predators)
        filename = "S%dG%dPrey.npy" % (seed, self.gen + 1)
        np.save(filename, os.path.join(self.filedir, self.preypop))
        filename = "S%dG%dPreym.npy" % (seed, self.gen + 1)
        np.save(filename, os.path.join(self.filedir, self.preypopm))
        filename = "S%dG%dPreyv.npy" % (seed, self.gen + 1)
        np.save(filename, os.path.join(self.filedir, self.preypopv))

    # Check overfitting and eventually restore previous generations (use evaluate() and evaladditional())
    """
    def checkoverfitting(self):
        if (len(self.predcompcheck) == 0): # we evaluate for the first time
            if (np.shape(self.predarchive)[0] > 100 and np.shape(self.preyarchive)[0] > 100):               
                self.predcompcheck  = random.sample(range(np.shape(self.preyarchive)[0]), 100)
                self.preycompcheck  = random.sample(range(np.shape(self.predarchive)[0]), 100)
                self.predfitold, self.preyfitold = self.evaluate()
                #print(self.predcompcheck)
                #print(self.predfitold)
                #print(self.preycompcheck)
                #print(self.preyfitold)
                self.predpopold = np.copy(self.predpop)
                self.predpopmold = np.copy(self.predpopm)
                self.predpopvold = np.copy(self.predpopv)
                self.preypopold = np.copy(self.preypop)
                self.preypopmold = np.copy(self.preypopm)
                self.preypopvold = np.copy(self.preypopv) 
        else:
            self.predfit, self.preyfit = self.evaluate()
            #print(self.predcompcheck)
            #print(self.predfitold)
            #print(self.preyfit)
            #print(self.preycompcheck)
            #print(self.preyfitold)
            #print(self.preyfit)
            print("Global progress (pred - prey) ", end = "")
            for i in range (self.popsize):
                print("(pop%d) " % (i), end="")
                predf = np.sum(self.predfit[i] / len(self.predcompcheck))
                preyf = np.sum(self.preyfit[i] / len(self.preycompcheck))
                predfold = np.sum(self.predfitold[i] / len(self.predcompcheck))
                preyfold = np.sum(self.preyfitold[i] / len(self.preycompcheck))                        
                if (predf < predfold):
                    for p in range(self.nparams):
                        self.predpop[i][p] = self.predpopold[i][p]
                        self.predpopm[i][p] = self.predpopmold[i][p]
                        self.predpopv[i][p] = self.predpopvold[i][p]
                print("%f " % (predf - predfold), end="")
                if (preyf < preyfold):
                    for p in range(self.nparams):
                        self.preypop[i][p] = self.preypopold[i][p]
                        self.preypopm[i][p] = self.preypopmold[i][p]
                        self.preypopv[i][p] = self.preypopvold[i][p]
                print("%f " % (preyf - preyfold), end = "")
            print("")
            self.predfitold, self.preyfitold = self.evaladditional(self.predfit, self.preyfit)
            #print(self.predcompcheck)
            #print(predfitold)
            #print(self.preycompcheck)
            #print(preyfitold)
    """

    # re-evaluate the evolving individuals agaist a set of competitors to check global progress 
    def evaluate(self):
        predfit = np.zeros((self.popsize, len(self.predcompcheck)))
        for i in range(self.popsize):
            cc = 0
            for c in self.predcompcheck:
                self.policy.set_trainable_flat(np.concatenate((self.predpop[i], self.preyarchive[c])))
                predfit[i][cc], eval_length = self.policy.rollout(1, seed=(self.seed+10000))
                self.steps += eval_length
                cc += 1
        preyfit = np.zeros((self.popsize, len(self.preycompcheck)))
        for i in range(self.popsize):
            cc = 0
            for c in self.preycompcheck:
                self.policy.set_trainable_flat(np.concatenate((self.predarchive[c], self.preypop[i])))
                eval_rew, eval_length = self.policy.rollout(1, seed=(self.seed+10000))
                self.steps += eval_length
                preyfit[i][cc] = 1.0 - eval_rew
                cc += 1
        return predfit, preyfit

    # continue the evaluate by adding 10 new competitors
    # remove the first 10 competitors from the actual data
    def evaladditional(self, predfit, preyfit):
        #shift data back for the last popsize-10 columns
        for i in range(self.popsize):
            for c in range(self.popsize-10):
                predfit[i][c] = predfit[i][c+10]
                preyfit[i][c] = preyfit[i][c+10]
                self.predcompcheck[c] = self.predcompcheck[c+10]
                self.preycompcheck[c] = self.preycompcheck[c+10]
        #select 10 new competitors
        c1 = len(self.predcompcheck) - 10
        c2 = len(self.predcompcheck) - 10 
        for c in range(10):
            cc = len(self.predcompcheck) - 10 
            self.predcompcheck[c1 + c] = random.randrange(0,np.shape(self.preyarchive)[0])
            self.preycompcheck[c2 + c] = random.randrange(0,np.shape(self.preyarchive)[0]) 
        #evaluate against the new 10 competitors
        size = np.shape(predfit)[0]
        for i in range(self.popsize):
            for c in range(10):
                cc = self.predcompcheck[size+c-10]
                self.policy.set_trainable_flat(np.concatenate((self.predpop[i], self.preyarchive[cc])))
                predfit[i][size+c-10], eval_length = self.policy.rollout(1, seed=(self.seed+10000))
                self.steps += eval_length
        for i in range(self.popsize):
            for c in range(10):
                cc = self.preycompcheck[size+c-10]
                self.policy.set_trainable_flat(np.concatenate((self.predarchive[cc], self.preypop[i])))
                eval_rew, eval_length = self.policy.rollout(1, seed=(self.seed+10000))
                self.steps += eval_length
                preyfit[i][size+c-10] = 1.0 - eval_rew
        return predfit, preyfit                 

    # select and return a list of competitors
    # if prefertial > 1, select the strongest among multiple samples 
    def selcompetitors(self):
        competitors = np.zeros(self.ncompetitors, dtype=np.int)
        if (self.seldifferentiated == 1):
            for c in range(self.ncompetitors):
                competitors[c] = (random.randrange(0,int(np.shape(self.predarchive)[0]/self.popsize)) * self.popsize) + (c % self.popsize)
        else:
            for c in range(self.ncompetitors):
                cstrength = -999
                for t in range(self.preferential):
                    sc = random.randrange(0,np.shape(self.predarchive)[0])
                    if (self.predarchivep[sc] > cstrength):
                        selc = sc
                        cstrength = self.predarchivep[sc]                     
                competitors[c] = selc
        return competitors

           
    # performs a generation  
    def runphase(self, sind, nparams):
        # Initialize the Adam stochastic optimizer 
        epsilon = 1e-08
        beta1 = 0.9
        beta2 = 0.999
        # Weight initialization
        weights = zeros(self.batchSize)
        # Current generation best (fitness and the corresponding individual)
        cbest = -99999.0
        cbestind = None
        for it in range (1):
            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = self.rs.randn(self.batchSize, nparams)
            fitness = zeros(self.batchSize * 2)
            for b in range(self.batchSize):
                # Symmetric samples are evaluated against the same competitors
                self.competitors = self.selcompetitors()
                for bb in range(2):
                    if (bb == 0):
                        # Positive variation
                        for g in range(nparams):
                            self.candidate[g] = self.predpop[sind][g] + samples[b,g] * self.noiseStdDev
                    else:
                        # Negative variation
                        for g in range(nparams):
                            self.candidate[g] = self.predpop[sind][g] - samples[b,g] * self.noiseStdDev
                    # Evaluate against competitors
                    ave_rews = 0
                    for c in range(self.ncompetitors):
                        self.policy.set_trainable_flat(np.concatenate((self.candidate, self.predarchive[self.competitors[c]])))
                        eval_rews, eval_length = self.policy.rollout(1, seed=(self.seed+self.gen*1000+it*100+b))
                        self.predarchivep[self.competitors[c]] = (self.predarchivep[self.competitors[c]] * 0.9) + ((1.0 - eval_rews) * 0.1)
                        self.steps += eval_length
                        ave_rews += eval_rews
                    fitness[b*2+bb] = ave_rews / float(self.ncompetitors)
                    # Check whether or not the offspring is better than current best
                    if fitness[b*2+bb] > cbest:
                        # Found a new best, update it
                        cbest = fitness[b*2+bb]
                        cbestind = np.copy(self.candidate)

            # Sort by fitness and compute weighted mean into center
            fitness, index = ascendent_sort(fitness)
            # Now me must compute the symmetric weights in the range [-0.5,0.5]
            utilities = zeros(self.batchSize * 2)
            for i in range(self.batchSize * 2):
                utilities[index[i]] = i
            utilities /= (self.batchSize * 2 - 1)
            utilities -= 0.5
            # Now we assign the weights to the samples
            for i in range(self.batchSize):
                idx = 2 * i
                weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg

            # Compute the gradient
            g = 0.0
            i = 0
            while i < self.batchSize:
                gsize = -1
                if self.batchSize - i < 500:
                    gsize = self.batchSize - i
                else:
                    gsize = 500
                g += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                i += gsize
            # Normalization over the number of samples
            g /= (self.batchSize * 2)
            # Weight decay
            if (self.wdecay == 1):
                globalg = -g + 0.005 * self.predpop[sind]
            else:
                globalg = -g
            # ADAM stochastic optimizer
            # a = self.stepsize * sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
            a = self.stepsize # bias correction is not implemented
            self.predpopm[sind] = beta1 * self.predpopm[sind] + (1.0 - beta1) * globalg
            self.predpopv[sind] = beta2 * self.predpopv[sind] + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * self.predpopm[sind] / (sqrt(self.predpopv[sind]) + epsilon)
            # update center
            self.predpop[sind] += dCenter

        #print(cbest)
        # We store sample best fitness
        sbest = cbest

        # store the best new individual in the archive
        if ((self.gen % self.addtoarchiveevery) == 0):
            self.predarchive = np.vstack((self.predarchive, cbestind))
            self.predarchivep = np.append(self.predarchivep, 0.5)

        return ave_rews / (self.batchSize * 2.0), cbest

    def testusage(self):
        print("ERROR, To post-evaluate with the coevo algorithm you should specify with the -g parameter a string containing:")
        print("P-g-gg-n : Postevaluates the best n from generation g against the best n from generation gg by showing the behavior")
        print("p-g-gg   : Postevaluates the centroid of generation g against the centroid of generation gg withou showing the behavior")        
        print("M-g-gg-ni: Creates a master.npy matrix by post-evaluating performance up to generation g every gg generations by averaging ni*ni consecutive individuals")
        print("         :   Also extract the strongest individuals against each postevaluated individual")              
        print("S-g-gg   : Generates an AgainstStrongest matrix by postevaluate individuals up to generation g, every gg individuals, against the strongests")
        print("X-gg-g   : Postevaluates up to geration g every gg individuals agaist pre-trained opponent")
        print("C-dir1-dir2-nseeds-ninds: Crosstest experiments of dir1 agaist dir2 (nseeds: number of seeds, ninds: number of individuals tested")
        print("c-dir1-dir2-nseeds: Crosstest experiments best of dir1 agaist best of dir2")
        print("K-nseeds-n: Extract the best individual among the n last individuals on the basis of the performance obtained agaist 100 competitors of different generations")
        
        sys.exit()

    def test(self, testparam):
        if testparam is None:
            self.testusage()         
        seed = self.seed
        if (len(testparam) > 1):
            parsen = testparam.split("-")
        else:
            parsen = [testparam]
        if (not parsen[0] in ["P", "p", "M", "m", "S", "C", "c", "X", "B", "K"]):
            self.testusage()
        # P-g-gg: Postevaluate the centroid of generation g against the competitor of generation gg
        # P renders behavior, "p" only print fitness            
        if (parsen[0] == "p" or parsen[0] == "P"):
            if (parsen[0] == "P"):
                self.policy.test = 1
                rendt = True
            else:
                self.policy.test = 0
                rendt = False
            popfile = "S%dPredArchive.npy" % (seed)
            predpop = np.load(popfile)
            popshape = predpop.shape
            popsize1 = popshape[0]
            popfile = "S%dPreyArchive.npy" % (seed)
            preypop = np.load(popfile)
            popshape = preypop.shape
            popsize2 = popshape[0]
            print("archives contains %d %d individuals " % (popsize1, popsize2))
            
            g = int(parsen[1])
            gg = int(parsen[2])
            ninds = 5
            if (len(parsen) >= 4): 
                ninds = int(parsen[3])
            if (g > popsize1):
                g = popsize1 - ninds
            if (gg > popsize2):
                gg = popsize2 - ninds
            tot_rew = 0
            print("seed %d: Postevaluate gen %d against gen %d (%d * %d individuals)" % (seed, g, gg, ninds, ninds))
            for i in range(ninds):
                for ii in range(ninds):
                    self.policy.set_trainable_flat(np.concatenate((predpop[g+ii], preypop[gg+i])))
                    eval_rews, eval_length = self.policy.rollout(1)
                    tot_rew += eval_rews
                    print("fitness %d against %d: %.2f " % (g+ii, gg+i, eval_rews))

        # "M-n1-n2, Master tournament, test all generations (up to n1) against every generations, every n2 generations
        if (parsen[0] == "M"):
            gpop1 = 0
            gpop2 = 0
            popfile = "S%dPredArchive.npy" % (seed)
            preypop = np.load(popfile)
            preypop = preypop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = preypop.shape
            popsize = popshape[0]
            print("Size of the archive %d" % (popsize))
            if len(parsen) == 2:
                subpop = int(parsen[1])
                assert subpop >= 0 and subpop <= 1000, "Invalid subpop value %d".format(subpop)
                fraction = subpop / 1000
                psize = fraction * popsize
                popsize = int(psize)
            everygen = int(popsize / 10)
            if (everygen < 1):
                everygen = 1
            uptogen = int(10 * everygen)
            ninds = 40
            if (len(parsen) >= 2): 
                uptogen = int(parsen[1])                # up to generation N
            if (len(parsen) >= 3): 
                everygen = int(parsen[2])               # every N generation
            if (len(parsen) >= 4): 
                ninds = int(parsen[3]) 
            print("Size of the archive %d" % (popsize))
            numgen = int((uptogen / everygen))         # number of postevaluation test
            self.policy.test = 0                       # do not shows the behavior
            print("seed %d: postevaluation all generations (up to %d) against all generations every %d generations by averaging %d*%d consecutive individuals" % (seed, uptogen, everygen, ninds, ninds))
            master = np.zeros((numgen, numgen), dtype=np.float64) # matrix with the average performance of every generation against every other generation
            for pn1 in range(numgen):
                for pn2 in range(numgen):
                    rews = 0
                    for i1 in range(ninds):
                        for i2 in range(ninds):
                            self.policy.set_trainable_flat(np.concatenate((preypop[pn1*everygen], preypop[pn2*everygen])))                              
                            rew, eval_length = self.policy.rollout(1)
                            rews += rew
                    master[pn1, pn2] = rews / (ninds * ninds)
            mfile = "masterS%d.npy" % (seed)
            np.save(mfile, master)
            """
            # extract the strongest individuals against each individual
            # the strongest against the best, the second best, the third best and so on
            # the number of individuals extracted is variable since an individual can be the stongest against several individuals
            rowsum = np.sum(master, axis=1)
            print(master)
            print(rowsum)
            rowsum2, index = ascendent_sort(rowsum)
            # Predators
            predstrongest = np.empty(len(predpop[0]))
            pred_strongest_list = []
            for s in range(numgen):
                maxf = -99999
                maxi = 0
                i = numgen - 1
                while (i >= 0):
                    if master[index[i], s] > maxf:
                        maxf = master[index[i], s]
                        maxi = index[i]
                    i -= 1
                print("best predator against prey %d: extracted %d maxf %.2f " % (s, maxi, maxf))
                if maxi not in pred_strongest_list:
                    pred_strongest_list.append(maxi)
                    predstrongest = np.vstack((predstrongest, predpop[maxi]))
            print(pred_strongest_list)
            filename = "S%dPredStrongest.npy" % (seed)
            np.save(filename, predstrongest)
            # Preys
            preystrongest = np.empty(len(preypop[0]))
            prey_strongest_list = []
            colsum = np.sum(master, axis=1)
            colsum2, index = ascendent_sort(colsum)
            for s in range(numgen):
                maxf = -99999
                maxi = 0
                i = numgen - 1
                while (i >= 0):
                    if (1.0 - master[s, index[i]]) > maxf:
                        maxf = (1.0 - master[s, index[i]])
                        maxi = index[i]
                    i -= 1
                print("best prey against predator %d: extracted %d maxf %.2f " % (s, maxi, maxf))
                if maxi not in prey_strongest_list:
                    prey_strongest_list.append(maxi)
                    preystrongest = np.vstack((preystrongest, preypop[maxi]))
            print(prey_strongest_list)
            filename = "S%dPreyStrongest.npy" % (seed)
            np.save(filename, preystrongest)
            """
        # "m-n1-n2-n3 Last n3 individuals of generation n1 tested against previous individuals every n2 generations
        if (parsen[0] == "m"):        
            popfile = "S%dPreyArchive.npy" % (seed)
            preypop = np.load(popfile)
            preypop = preypop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = preypop.shape
            popsize = popshape[0]
            ninds = 80
            lastgen = popsize1 - ninds
            everygen = int(popsize / 10)
            if (len(parsen) >= 2): 
                lastgen = int(parsen[1])                # up to generation N
            if (len(parsen) >= 3): 
                everygen = int(parsen[2])               # every N generation
            if (len(parsen) >= 4): 
                ninds = int(parsen[3])                  # number of successive individuals to be postevaluated
            numgen = int(lastgen / everygen) + 1
            print("Size of the archives %d " % (popsize1))
            self.policy.test = 0                       # do not shows the behavior
            print("seed %d: postevaluate gen %d agaist older competitors every %d by averaging %d consecutive individuals" % (seed, lastgen, everygen, ninds))
            master = np.zeros((numgen, numgen), dtype=np.float64) # matrix with the average performance of every generation against every other generation
            print("predators : ", end="")
            for g in range(numgen):
                rews = 0.0
                for i1 in range(ninds):
                    for i2 in range(ninds):
                        if ((numgen - g - 1) == 0):
                            self.policy.set_trainable_flat(np.concatenate((preypop[lastgen+i1-ninds], preypop[(numgen-g-1)*everygen+i2])))
                        else:
                            self.policy.set_trainable_flat(np.concatenate((preypop[lastgen+i1-ninds], preypop[(numgen-g-1)*everygen+i2-ninds])))
                        rew, eval_length = self.policy.rollout(1)
                        rews += rew
                print("%.2f " % (rews / (ninds * ninds)), end="")
            print("")
        # "S-n1-n2, Set tournament, test all generations (up to n1) against Strongest competitors
        if (parsen[0] == "S"):
            gpop1 = 0
            gpop2 = 0
            uptogen = int(parsen[1])                # up to generation N
            everygen = int(parsen[2])               # every N generation
            popfile = "S%dArchive.npy" % (seed)
            pop = np.load(popfile)
            popshape = pop.shape
            popsize = int(popshape[0])
            numgen = int((uptogen / everygen) + 1)   # number of generations
            self.policy.test = 0                     # do not shows the behavior
            popfile = "S%dStrongest.npy" % (seed)
            pop2 = np.load(popfile)
            print("seed %d: postevaluation all generations (up to %d) against all generations every %d generations" % (seed, uptogen, everygen))
            master = np.zeros((numgen, len(pop2)), dtype=np.float64)
            for pn1 in range(numgen):
                for pn2 in range(len(pop2)):
                    self.policy.set_trainable_flat(np.concatenate((pop[pn1*everygen], pop2[pn2])))                              
                    rew, eval_length = self.policy.rollout(1)
                    master[pn1, pn2] = rew
            mfile = "againstStrongestS%d.npy" % (seed)
            np.save(mfile, master)            
           
        # "X-n1-n2, Test evolved agents of different generations against a pre-trained opponent
        if (parsen[0] == "X"):
            cgen = 0
            self.policy.test = 0
            print("seed %d: postevaluation every %d generations up to generation %d against a pre-trained opponent" % (seed, int(parsen[1]), int(parsen[2])))
            popfile = "S%dPredArchive.npy" % (seed)
            predpop = np.load(popfile)
            predpop = predpop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = predpop.shape
            popsize = popshape[0]
            popfile = "S%dPreyArchive.npy" % (seed)
            preypop = np.load(popfile)
            preypop = preypop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = preypop.shape
            popsize = popshape[0]
            g = int(parsen[1])
            gg = int(parsen[2])
            nphases = int(gg / g)# + 1
            #print(nphases)
            #sys.exit()
            for i in range (nphases):
                tot_rew = 0
                max_ind_rew = -99999
                for ii in range(g):
                    ind_rew = 0
                    """
                    if i == nphases - 1:
                        print(i, gg - 1, i*g+ii)
                    """
                    self.policy.set_trainable_flat(np.concatenate((predpop[gg - 1], preypop[i*g+ii])))                              
                    rew, eval_length = self.policy.rollout(1)
                    if (rew > max_ind_rew):
                        max_ind_rew = rew
                    tot_rew += rew
                    #print("pred - gen %d-%d fit %.2f" % (gg, i*g+ii, rew))
                print("pred - gen %d-%d ave %.2f max %.2f" % (gg, i*g, (tot_rew / g), max_ind_rew))
            for i in range (nphases):
                tot_rew = 0
                max_ind_rew = -99999
                for ii in range(g):
                    ind_rew = 0
                    self.policy.set_trainable_flat(np.concatenate((predpop[i*g+ii], preypop[gg - 1])))                              
                    rew, eval_length = self.policy.rollout(1)
                    rew = (1.0 - rew)
                    if (rew > max_ind_rew):
                        max_ind_rew = rew
                    tot_rew += rew
                    #print("prey - gen %d-%d fit %.2f" % (i*g+ii, gg, rew))
                print("prey - gen %d-%d ave %.2f max %.2f" % (gg, i*g, (tot_rew / g), max_ind_rew))
        # "B-n1-n2, Test evolved agents of different generations against a pre-trained opponent
        if (parsen[0] == "B"):
            cgen = 0
            self.policy.test = 0
            print("seed %d: postevaluation every %d generations up to generation %d against a pre-trained opponent" % (seed, int(parsen[1]), int(parsen[2])))
            popfile = "S%dPredBest.npy" % (seed)
            predpop = np.load(popfile)
            predpop = predpop[self.ncompetitors:,:]
            popshape = predpop.shape
            popsize = popshape[0]
            popfile = "S%dPreyArchive.npy" % (seed)
            preypop = np.load(popfile)
            preypop = preypop[self.ncompetitors:,:]
            popshape = preypop.shape
            popsize = popshape[0]
            g = int(parsen[1])
            gg = int(parsen[2])
            nphases = int(gg / g)
            for i in range (nphases):
                tot_rew = 0
                max_ind_rew = -99999
                for ii in range(g):
                    ind_rew = 0
                    self.policy.set_trainable_flat(np.concatenate((predpop[gg - 1], preypop[i*g+ii])))                              
                    rew, eval_length = self.policy.rollout(1)
                    if (rew > max_ind_rew):
                        max_ind_rew = rew
                    tot_rew += rew
                    #print("pred - gen %d-%d fit %.2f" % (gg, i*g+ii, rew))
                print("pred - gen %d-%d ave %.2f max %.2f" % (gg, i*g, (tot_rew / g), max_ind_rew))
            for i in range (nphases):
                tot_rew = 0
                max_ind_rew = -99999
                for ii in range(g):
                    ind_rew = 0
                    self.policy.set_trainable_flat(np.concatenate((predpop[i*g+ii], preypop[gg - 1])))                              
                    rew, eval_length = self.policy.rollout(1)
                    rew = (1.0 - rew)
                    if (rew > max_ind_rew):
                        max_ind_rew = rew
                    tot_rew += rew
                    #print("prey - gen %d-%d fit %.2f" % (i*g+ii, gg, rew))
                print("prey - gen %d-%d ave %.2f max %.2f" % (gg, i*g, (tot_rew / g), max_ind_rew))
        # "C-dir1-dir2-seeds-ninds, postevaluate the last 20 individuals of file1 against the last 20 individuals of file2
        if (parsen[0] == "C"):
            self.policy.test = 0
            dir1 = parsen[1]
            dir2 = parsen[2]
            if (len(parsen) >= 4): 
                nseeds = int(parsen[3])
            else:
                nseeds = 1
            if (len(parsen) == 5): 
                ninds = int(parsen[4])
            else:
                ninds = 50                
            print("Crosstest experiments of dir %s against dir %s (%d seeds, %d individuals) " % (parsen[1], parsen[2], nseeds, ninds))
            tot_seeds = [0,0]
            crossmatrix1 = np.zeros((nseeds, nseeds), dtype=np.float64)
            for s1 in range(nseeds):
                for s2 in range(nseeds):
                    tot_seed = [0,0]
                    for t in range(2):  #dir1 against dir1 and dir1 against dir 2
                        pop1file = "%sS%dPreyArchive.npy" % (dir1, s1 + 1)
                        pop1 = np.load(pop1file)
                        popshape1 = pop1.shape
                        popsize1 = int(popshape1[0])
                        if (t == 0):
                            pop2file = "%sS%dPreyArchive.npy" % (dir1, s2 + 1)
                        else:
                            pop2file = "%sS%dPreyArchive.npy" % (dir2, s2 + 1)
                        pop2 = np.load(pop2file)
                        popshape2 = pop2.shape
                        popsize2 = int(popshape2[0])
                        for i1 in range(ninds):
                            for i2 in range(ninds):
                                self.policy.set_trainable_flat(np.concatenate((pop1[popsize1-i1-1], pop2[popsize2-i2-1])))                              
                                rew, eval_length = self.policy.rollout(1)
                                tot_seed[t] += rew
                                tot_seeds[t] += rew
                        tot_seed[t] /= ninds*ninds
                    print("seeds %d-%d Fitness gain with different competitors: %.2f " % (s1 + 1, s2 + 1, tot_seed[1] - tot_seed[0]), flush=True)
                    crossmatrix1[s1][s2] = (tot_seed[1] - tot_seed[0])
            tot_seeds[0] /= nseeds*nseeds*ninds*ninds
            tot_seeds[1] /= nseeds*nseeds*ninds*ninds            
            print("Fitness gain with different competitors %.2f" % (tot_seeds[1] - tot_seeds[0]), flush=True)
            cfile = "crosstest1.npy" 
            np.save(cfile, crossmatrix1)
        # "c-dir1-dir2-seeds, postevaluate the best individuals of dir1 against the best individuals of dir2
        if (parsen[0] == "c"):
            self.policy.test = 0
            dir1 = parsen[1]
            dir2 = parsen[2]
            if (len(parsen) >= 4): 
                nseeds = int(parsen[3])
            else:
                nseeds = 1               
            print("Crosstest experiments best individuals of dir %s against dir %s (%d seeds) " % (parsen[1], parsen[2], nseeds))
            averagedif = 0.0
            crossmatrix1 = np.zeros((nseeds, nseeds), dtype=np.float64)
            for s1 in range(nseeds):
                for s2 in range(nseeds):
                    for t in range(2):  #dir1 against dir1 and dir1 against dir 2
                        pop1file = "%sS%dBestPop.npy" % (dir1, s1 + 1)
                        pop1 = np.load(pop1file)
                        if (t == 0):
                            pop2file = "%sS%dBestPop.npy" % (dir1, s2 + 1)
                        else:
                            pop2file = "%sS%dBestPop.npy" % (dir2, s2 + 1)
                        pop2 = np.load(pop2file)
                        self.policy.set_trainable_flat(np.concatenate((pop1, pop2)))                              
                        rew, eval_length = self.policy.rollout(1)
                        if t == 0:
                            rewsame = rew
                        else:
                            rewdiff = rew
                    crossmatrix1[s1][s2] = rewdiff - rewsame
                    averagedif += rewdiff - rewsame
                    if (rewdiff - rewsame) >= 0:
                        print(" %.2f " % (rewdiff - rewsame), flush=True, end='')
                    else:
                        print("%.2f " % (rewdiff - rewsame), flush=True, end='')
                print("") 
            averagedif /= nseeds*nseeds
            print("Average %.2f" % (averagedif))
            cfile = "crosstest.npy" 
            np.save(cfile, crossmatrix1)
        #  "K-nseeds-ninds Extract the best individual among the n last individuals on the basis of the performance obtained agaist 100 competitors of different generations")
        if (parsen[0] == "K"):
            self.policy.test = 0
            if (len(parsen) >= 2): 
                nseeds = int(parsen[1])
            else:
                nseeds = 1
            if (len(parsen) >= 3): 
                ninds = int(parsen[2])
            else:
                ninds = 1
            for s in range(nseeds):
                popfile = "S%dPreyArchive.npy" % (s+1)
                pop = np.load(popfile)
                popshape = pop.shape
                popsize = int(popshape[0])
                print("Seed %d: Extract the best individual among the last ninds individuals by testing the performance against 100 ancestors" % (s+1))
                print("pop :", end='')
                tot = np.zeros(ninds)
                if ninds > 1:
                    for i1 in range(ninds):
                        for i2 in range(100):
                            i2id = int(i2 * (popsize / 100)) 
                            for i3 in range(ninds):
                                self.policy.set_trainable_flat(np.concatenate((pop[popsize-ninds+i1], pop[i2id+i3])))                              
                                rew, eval_length = self.policy.rollout(1)
                                tot[i1] += rew
                        tot[i1] /= 100*ninds
                        print("%.2f " % (tot[i1]), end = "")
                    print("")
                bestindex = 0
                bestval = -999
                for i1 in range(ninds):
                    if tot[i1] >= bestval:
                        bestval = tot[i1]
                        bestindex = i1
                bestgeno = pop1[popsize-ninds+bestindex]
                bestfile = "S%dBestPop.npy" % (s+1)
                np.save(bestfile, bestgeno)
 
