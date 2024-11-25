#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   coevo2.py include an implementation of an competitive co-evolutionary algorithm analogous
   to that described in:
   Simione L and Nolfi S. (2019). Long-Term Progress and Behavior Complexification in Competitive Co-Evolution, arXiv:1909.08303.

   Requires es.py policy.py and evoalgo.py
   Also requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py  
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
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

def debug_print(msg, verbose=False):
    if verbose:
        print(msg)

# competitive coevolutionary algorithm operating on two populations
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)
        # Default number of competitors (required for test purposes)
        self.ncompetitors = 10

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.popsize = 1
            self.ncompetitors = 10
            self.maxsteps = 1000000
            self.ngenerations = 200
            self.stepsize = 0.01
            self.batchSize = 20
            self.noiseStdDev = 0.02
            self.wdecay = 0
            self.saveeach = 100
            self.nphases = 1
            self.fullarchive = True
            self.archivelen = 1000
            self.verbose = 0
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
                if o == "nphases":
                    self.nphases = config.getint("ALGO","nphases")
                    found = 1
                if o == "fullarchive":
                    self.fullarchive = bool(config.getint("ALGO","fullarchive"))
                    found = 1
                if o == "archivelen":
                    self.archivelen = config.getint("ALGO","archivelen")
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
                if o == "verbose":
                    self.verbose = config.getint("ALGO","verbose")
                    # Convert verbose to a boolean value
                    self.verbose = bool(self.verbose)
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("ngenerations [integer]    : max number of generations (default 200)")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("ncompetitors [integer]    : number of competitors (default 10)")
                    print("nphases [integer]         : number of phases the evolution is split into (default 10)")
                    print("archivelen [integer]      : number of competitors in the archive (default 1000)")
                    print("fullarchive [integer]     : enable full archive length (default 0)")
                    print("stepsize [float]          : learning stepsize (default 0.01)")
                    print("samplesize [int]          : samplesize/2 (default 20)")
                    print("noiseStdDev [float]       : samples noise (default 0.02)")
                    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
                    print("saveeach [integer]        : save file every N generations (default 100)")
                    print("verbose [integer]         : verbosity flag (default 0)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))
   

       

    def run(self):

        self.loadhyperparameters()           # load hyperparameters

        seed = self.seed
        self.rs = np.random.RandomState(self.seed)
        #random.seed(self.seed) # Uncomment this line if we want deterministic sampling order for competitors (i.e., deterministic case)

        # Extract the number of parameters
        nparams = int(self.policy.nparams / 2)                                   # parameters required for a single individual
       
        self.candidate = np.arange(nparams, dtype=np.float64)                    # the vector used to store offspring      
       
        # initialize the populations
        # Predators
        self.predpop = []                                                            # the populations (the individuals of the second pop follow)
        self.predpopm = []                                                           # the momentum of the populations
        self.predpopv = []                                                           # the squared momentum of the populations
        self.policy.nn.initWeights()
        randomparams = np.copy(self.policy.get_trainable_flat())
        self.predpop.append(randomparams[:nparams])
        self.predpopm.append(zeros(nparams))
        self.predpopv.append(zeros(nparams))
        self.predpop = np.asarray(self.predpop)
        self.predpopm = np.asarray(self.predpopm)
        self.predpopv = np.asarray(self.predpopv)

        # initialize the archives
        self.predarchive = []
        # The archives are initialized randomly at the beginning!!!
        for i in range(self.ncompetitors):
            # Predators
            self.policy.nn.initWeights()
            randomparams = np.copy(self.policy.get_trainable_flat())
            self.predarchive.append(randomparams[:nparams])
        self.predarchive = np.asarray(self.predarchive)

        # Initialize the matrices containing information about competitors across phases (i.e., overall fitness (f_tot) and number of evaluations (N) --> average fitness is obtained as f_avg = f_tot / N). The matrix contains also the index of the competitor as it appears during evolution and the generation at which a particular competitor is faced the first time. There are two matrices, one for predators and one for preys
        self.predcompmat = np.zeros((self.ncompetitors, (self.nphases * 2) + 2), dtype=np.float64) # Pair (f_tot,N) for each phase + first generation + competitor index
        # Fill first entry (column) of the matrix with the index of the competitor (as it appears during evolution)
        for i in range(self.ncompetitors):
            self.predcompmat[i,0] = i
        # Counter for competitors (to fill the first column of the competitor matrix)
        self.predcompcnt = self.ncompetitors

        # Temporary matrix containing fitness obtained by the different samples against their competitors (the randomly extracted ones)
        self.compmat = np.zeros(((self.batchSize * 2), self.ncompetitors), dtype=np.float64)

        # Initialize the bests with elements in the archive
        self.predbest = []
        # The archives are initialized randomly at the beginning!!!
        for i in range(self.ncompetitors):
            # Predators
            self.predbest.append(self.predarchive[i])
        self.predbest = np.asarray(self.predbest)

        print("Coevo-archive seed %d competitors %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d nparams %d fullarchive %d" % (self.seed, self.ncompetitors, self.batchSize, self.stepsize, self.noiseStdDev, self.wdecay, nparams, self.fullarchive))

        # Verbose flag (for printing all information and debugging code)
        #print(self.verbose)#=True
        
        # Save initial populations (centroids)
        filename = "S%dG0Pred.npy" % (seed)
        np.save(filename, self.predpop)
        # Save initial random archives too!!! (maybe this can be removed)
        filename = "S%dG0PredArchive.npy" % (seed)
        np.save(filename, self.predarchive)

        # Number of generations in each phase
        if self.nphases > 0:
            self.nphasegenerations = int(self.ngenerations / self.nphases)
        else:
            # Default case is constituted of 1 single phase lasting <ngenerations> generations
            self.nphasegenerations = self.ngenerations
            self.nphases = 1

        # Current evolutionary phase
        self.phase = 0
        debug_print(("Evolutionary started at phase %d" % self.phase), verbose=self.verbose)

        # Number of attempts performed to remove a bad competitor from the archive
        nattempts = 10 # To be parametrized??!!!

        # main loop
        gen = 0
        ceval = 0
        while ceval < self.maxsteps:
        #for gen in range(self.ngenerations):

            # Check whether we move to next phase
            if gen >= (self.phase + 1) * self.nphasegenerations:
                # Move to next phase
                self.phase += 1
                debug_print(("Evolution moved to phase %d" % self.phase), verbose=self.verbose)

            self.gen = gen
            # select competitors
            # chooses the selected competitors randomly from the archive
            self.competitors  = random.sample(range(np.shape(self.predarchive)[0]), self.ncompetitors)
                                                                                           
            # evolve individuals for one generation
            avefit, bestfit, bestsamfit, nsteps = self.runphase(0, nparams)
            ceval += nsteps
            #print("centroidfit %.2f" % avefit)

            # Print the competitor matrices every 10 generations (and also at generation 0)
            # We print only the filled rows and the filled columns (i.e., up to current phase, next phases are void/zeroed)
            """
            if gen % 10 == 0:
                predarchivelen = np.shape(self.predarchive)[0]
                preyarchivelen = np.shape(self.preyarchive)[0]
                debug_print(("Preys - predcompmat up to entry %d" % predarchivelen), verbose=self.verbose)
                debug_print((self.predcompmat[0:predarchivelen,0:(2*self.phase+1+2+1)]), verbose=self.verbose)
                debug_print(("Predators - preycompmat up to entry %d" % preyarchivelen), verbose=self.verbose)
                debug_print((self.preycompmat[0:preyarchivelen,0:(2*self.phase+1+2+1)]), verbose=self.verbose)
            """

            if not self.fullarchive:
                # Now check size of the competitor matrix
                if np.shape(self.predarchive)[0] >= self.archivelen:
                    # If the competitor archive (the preyarchive in this case) is longer than the maximum archive length, we try to remove bad competitors
                    # Look for some competitor that is "dominated"
                    candidates = random.sample(range(np.shape(self.predarchive)[0]), nattempts)
                    # Index of the removed competitor
                    remid = -1
                    # Generation at which the removed competitor has been faced
                    remgen = -1
                    # Flags the success of the search for a dominated competitor
                    found = False
                    attempt = 0
                    while attempt < nattempts and not found:
                        # Get the index of the candidate that might be removed
                        c = candidates[attempt]
                        debug_print(("attempt %d - candidate %d" % (attempt,c)), verbose=self.verbose)
                        # Skip candidate flag
                        skip_cand = False
                        # Check whether or not the current candidate must be considered for potential deletion
                        if np.sum(self.predcompmat[c,1:]) == 0.0: # We exclude the first entry (i.e., the competitor id)
                            # Candidate has not yet been evaluated, we skip it
                            debug_print(("candidate %d has not yet been evaluated, we cannot delete it" % c), verbose=self.verbose)
                            skip_cand = True
                        if not skip_cand:
                            # Row index
                            row = 0
                            # Loop over rows of the competitor matrix
                            while row < np.shape(self.predcompmat)[0] and not found:
                                skip_row = False
                                # Skip row if it matches the candidate or if the row is fully zeroed (i.e., competitors not yet faced)
                                if row == c or np.sum(self.predcompmat[row,1:]) == 0.0:
                                    debug_print(("skipped row %d" % row), verbose=self.verbose)
                                    skip_row = True
                                if not skip_row:
                                    debug_print(("candidate %d against %d" % (c, row)), verbose=self.verbose)
                                    # Flags whether we must stop search for that row
                                    stop = False
                                    # Column index
                                    col = 2 # We do not take into account the ids and the generations for the comparison
                                    # Loop over columns of the competitor matrix
                                    while col < np.shape(self.predcompmat)[1] and not stop:
                                        # Compute average fitness of the candidate for that particular phase
                                        if self.predcompmat[c,col+1] > 0:
                                            cand_fit = self.predcompmat[c,col] / self.predcompmat[c,col+1]
                                        else:
                                            cand_fit = 0.0
                                        # Compute average fitness of the other row for that particular phase
                                        if self.predcompmat[row,col+1] > 0:
                                            opp_fit = self.predcompmat[row,col] / self.predcompmat[row,col+1]
                                        else:
                                            opp_fit = 0.0
                                        debug_print(("column %d - candidate fitness %.3f vs other fitness %.3f" % (col, cand_fit, opp_fit)), verbose=self.verbose)
                                        # If candidate is better than the other row, stop search since the candidate is better than another competitor
                                        if cand_fit > opp_fit:
                                            stop = True
                                        col += 2
                                    if not stop:
                                        debug_print(("candidate %d is dominated by %d" % (c,row)), verbose=self.verbose)
                                        found = True
                                row += 1
                        if found:
                            remid = self.predcompmat[c,0]
                            remgen = self.predcompmat[c,1]
                            self.predcompmat = np.delete(self.predcompmat, c, 0)
                            # Delete row from archive
                            self.predarchive = np.delete(self.predarchive, c, 0)
                            # Delete row from best
                            self.predbest = np.delete(self.predbest, c, 0)
                            print("Removed competitor %d of generation %d inserted at row %d of compmat - attempt %d" % (remid, remgen, c, attempt))
                        # Update the number of attempts
                        attempt += 1
                    # Print the result of the search for a dominated individual
                    if not found:
                        print("None of the candidates is dominated in compmat - candidates:", candidates)
                    else:
                        if attempt > 1:
                            debug_print(("Generation %d - performed %d attempts to remove a dominated individual from the archive" % (gen, attempt)), verbose=self.verbose)  
                        else:
                            debug_print(("Generation %d - performed %d attempt to remove a dominated individual from the archive" % (gen, attempt)), verbose=self.verbose)                    

            # save evolving populations
            #if gen > 0 and ((gen % self.saveeach) == 0):
            if (((gen + 1) % self.saveeach) == 0):
                # Save archives (predators and preys)
                filename = "S%dPredArchive.npy" % (seed)
                np.save(filename, self.predarchive)
                # Save bests (predators and preys)
                filename = "S%dPredBest.npy" % (seed)
                np.save(filename, self.predbest)
                # Save competitor matrices (predators and preys)
                filename = "S%dPredCompMat.npy" % (seed)
                np.save(filename, self.predcompmat)
                # Save stats too!
                fname = self.filedir + "/statS" + str(self.seed)
                np.save(fname, self.stat)
            # print and save statistics
            print("seed %d gen %d step %d centroidfit %.2f bestsamplefit %.2f bestfit %.2f archive %d weights %.2f" % (seed, gen, ceval, avefit, bestsamfit, bestfit, len(self.predarchive), np.average(np.absolute(self.predpop))))
            self.stat = np.append(self.stat, [0, avefit, avefit, avefit, avefit, 0])  # store performance across generations

            # Update generation
            gen += 1

        # Save all files at the end of the evolutionary process
        # Archives (predators and preys)
        filename = "S%dPredArchive.npy" % (seed)
        np.save(filename, self.predarchive)
        # Best solutions (predators and preys)
        filename = "S%dPredBest.npy" % (seed)
        np.save(filename, self.predbest)
        # Save competitor matrices (predators and preys)
        filename = "S%dPredCompMat.npy" % (seed)
        np.save(filename, self.predcompmat)
        # Centroid and momentum vectors (m and v) for predators)
        filename = "S%dG%dPred.npy" % (seed, gen + 1)
        np.save(filename, self.predpop)
        filename = "S%dG%dPredm.npy" % (seed, gen + 1)
        np.save(filename, self.predpopm)
        filename = "S%dG%dPredv.npy" % (seed, gen + 1)
        np.save(filename, self.predpopv)
       
           
    # performs a generation  
    def runphase(self, sind, nparams, evolvepreys=False):
        # Number of steps run in this generation
        nsteps = 0
        # Adam optimizer parameters
        epsilon = 1e-08
        beta1 = 0.9
        beta2 = 0.999
        # Weight initialization
        weights = zeros(self.batchSize)
        # Current generation best (fitness and the corresponding individual)
        cbest = -99999.0
        cbestid = -1
        cbestind = None
        for it in range (1):
            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = self.rs.randn(self.batchSize, nparams)
            fitness = zeros(self.batchSize * 2)
            # Evaluate offspring
            for b in range(self.batchSize):
                # For each sample we evaluate both positive variation and negative one
                for bb in range(2):
                    if (bb == 0):
                        # Positive variation
                        for g in range(nparams):
                            self.candidate[g] = self.predpop[sind][g] + samples[b,g] * self.noiseStdDev
                    else:
                        # Negative variation
                        for g in range(nparams):
                            self.candidate[g] = self.predpop[sind][g] - samples[b,g] * self.noiseStdDev
                    # Evaluate against all competitors
                    ave_rews = 0
                    for c in range(self.ncompetitors):
                        self.policy.set_trainable_flat(np.concatenate((self.candidate, self.predarchive[self.competitors[c]])))
                        eval_rews, eval_length = self.policy.rollout(1, seed=(self.seed+self.gen*1000+it*100+b))
                        ave_rews += eval_rews
                        nsteps += eval_length # Update the number of run steps
                        # Fill temporary matrix with competitor fitness
                        self.compmat[b*2+bb,c] = eval_rews # Fitness is inverted since the competitor is a prey
                        fitness[b*2+bb] = ave_rews / float(self.ncompetitors)
                        # Check whether or not the offspring is better than current best
                        if fitness[b*2+bb] > cbest:
                            # Found a new best, update it
                            cbest = fitness[b*2+bb]
                            cbestid = b*2+bb
                            cbestind = np.copy(self.candidate)
            # Once the best sample has been found, we store the values obtained against competitors in the corresponding competitor matrix
            for c in range(self.ncompetitors):
                if self.predcompmat[self.competitors[c],1] == 0:
                    self.predcompmat[self.competitors[c],1] = self.gen
                # Update the fitness for that competitor
                self.predcompmat[self.competitors[c],2+(2*self.phase)] += self.compmat[cbestid,c]
                # Update the number of times the competitor has been evaluated
                self.predcompmat[self.competitors[c],2+(2*self.phase+1)] += 1.0
                debug_print(("Updating predcompmat with data for competitor %d" % self.competitors[c]), verbose=self.verbose)
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
            a = self.stepsize # bias correction is not implemented# Generations for predators and preys
            self.predpopm[sind] = beta1 * self.predpopm[sind] + (1.0 - beta1) * globalg
            self.predpopv[sind] = beta2 * self.predpopv[sind] + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * self.predpopm[sind] / (sqrt(self.predpopv[sind]) + epsilon)
            # update center
            self.predpop[sind] += dCenter

        #print(cbest)
        # We store sample best fitness (for printing information)
        sbest = cbest

        # evaluate the evolving individual at the end of the evolution phase
        # Evaluate against all competitors
        ave_rews = 0
        for c in range(self.ncompetitors):
            self.policy.set_trainable_flat(np.concatenate((self.predpop[sind], self.predarchive[self.competitors[c]])))
            eval_rews, eval_length = self.policy.rollout(1)
            ave_rews += eval_rews
            nsteps += eval_length # Update the number of run steps
        ave_rews /= float(self.ncompetitors)
        # Check whether or not centroid is better than current best
        if ave_rews > cbest:
            # Found a new best, update it
            cbest = ave_rews
            cbestind = np.copy(self.predpop[sind])
        #print("%.2f " % (ave_rews), end = '')
       
        # store the new individual in the archive
        self.predarchive = np.vstack((self.predarchive, self.predpop[sind]))
        # Add a new row (entry) for the new potential competitor
        compvec = np.zeros((self.nphases * 2) + 2, dtype=np.float64)
        # Set the index of the new potential competitor
        compvec[0] = self.predcompcnt
        self.predcompmat = np.vstack((self.predcompmat, compvec))
        debug_print(("Added a new potential competitor %d at predcompmat, size = %d - archive size = %d" % (self.predcompcnt, np.shape(self.predcompmat)[0], np.shape(self.predarchive)[0])), verbose=self.verbose)
        # Update index
        self.predcompcnt += 1
        # Store the new best
        self.predbest = np.vstack((self.predbest, cbestind))

        #print(ave_rews)
        #print("gen %d cbest %.2f" % (self.gen, cbest))
        return ave_rews, cbest, sbest, nsteps

    def testusage(self):
        print("ERROR: To post-evaluate with the coevo algorithm you should specify with the -g parameter a string containing:")
        print("P-g-gg  : Postevaluates the centroid of generation g against the centroid of generation gg by showing the behavior")
        print("p-g-gg  : Postevaluates the centroid of generation g against the centroid of generation gg withou showing the behavior")        
        print("M-g-gg  : Creates a master.npy matrix with the post-evaluation performance up to generation g every gg generations")
        print("        : Also extract the strongest individuals against each postevaluated individual")              
        print("S-g-gg  : Generates an AgainstStrongest matrix by postevaluate individuals up to generation g, every gg individuals, against the strongests")
        print("X-gg-g  : Postevaluates up to geration g every gg individuals agaist pre-trained opponent")
        print("B-gg-g  : Postevaluates up to geration g every gg individuals agaist pre-trained opponent (best individuals)")
        print("s-s1-s2 : Postevaluates the last individual of seed s1 against the last individual of seed s2")
        print("C-f1-f2 : Postevaluates the last 20 individuals of file f1 against the last 20 individuals of file f2")
        sys.exit()

    def test(self, testparam):
        if testparam is None:
            self.testusage()
        if "-" not in testparam:
            self.testusage()      
        seed = self.seed
        parsen = testparam.split("-")
        #if (len(parsen) != 3 or not parsen[0] in ["P", "p", "M","S", "C", "X", "B", "s"]):
        if not parsen[0] in ["P", "p", "M", "S", "C", "c", "X", "B", "s", "x", "K"]:
            self.testusage()
        # Load hyperparameters (we need this in order to know the number of competitors)
        self.loadhyperparameters()           # load hyperparameters
        # P-g-gg: Postevaluate the centroid of generation g against the competitor of generation gg
        # P renders behavior, "p" only print fitness            
        if (parsen[0] == "p" or parsen[0] == "P"):
            if (parsen[0] == "P"):
                self.policy.test = 1
                rendt = True
            else:
                self.policy.test = 0
                rendt = False
            # Enable rendering
            print("TEST P")
            #self.env.render()
            try:
                # Set test flag (to print debug information)
                self.env.setTest()
            except:
                pass
            popfile = "S%dPredArchive.npy" % (seed)
            predpop = np.load(popfile)
            popshape = predpop.shape
            popsize = popshape[0]
            #print(popsize)
            #popfile = "S%dPreyArchive.npy" % (seed)
            #preypop = np.load(popfile)
            

            g = int(parsen[1])
            gg = int(parsen[2])
            tot_rew = 0
            print("seed %d: Postevaluate gen %d against gen %d " % (seed, g, gg))
 
            self.policy.set_trainable_flat(np.concatenate((predpop[g], predpop[gg])))
            eval_rews, eval_length = self.policy.rollout(1)
            tot_rew += eval_rews
            print("fitness gen %d : %.2f " % (g, tot_rew))

        # "M-n1-n2, Master tournament, test all generations (up to n1) against every generations, every n2 generations
        if (parsen[0] == "M"):
            gpop1 = 0
            gpop2 = 0
            popfile = "S%dPredArchive.npy" % (seed)
            preypop = np.load(popfile)
            preypop = preypop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = preypop.shape
            popsize = popshape[0]
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
            gpop1 = 0
            gpop2 = 0
            uptogen = int(parsen[1])                # up to generation N
            everygen = int(parsen[2])               # every N generation
            popfile = "S%dPredArchive.npy" % (seed)
            predpop = np.load(popfile)
            #predpop = predpop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = predpop.shape
            #print(popshape)
            popsize = popshape[0]
            popfile = "S%dPredArchive.npy" % (seed)
            predpop = np.load(popfile)
            #predpop = predpop[self.ncompetitors:,:] # We exclude first random competitors
            popshape = predpop.shape
            #print(popshape)
            #sys.exit()
            popsize = popshape[0]
            numgen = int((uptogen / everygen))# + 1)   # number of generations
            self.policy.test = 0                     # do not shows the behavior
            print("seed %d: postevaluation all generations (up to %d) against all generations every %d generations" % (seed, uptogen, everygen))
            master = np.zeros((numgen, numgen), dtype=np.float64) # matrix with the average performance of every generation against every other generation
            for pn1 in range(numgen):
                for pn2 in range(numgen):
                    self.policy.set_trainable_flat(np.concatenate((predpop[pn1*everygen], predpop[pn2*everygen])))                              
                    rew, eval_length = self.policy.rollout(1)
                    master[pn1, pn2] = rew
            mfile = "masterS%d.npy" % (seed)
            np.save(mfile, master)
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
            """
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
                    self.policy.set_trainable_flat(np.concatenate((predpop[gg - 1], predpop[i*g+ii])))                              
                    rew, eval_length = self.policy.rollout(1)
                    if (rew > max_ind_rew):
                        max_ind_rew = rew
                    tot_rew += rew
                    #print("pred - gen %d-%d fit %.2f" % (gg, i*g+ii, rew))
                print("pred - gen %d-%d ave %.2f max %.2f" % (gg, i*g, (tot_rew / g), max_ind_rew))
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
            g = int(parsen[1])
            gg = int(parsen[2])
            nphases = int(gg / g)
            for i in range (nphases):
                tot_rew = 0
                max_ind_rew = -99999
                for ii in range(g):
                    ind_rew = 0
                    self.policy.set_trainable_flat(np.concatenate((predpop[gg - 1], predpop[i*g+ii])))                              
                    rew, eval_length = self.policy.rollout(1)
                    if (rew > max_ind_rew):
                        max_ind_rew = rew
                    tot_rew += rew
                    #print("pred - gen %d-%d fit %.2f" % (gg, i*g+ii, rew))
                print("pred - gen %d-%d ave %.2f max %.2f" % (gg, i*g, (tot_rew / g), max_ind_rew))
        # s-g-gg: Postevaluate the centroid of generation g against the competitor of generation gg
        # P renders behavior, "p" only print fitness            
        if (parsen[0] == "s"):
            try:
                # Set test flag (to print debug information)
                self.env.setTest()
            except:
                pass
            s1 = int(parsen[1])
            s2 = int(parsen[2])
            popfile = "S%dPredArchive.npy" % (s1)
            predpop = np.load(popfile)
            predshape = predpop.shape
            predsize = predshape[0]
            
            tot_rew = 0
            print("Postevaluate individual of seed %d against individual of seed %d " % (s1, s2))
            
            # We get last entry from both archives (i.e., last centroids that have been stored in the archives)
            self.policy.set_trainable_flat(np.concatenate((predpop[predsize - 1], predpop[preysize - 1])))
            eval_rews, eval_length = self.policy.rollout(1)
            tot_rew += eval_rews
            print("fitness seed %d vs seed %d : %.2f " % (s1, s2, tot_rew))
        # "C-file1-file2, postevaluate the last 20 individuals of file1 against the last 20 individuals of file2
        if (parsen[0] == "C"):
            self.policy.test = 0
            ninds = 100#20
            print("Postevaluate the last %d individuals of %s against the last %d individuals of %s " % (ninds, parsen[1], ninds, parsen[2]))
            # Predators
            fname1 = parsen[1] + "PredArchive.npy"
            pop1 = np.load(fname1)#parsen[1])
            popshape1 = pop1.shape
            popsize1 = int(popshape1[0])
            fname2 = parsen[2] + "PredArchive.npy"
            pop2 = np.load(fname2)#parsen[2])
            popshape2 = pop2.shape
            popsize2 = int(popshape2[0])
            #assert popshape1[1] == popshape2[1], "the number of parameters in the two file is inconsistent"
            tot_rew = 0.0
            for i1 in range(ninds):
                for i2 in range(ninds):
                    self.policy.set_trainable_flat(np.concatenate((pop1[popsize1-i1-1], pop2[popsize2-i2-1])))                              
                    rew, eval_length = self.policy.rollout(1)
                    #print(i1, i2, rew)
                    tot_rew += rew
            tot_rew /= ninds*ninds        
            print("Individuals of %s against individuals of %s - Avg fit: %.2f " % (fname1, fname2, tot_rew), flush=True)
        # "c-dir1-dir2-seeds-ninds, postevaluate the last 20 individuals of file1 against the last 20 individuals of file2
        if (parsen[0] == "c"):
             self.policy.test = 0
             dir1 = parsen[1]
             dir2 = parsen[2]
             if not dir1.endswith("/"):
                 dir1 += "/"
             if not dir2.endswith("/"):
                 dir2 += "/"
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
             for s1 in range(nseeds):
                 for s2 in range(nseeds):
                     tot_seed = [0,0]
                     for t in range(2):  #dir1 against dir1 and dir1 against dir 2
                         pop1file = "%sS%dPredArchive.npy" % (dir1, s1 + 1)
                         pop1 = np.load(pop1file)
                         popshape1 = pop1.shape
                         popsize1 = int(popshape1[0])
                         if (t == 0):
                             pop2file = "%sS%dPredArchive.npy" % (dir1, s2 + 1)
                         else:
                             pop2file = "%sS%dPredArchive.npy" % (dir2, s2 + 1)
                         pop2 = np.load(pop2file)
                         popshape2 = pop2.shape
                         popsize2 = int(popshape2[0])
                         for i1 in range(ninds):
                             for i2 in range(ninds):

                                 self.policy.set_trainable_flat(np.concatenate((pop1[popsize1-i1-1], pop2[popsize2-i2-1])))
                                 rew, eval_length = self.policy.rollout(1)
                                 #print(s1, s2, i1, i2, rew)
                                 tot_seed[t] += rew
                                 tot_seeds[t] += rew
                         tot_seed[t] /= ninds*ninds
                     print("Indidivual seeds %d-%d Fitness gain with different competitors: %.2f " % (s1 + 1, s2 + 1, tot_seed[1] - tot_seed[0]), flush=True)
             tot_seeds[0] /= nseeds*nseeds*ninds*ninds
             tot_seeds[1] /= nseeds*nseeds*ninds*ninds
             print("Individuals Fitness gain with different competitors %.2f" % (tot_seeds[1] - tot_seeds[0]), flush=True)
        # "x-dir-seeds-ninds, postevaluate the last 20 individuals of file1 against the last 20 individuals of file2
        if (parsen[0] == "x"):
             self.policy.test = 0
             cdir = parsen[1]
             if not cdir.endswith("/"):
                 cdir += "/"
             if (len(parsen) >= 3):
                 nseeds = int(parsen[2])
             else:
                 nseeds = 1
             if (len(parsen) == 4):
                 ninds = int(parsen[3])
             else:
                 ninds = 50
             print("Crosstest experiments of dir %s (%d seeds, %d individuals) " % (cdir, nseeds, ninds))
             tot_seeds = 0
             for s1 in range(nseeds):
                 for s2 in range(nseeds):
                     tot_seed = 0
                     #dir1 against dir1 and dir1
                     popfile = "%sS%dPredArchive.npy" % (cdir, s1 + 1)
                     pop = np.load(popfile)
                     popshape = pop.shape
                     popsize= int(popshape[0])
                     for i1 in range(ninds):
                         for i2 in range(ninds):
                             self.policy.set_trainable_flat(np.concatenate((pop[popsize-i1-1], pop[popsize-i2-1])))
                             rew, eval_length = self.policy.rollout(1)
                             tot_seed += rew
                             tot_seeds += rew
                     tot_seed /= ninds*ninds
                     print("Indidivual seeds %d-%d Fitness gain with different competitors: %.2f " % (s1 + 1, s2 + 1, tot_seed), flush=True)
             tot_seeds /= nseeds*nseeds*ninds*ninds
             print("Individuals Fitness gain with different competitors %.2f" % (tot_seeds), flush=True)
         #  "K-nseeds-ninds the best individual among the n last individuals on the basis of the performance obtained agaist 100 competitors of different generations")
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
                 popfile = "S%dPredArchive.npy" % (s+1)
                 pop = np.load(popfile)
                 popshape = pop.shape
                 popsize = int(popshape[0])
                 print("Seed %d: Extract the best individual among the last ninds individuals by testing the performance against 100 ancestors" % (s+1))
                 #print("popsize1 %d vs popsize2 %d" % (popsize1, popsize2))
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
                 bestgeno = pop[popsize-ninds+bestindex]
                 bestfile = "S%dBestPop.npy" % (s+1)
                 np.save(bestfile, bestgeno)
