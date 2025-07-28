#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:59:28 2024

@author: pedro
"""

#import os
import math
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import random as rd
from os.path import dirname, abspath
import time
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class DoubleTMazeEnv(gym.Env):
    #angles are in rads, distance is in cms (when in mms its specified in comments)
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps' : 50
    }
    
    def __init__(self, render_mode: Optional[str] = None, record: Optional[str] = None, options: dict = {}):
        
        if options == None:
            options = {}
        self.options = options
        
        self.episode = 0
        self.terminated = False
        self.truncated = False
        self.reward = 0.
        self.info = {}
        
        self.SCORES = []
        
        self.timelimit = options.get("timelimit", 800)
        self.stall = options.get("stall", 100)
        self.scale = options.get('scale', 0.75)
        
        self.sh = options.get("screen_hight", 850) 
        self.sw = options.get("screen_width", 700) 
        
        #initialize robot
        self.robot_radius = 8.5 * self.scale
        self.robotAxleLenght = 10.4 * self.scale
        self.maxV = 3.6 * self.scale
        
        #initialize IR sensor
        self.IRmin = 0.4 * self.scale
        self.IRmax = 10.4 * self.scale
        
        #cam
        self.camrange = [
            [0, math.pi/2],
            [math.pi/2, math.pi],
            [math.pi, math.pi * 3/2],
            [math.pi * 3/2, math.pi *2]
            ]
        
        #initialize beacons
        self.beacon_radius = options.get("beacon_radius") * self.scale if "beacon_radius" in options else 8.5 * self.scale
        
        #calc max beacon
        if self.beacon_radius >= self.robot_radius:
            self.max_beacon = 1
        
        else:
            ori = math.pi /4
            
            topvec = [-self.robot_radius, self.beacon_radius]
            botvec = [-self.robot_radius, -self.beacon_radius]
            
            topang = math.atan2(topvec[1], topvec[0]) +math.pi + ori
            botang = math.atan2(botvec[1], botvec[0]) +math.pi + ori ###ADDING +pi to normalize atan2 result from -pi, pi to 0,2pi
                
            rg = [min(topang, botang), max(topang, botang)]
            
            while rg[0] > math.pi *2:
                rg[0] -= math.pi * 2
            while rg[0] < 0:
                rg[0] += math.pi * 2
                
            while rg[1] > math.pi *2:
                rg[1] -= math.pi * 2
            while rg[1] < 0:
                rg[1] += math.pi * 2
                
            if rg[1] < 0 or math.pi/2 < rg[0]:
                self.max_beacon = 0
                cross = None
            
            else:
                cross = [ max(rg[0], 0),
                          min(rg[1], math.pi/2)]
                
                self.max_beacon = abs( cross[1] - cross[0] ) /  abs( math.pi/2)
                
        #initialize corner cylinders
        self.cylr = 8.5 * self.scale
               
        #initialize scanner
        
        self.scanner_range = 1000 * self.scale
        
        self.neurs = 6 # nr of neurons/sectors that read from scanner
        self.n_beams = 3 # nr of beams p/ sector
        
        self.scannerOrs = [-27.5, -25, -22.5, #values in degrees
                 -17.5, -15, -12.5,
                 -7.5, -5, -2.5,
                 2.5, 5, 7.5,
                 12.5, 15, 17.5,
                 22.5, 25, 27.5]
        
        #spaces
        observation_size = 8 + len(self.camrange)*2 + self.neurs
        
        # shape of action
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        
        # shape of observation
        self.observation = []
        self.observation_space = spaces.Box(0.0, 1.0, shape=(observation_size,), dtype='float32')
        
        #render
        self.render_mode = render_mode
        self.frames = []
        self.record = record
        self.screen = None
        
        #define self.IRwall, loading values from file
        with open( 'IRwall.txt',
                  'r') as data:
            self.IRwall = data.read()
            data.close()
            
        self.IRwall = self.IRwall.split('\n')

        for i in range ( len ( self.IRwall)):
            self.IRwall[i] = self.IRwall[i].split(' ')
        
        for i in range ( len (self.IRwall)):
            if len( self.IRwall[i]) == 8:
                for j in range( len(self.IRwall[i])):
                    self.IRwall[i][j] = float(self.IRwall[i][j])
        
        #define self.IRcylinder, loading values from file
        with open( 'IRcylinder.txt',
                  'r') as data:
            self.IRcylinder = data.read()
            data.close()
            
        self.IRcylinder = self.IRcylinder.split('\n')

        for i in range ( len ( self.IRcylinder)):
            self.IRcylinder[i] = self.IRcylinder[i].split(' ')
        
        for i in range ( len (self.IRcylinder)):
            if len( self.IRcylinder[i]) == 8:
                for j in range( len(self.IRcylinder[i])):
                    self.IRcylinder[i][j] = float(self.IRcylinder[i][j])
        
    def reset(self, options: dict = {}, seed = None):
        
        #initialize global values
        
        if options == None:
            options = {}
            
        options = options.copy()
        options.update(self.options)
        
        seed = options.get('seed', seed)
        super().reset(seed=seed)
        
        self.envStepCounter = 0
        
        #initialize the maze
        if "width_noise" in options and "width_noise" != 0:
            width_noise = rd.randrange(0 - options.get("width_noise"), 0 + options.get("width_noise") +1, 1) 
        else: width_noise = 0
        
        if "c1l_noise" in options and options.get("c1l_noise") != 0:
            c1l_noise = rd.randrange(0 - options.get("c1l_noise"), 0 + options.get("c1l_noise") +1, 1) 
        if "c1l_noise" in options and options.get("c1l_noise") == 0:
            c1l_noise = 0
        else: c1l_noise = rd. randrange(-50, 51, 1)
        
        if "c2l_noise" in options and options.get("c2l_noise") != 0:
            c2l_noise = rd.randrange(0 - options.get("c2l_noise"), 0 + options.get("c2l_noise") +1, 1) 
        if "c2l_noise" in options and options.get("c2l_noise") == 0:
            c2l_noise = 0
        else: c2l_noise = rd. randrange(-50, 51, 1)
        
        if "c3l_noise" in options and options.get("c3l_noise") != 0:
            c3l_noise = rd.randrange(0 - options.get("c3l_noise"), 0 + options.get("c3l_noise") +1, 1)
        elif "c3l_noise" in options and options.get("c3l_noise") == 0:
            c3l_noise = 0
        else: c3l_noise = rd. randrange(-50, 51, 1)
        
        self.chw = (options.get("width") + width_noise)*self.scale if "width" in options else (40 + width_noise)*self.scale
        self.c1l = (options.get("c1l") + c1l_noise)*self.scale if "c1l" in options else (300 + c1l_noise)*self.scale
        self.c2l = ((options.get("c2l") - self.chw*2 + c2l_noise)/2)*self.scale if "c2l" in options else ((300 - self.chw*2 + c2l_noise)/2)*self.scale
        self.c3l = ((options.get("c3l") - self.chw*2 + c3l_noise)/2)*self.scale if "c3l" in options else ((400 - self.chw*2 + c3l_noise)/2)*self.scale
        self.goal_size = options.get("goal_size") if "goal_size" in options else self.c3l /3
        beacon_pos = options.get("beacon_pos") if "beacon_pos" in options else [self.goal_size, self.goal_size + (self.c1l-self.goal_size)/2]
        self.scn = options.get("trial")%4 +1 if "trial"  in options else rd.randrange(1,5)
            
        self.robotOr = options.get("orientation") * math.pi / 180 if "orientation" in options else 90 * math.pi / 180
        self.ix_noise = int ( Decimal (options.get("xnoise") /10 * self.scale).to_integral_value(rounding=ROUND_HALF_UP)) if "xnoise" in options else int ( Decimal (self.chw/4 * self.scale).to_integral_value(rounding=ROUND_HALF_UP))
        self.iy_noise = int ( Decimal (options.get("ynoise") /10 * self.scale).to_integral_value(rounding=ROUND_HALF_UP)) if "ynoise" in options else int ( Decimal (self.chw/4 * self.scale).to_integral_value(rounding=ROUND_HALF_UP))
        self.robotPos = options.get("position") if "position" in options else [float(0), self.robot_radius + float(self.iy_noise) + self.goal_size/2]
        self.robotPos = [float(self.robotPos[0]), float(self.robotPos[1])]
        self.o_noise = options.get("onoise") if "onoise" in options else 45
        
        #actuators
        self.vt = [0,0] # initialize velocity for the wheels
        
        #set initial position based on values passed in options
        
        if self.ix_noise == 0 and self.iy_noise != 0:
            self.robotPos = [self.robotPos[0], (self.robotPos[1] + rd.randrange(-self.iy_noise *100, (self.iy_noise)*100 +1, 1)/100)]
        elif self.ix_noise != 0 and self.iy_noise == 0:
            self.robotPos = [(self.robotPos[0] + rd.randrange(-self.ix_noise*100, (self.ix_noise)*100 +1, 1)/100), self.robotPos[1]]
        elif self.ix_noise == 0 and self.iy_noise == 0:
            pass
        else:
            self.robotPos = [(self.robotPos[0] + rd.randrange(-self.ix_noise*100, (self.ix_noise)*100 +1, 1)/100), (self.robotPos[1] + rd.randrange(-self.iy_noise*100, (self.iy_noise)*100 +1, 1)/100)]
            
        #set initial orientation based on values passed in options
        if self.o_noise != 0:
            self.robotOr += rd.randrange(-self.o_noise, self.o_noise +1, 1) * math.pi / 180
        
        while self.robotOr > math.pi * 2:
            self.robotOr -= math.pi * 2
        while self.robotOr < 0:
            self.robotOr += math.pi * 2
        
        #RESET ENV:
        self.reward = 0
        self.rsum = 0
        self.prog = 0
        self.rpeak = 0
        self.timer = 0
        self.truncated = False
        self.terminated = False
        
        #calculate maximum fitness/reward value
        self.md = self.c1l + self.c2l + self.c3l + 2*self.chw - self.goal_size
        
        #initialize goal destinations
        self.goal_matrix = [
            [-self.c2l-self.chw*3,
             self.c1l-self.c3l+self.goal_size,
             -self.c2l-self.chw,
             self.c1l-self.c3l+self.goal_size],
            
            [-self.c2l-self.chw*3,
             self.c1l+self.c3l+self.chw*2-self.goal_size,
             -self.c2l-self.chw,
             self.c1l+self.c3l+self.chw*2-self.goal_size],
            
            [self.c2l+self.chw*3,
             self.c1l+self.c3l+self.chw*2-self.goal_size,
             self.c2l+self.chw,
             self.c1l+self.c3l+self.chw*2-self.goal_size],
            
            [self.chw+self.c2l,
             self.c1l-self.c3l+self.goal_size,
             self.c2l+self.chw*3,
             self.c1l-self.c3l+self.goal_size]
            ]
        
        #initialize walls
        self.wall_matrix = [
        #each wall is a vector of 4 values [xmin, ymin, xmax, ymax]
            #start corridor
            [-self.chw,
             0,
             self.chw,
             0],
            
            [-self.chw,
             0,
             -self.chw,
             self.c1l],
            
            [self.chw,
             0,
             self.chw,
             self.c1l],
            
            #middle corridor
            [-(self.chw + self.c2l),
             self.c1l,
             -self.chw,
             self.c1l],
            
            [self.chw,
             self.c1l,
             self.chw + self.c2l,
             self.c1l],
            
            [-(self.chw + self.c2l),
             self.chw*2 + self.c1l,
             self.chw + self.c2l,
             self.chw*2 + self.c1l],
            
            #left corridor
            [-(self.chw + self.c2l),
             self.c1l - self.c3l,
             -(self.chw + self.c2l),
             self.c1l],
            
            [-(self.chw*3 + self.c2l),
             self.c1l - self.c3l,
             -(self.chw + self.c2l),
             self.c1l - self.c3l],
            
            [-(self.chw + self.c2l),
             self.chw*2 + self.c1l,
             -(self.chw + self.c2l),
             self.chw*2 + self.c1l + self.c3l],
            
            [-(self.chw*3 + self.c2l),
             self.chw*2 + self.c1l + self.c3l,
             -(self.chw + self.c2l),
             self.chw*2 + self.c1l + self.c3l],
            
            [-(self.chw*3 + self.c2l),
             self.c1l - self.c3l,
             -(self.chw*3 + self.c2l),
             self.chw*2 + self.c1l + self.c3l],
            
            #right corridor
            [self.chw + self.c2l,
             self.c1l - self.c3l,
             self.chw + self.c2l,
             self.c1l],
            
            [self.chw + self.c2l,
             self.c1l - self.c3l,
             self.chw*3 + self.c2l,
             self.c1l - self.c3l],
            
            [self.chw + self.c2l,
             self.chw*2 + self.c1l,
             self.chw + self.c2l,
             self.chw*2 + self.c1l + self.c3l],
            
            [self.chw + self.c2l,
             self.chw*2 + self.c1l + self.c3l,
             self.chw*3 + self.c2l,
             self.chw*2 + self.c1l + self.c3l],
            
            [self.chw*3 + self.c2l,
             self.c1l - self.c3l,
             self.chw*3 + self.c2l,
             self.chw*2 + self.c1l + self.c3l],
            
            ]
        
        #each cylinder is defined by its center
        self.cylinder_matrix = [[-self.chw, self.c1l], #1st left corner
                                [self.chw, self.c1l], #1st right corner
                                [-(self.chw + self.c2l), self.c1l], #far left corner
                                [self.chw + self.c2l, self.c1l], #far right corner
                                [-(self.chw + self.c2l), self.c1l + self.chw*2], #top left corner
                                [self.chw + self.c2l, self.c1l + self.chw*2] #top right corner
                                ]
        
        beacon_pos = [self.c1l/3, self.c1l/3*2] #sets y lower and higher pos of beacons
        
        #initialize beacons
        self.beacon_matrix = [
            [-self.chw, beacon_pos[0]],
            [self.chw, beacon_pos[0]],
            
            [-self.chw, beacon_pos[1]],
            [self.chw, beacon_pos[1]]
            ]
        
        #self.blue_beacon = None
        
        if self.scn == 1:
            self.beacons_on = [0,2]
            self.blue_beacon = [-(2 * self.chw +self.c2l), self.c1l-self.c3l ]
        elif self.scn == 2:
            self.beacons_on = [0,3]
            self.blue_beacon = [-2 * self.chw -self.c2l, self.c1l + self.c3l + 2*self.chw ]
        elif self.scn == 3:
            self.beacons_on = [1,2]
            self.blue_beacon = [2 * self.chw + self.c2l, self.c1l + self.c3l + 2*self.chw ]
        else: #scn == 4
            self.beacons_on = [1,3]
            self.blue_beacon = [2 * self.chw + self.c2l, self.c1l-self.c3l]
            
        #initialize controller inputs
        self.observation = self.observe()
        
        self.info = {"position" : self.robotPos,
                "orientation" : self.robotOr,
                "width" : self.chw,
                "c1l" : self.c1l,
                "c2l" : self.c2l,
                "c3l" : self.c3l,
                "max_dist" : self.md,
                "scenario" : self.scn,
                "iteration" : self.envStepCounter,
         }
        
        if self.render_mode == 'human':
            self.render()
            
        return np.array(self.observation), self.info
    
    def move(self, action):
        #this function gets the output of the network controller as input (action), then calculates the speed and moves the robot
        #action is a np.array(), shape = (2,) with values between 0.0 and 1.0 as in action_space
        
        #update speed:
        self.vt[0] = self.maxV * action[0]
        self.vt[1] = self.maxV * action[1]
        
        linear_velocity = (self.vt[0] + self.vt[1])/2
        angular_velocity = (self.vt[0]/10 - self.vt[1]/10)/(self.robotAxleLenght/100) * 0.1
        #for angular velocity speeds must be in m/s and axlelenght in m
        
        #update orientation
        #oldir = self.robotOr
        self.robotOr += angular_velocity
        
        #self.robotOr %= math.pi*2
        while self.robotOr > math.pi *2:
            self.robotOr -= math.pi * 2
        while self.robotOr < 0:
            self.robotOr += math.pi * 2
        
        #update position
        self.robotPos[0] += (math.cos(self.robotOr) ) * linear_velocity
        self.robotPos[1] += (math.sin(self.robotOr) )* linear_velocity
            
    def eval_state(self):
        #check if the new state of the robot should truncate, terminate and/or reward
        
        oldprog= self.prog #records progress made on last step, 0 on reset
        
        self.md = self.c1l + (self.c2l + self.chw) + (self.c3l + self.chw) -self.goal_size
        #self.md = self.c1l
        
        #check if timelimit is reached
        if self.envStepCounter > self.timelimit: #how many steps before truncation?
            self.truncated = True
        
        else:
            self.truncated = False
            
            #if not truncated, calculate reward based on distance travelled towards goal
            #prog = dist traveled in axis from begining of corridor
            #the reward of this iteration is the dist travelled since last step so prog - oldprog
            
            #first corridor calc
            if (-self.chw +self.robot_radius < self.robotPos[0] < self.chw -self.robot_radius
                and 0 + self.robot_radius < self.robotPos[1] < self.c1l
            ):
                self.prog = self.robotPos[1]
            
            elif (-(self.chw + self.c2l + self.robot_radius) < self.robotPos[0] < (self.chw + self.c2l + self.robot_radius)
                and self.c1l < self.robotPos[1] < (self.chw*2 + self.c1l -self.robot_radius)
                ):
                
                self.prog = abs(self.robotPos[0])
                
                #self.terminated = True
                
                #wrong turn
                if self.scn <= 2:
                    if self.robotPos[0] > self.chw:
                        self.truncated = True
                        self.prog *= -1
                    elif self.robotPos[0] > 0:
                        self.prog = 0
                
                elif self.scn >= 3:
                    if self.robotPos[0] < -self.chw:
                        self.truncated = True
                        self.prog *= -1
                    elif self.robotPos[0] < 0:
                        self.prog = 0
                
                #account for transition between corridors
                if self.prog > (self.c2l+self.chw):
                    self.prog = self.c2l+self.chw
                
                #add previous corridor    
                self.prog += self.c1l 
            
            elif (self.c2l + self.chw + self.robot_radius < abs(self.robotPos[0]) < self.c2l + self.chw *3 -self.robot_radius
                and self.c1l -self.c3l +self.robot_radius < self.robotPos[1] < self.c1l + self.c3l +self.chw *2 -self.robot_radius
                ):
                    
                if self.scn == 1 or self.scn == 4:
                    
                    self.prog = abs( self.c1l + self.chw - self.robotPos[1] )
                    
                    if self.robotPos[1] < self.c1l - self.c3l + self.goal_size: #goal is found
                        self.terminated = True
                        #bonus = 1
                        
                    elif self.robotPos[1] > self.c1l+self.chw*2: #wrong turn
                        self.truncated = True
                    
                    elif self.robotPos[1] > self.c1l+self.chw:
                        self.prog = 0
                        
                elif self.scn == 2 or self.scn == 3:
                    
                    self.prog = abs(self.robotPos[1]-(self.c1l+self.chw) )
                    
                    if self.robotPos[1] > self.c1l + self.c3l +self.chw*2 -self.goal_size: #goal found
                        self.terminated = True
                        #bonus = 1
                        
                    elif self.robotPos[1] < self.c1l: #wrong turn
                        self.truncated = True
                    
                    elif self.robotPos[1] < self.c1l + self.chw:
                        self.prog = 0
                            
                """#junction
                if self.c1l < y < self.c1l + self.chw*2:
                    self.prog = 0
                #"""
                    
                #add privious corridors    
                self.prog += self.c1l + (self.c2l + self.chw)
                
            else: #out of maze
                self.truncated = True
                #punishment = -1
        
        #Halt stalling agents
        self.prog /= self.md
        
        if self.prog <= oldprog:
            self.timer += 1
            
        if self.timer > 50:
            self.truncated = True
            #print('stalling')
            #punishment += 1
        
        #calc step reward
        if self.truncated: #avoids negative rew
            self.reward = 0
        
        elif self.terminated: #makes sure reward is exactly 1 on termination
            self.reward = 1 - oldprog
            
        else:
            #print('neither')
            self.reward = self.prog - oldprog # + bonus + punishment
        
        #calc ep reward
        self.rsum += self.reward
        
        #update rew peak
        if self.rsum > self.rpeak + 0.05:
            self.rpeak = self.rsum
            self.timer = 0
   
    def updateIR(self):
        #initializes (and resets) list of IR activation for each wall
        IRvs_list = []
        
        #normalize robot radius and IR ranges to standard scale, needed to match file
        IRmin = self.IRmin / self.scale
        IRmax = self.IRmax / self.scale
        
        for wall in range ( len (self.wall_matrix)):
            
            self.robotPos = [float(self.robotPos[0]), float(self.robotPos[1])]
            
            if self.wall_matrix[wall][0] == self.wall_matrix[wall][2]:
                
                #nearest point within wall to robot
                if self.wall_matrix[wall][1] < self.robotPos[1] < self.wall_matrix[wall][3]:
                    point = [float(self.wall_matrix[wall][0]), self.robotPos[1]]
                
                #in case nearest point is the edge of the wall
                else:
                    point = None
            elif self.wall_matrix[wall][1] == self.wall_matrix[wall][3]:
                
                if self.wall_matrix[wall][0] < self.robotPos[0] < self.wall_matrix[wall][2]:
                    point = [self.robotPos[0], float(self.wall_matrix[wall][1])]
                    
                else:
                    point = None
            
            if point != None:
                
                vector = [point[0] - self.robotPos[0], point[1] - self.robotPos[1]]
                
                dist = math.sqrt( (vector[0])**2 +(vector[1])**2)
                
                dist -= self.robot_radius
                
                #print(wall, dist, self.IRmax)
                
                dist /= self.scale
                #normalize dist to standard scale, needed to match file
                
                #if wall within sensor range, calc relative angle and IR activation
                if dist < IRmax:
                    
                    relang = math.atan2(vector[1], vector[0])- self.robotOr #- math.pi #-pi is to match pygame referential
                    
                    relang %= math.pi*2
                    while relang > math.pi *2:
                        relang -= math.pi * 2
                    while relang < 0:
                        relang += math.pi * 2
                        
                    #calc within which listed angles is actual angle
                    #if within n and n+1, gets ref for closest
                    
                    degrees = relang * 180 / math.pi
                    
                    aref = abs (int ( Decimal( (degrees)).to_integral_value(rounding=ROUND_HALF_UP) / 2))
                    
                    if aref == 180:
                        aref = 0
                        
                    #if dist <= IRmin, reference values for IRmin
                    if dist < IRmin:
                        dref = 0
                        ix = 1 + (dref+1) + dref*180 + aref
                        #add one to move from file header. #add dref+1 to move from header of each block (=dref+2)
                        #add dref * 180 to move all lines of each unwanted block
                        #add aref to move nr of lines in wanted block
                        IRvs =  self.IRwall[ix]
                    
                    #else, calc within which listed values is actual dist
                    #if within n and n+1, gets ref for closest
                    else:
                        dref = int ( Decimal( (dist-IRmin)*10 /2).to_integral_value(rounding=ROUND_HALF_UP))
                        #to match files dists have to be converted to mms
                        m = int ( Decimal( (IRmax - IRmin)*10 /2 -1).to_integral_value(rounding=ROUND_HALF_UP))
                        if dref >= m:
                            dref = m #if dist is exactly the IR range, pick last values from file.
                        ix = 1 + (dref+1) + dref*180 + aref
                        #print(dist, IRmax, degrees, dref, aref)
                        IRvs =  self.IRwall[ix]
                
                else:
                    IRvs = [0.0]*8
                    
            else:
                IRvs = [0.0]*8
            
            IRvs_list.append(IRvs) 
            
        for cyl in self.cylinder_matrix:
            
            vector = [cyl[0] - self.robotPos[0], cyl[1] - self.robotPos[1]]
            
            dist = math.sqrt( (vector[0])**2 +(vector[1])**2)
            
            dist -= (self.robot_radius + self.cylr)
            
            dist /= self.scale
            
            #if cyl within sensor range, calc relative angle and IR activation
            if dist < IRmax:
                
                relang = math.atan2(vector[1], vector[0])- self.robotOr #- math.pi
                
                relang %= math.pi*2
                while relang > math.pi *2:
                    relang -= math.pi * 2
                while relang < 0:
                    relang += math.pi * 2
                    
                #calc within which listed angles is actual angle
                #if within n and n+1, gets ref for closest
                
                degrees = relang * 180 / math.pi
                
                aref = abs (int ( Decimal( (degrees)).to_integral_value(rounding=ROUND_HALF_UP) / 2))
                
                if aref == 180:
                    aref = 0
                    
                #if dist <= IRmin, reference values for IRmin
                if dist < IRmin:
                    dref = 0
                    ix = 1 + (dref+1) + dref*180 + aref
                    #add one to move from file header. #add dref+1 to move from header of each block (=dref+2)
                    #add dref * 180 to move all lines of each unwanted block
                    #add aref to move nr of lines in wanted block
                    IRvs =  self.IRcylinder[ix]
                
                #else, calc within which listed values is actual dist
                #if within n and n+1, gets ref for closest
                else:
                    dref = int ( Decimal( (dist-IRmin)*10 /2).to_integral_value(rounding=ROUND_HALF_UP))
                    #to match files dists have to be converted to mms
                    m = int ( Decimal( (IRmax - IRmin)*10 /2 -1).to_integral_value(rounding=ROUND_HALF_UP))
                    if dref >= m:
                        dref = m #if dist is exactly the IR range, pick last values from file.
                    ix = 1 + (dref+1) + dref*180 + aref
                    IRvs =  self.IRcylinder[ix]
            
            else:
                IRvs = [0.0]*8
            
            IRvs_list.append(IRvs)
        
        #print('walls: ', IRvs_list[:-6])
        #print('cylinders: ', IRvs_list[-6:])
        
        #sums the activation caused by each wall/cyl for each sensor
        IR_obs = [sum(x) for x in zip(*IRvs_list)]
        
        for i in range ( len (IR_obs)):
            
            IR_obs[i] += rd.randrange(-5,5,1)/100
            
            if IR_obs[i] > 1.0:
                IR_obs[i] = 1.0
        
        return IR_obs
    
    def Cam(self, color: str = 'green'):
        
        Cam_obs = [0.0, 0.0, 0.0, 0.0]
    
        if color == 'green':
            if abs(self.robotPos[0]) > abs(self.chw):
                return Cam_obs
    
            for beacon in self.beacons_on:
                top = [self.beacon_matrix[beacon][0], self.beacon_matrix[beacon][1] + self.beacon_radius]
                bot = [self.beacon_matrix[beacon][0], self.beacon_matrix[beacon][1] - self.beacon_radius]
    
                Cam_obs = self._process_beacon_angles(top, bot, Cam_obs)
    
        elif color == 'blue':
            if abs(self.robotPos[0]) < self.chw + self.c2l:
                return Cam_obs
    
            # Blue beacon assumed single beacon
            right = [self.blue_beacon[0] + self.beacon_radius, self.blue_beacon[1]]
            left = [self.blue_beacon[0] - self.beacon_radius, self.blue_beacon[1]]
    
            Cam_obs = self._process_beacon_angles(right, left, Cam_obs)
    
        else:
            raise ValueError(f"Unsupported color: {color}")
    
        # Normalize, noise, clip
        for i in range(len(Cam_obs)):
            Cam_obs[i] = Cam_obs[i] / self.max_beacon
            Cam_obs[i] += rd.randrange(-50, 51, 1) / 10_000.0
            Cam_obs[i] = max(0.0, min(1.0, Cam_obs[i]))
    
        return Cam_obs

    def _process_beacon_angles(self, pt1, pt2, Cam_obs):
        
        topvec = [pt1[0] - self.robotPos[0], pt1[1] - self.robotPos[1]]
        botvec = [pt2[0] - self.robotPos[0], pt2[1] - self.robotPos[1]]
    
        topang = math.atan2(topvec[1], topvec[0]) + math.pi + self.robotOr
        botang = math.atan2(botvec[1], botvec[0]) + math.pi + self.robotOr
        
        rg = [min(topang, botang), max(topang, botang)]
        
        while rg[0] > math.pi *2:
            rg[0] -= math.pi * 2
        while rg[0] < 0:
            rg[0] += math.pi * 2
            
        while rg[1] > math.pi *2:
            rg[1] -= math.pi * 2
        while rg[1] < 0:
            rg[1] += math.pi * 2
        
        range_start, range_end = rg
        
        for sector in range(len(self.camrange)):
            
            sector_start = math.pi / 2 * sector
            sector_end = math.pi / 2 * (sector + 1)
            
            if range_end >= sector_start and range_start <= sector_end:
                cross = [
                    max(range_start, sector_start),
                    min(range_end, sector_end)
                ]
                Cam_obs[sector] += abs(cross[1] - cross[0]) / (sector_end - sector_start)
        
        return Cam_obs
    
    def scanner(self):
        
        beam_activations = [0]*len(self.scannerOrs)
        
        scanner_obs = [0] * self.neurs #neurs = nr of neurons, make sure len(self.scannerOrs) % neurs == 0
        
        for beam in self.scannerOrs:
            
            #find beam or
            a = beam * math.pi/180 + self.robotOr
            
            #normalize a
            a %= math.pi*2
            while a > math.pi *2:
                a -= math.pi * 2
            while a < 0:
                a += math.pi * 2
            
            #find beam end point
            b = [self.robotPos[0] + (self.robot_radius + self.scanner_range) * math.cos(a),
                 self.robotPos[1] + (self.robot_radius + self.scanner_range) * math.sin(a)]
            
            for wall in self.wall_matrix:
                    
                if wall[0] == wall[2]:
                    
                    b2 = [wall[0],
                          b[1]] #find projection of b in infinite wall, b2
                    
                    d = b[0] - b2[0] #distance between b and b2
                    
                    teta = math.atan2(b2[1], b2[0])
                    
                    hipo = (d / math.cos(a)) #hipotenuse of the triangle b, b2, cross
                    
                    q = (math.sin(a) * hipo) #distance between b2 and cross
                    
                    cross = [b2[0], b2[1]-q] #coords of cross
                    
                    if b[0] <= self.robotPos[0]:
                        bound1 = b[0]
                        bound2 = self.robotPos[0]
                        
                    elif b[0] > self.robotPos[0]: 
                        bound1 = self.robotPos[0]
                        bound2 = b[0]
                        
                    if bound1 < cross[0] < bound2 and wall[1] < cross[1] < wall[3]: #check if cross within finite wall limits
                        
                        dist = math.sqrt( (self.robotPos[0] - cross[0])**2 + (self.robotPos[1] - cross[1])**2) - self.robot_radius
                        
                        #dist /= self.scale
                        #print('dist: ', dist, dist/self.scanner_range, 1- dist/self.scanner_range)
                        activation = 1 - dist / self.scanner_range
                    
                    else: activation = 0
                    
                    if activation > beam_activations[self.scannerOrs.index(beam)]:
                        beam_activations[self.scannerOrs.index(beam)] = activation #add beam activation to sensor value
                        
                elif wall[1] == wall[3]:
                    
                    b2 = [b[0],
                          wall[1]] #find projection of b in infinite wall, b2
                    
                    d = b[1] - b2[1] #distance between b and b2
                    
                    teta = math.pi/2 - a #math.atan2(b2[1], b2[0]) #+ math.pi/2
                    
                    hipo = (d / math.cos(teta)) #hipotenuse of the triangle
                    
                    q = (math.sin(teta) * hipo) #distance between b2 and cross
                    
                    cross = [b2[0] -q, b2[1]] #coords of cross
                    
                    if b[1] <= self.robotPos[1]:
                        bound1 = b[1]
                        bound2 = self.robotPos[1]
                        
                    elif b[1] > self.robotPos[1]: 
                        bound1 = self.robotPos[1]
                        bound2 = b[1]
                        
                    if bound1 < cross[1] < bound2 and wall[0] < cross[0] < wall[2]: #check if cross within finite wall limits
                        
                        dist = math.sqrt( (self.robotPos[0] - cross[0])**2 + (self.robotPos[1] - cross[1])**2) - self.robot_radius
                        
                        #dist /= self.scale
                        #print('dist: ', dist, dist/self.scanner_range, 1- dist/self.scanner_range)
                        activation = 1 - dist / self.scanner_range
                    
                    else: activation = 0
                        
                    if activation > beam_activations[self.scannerOrs.index(beam)]:
                        beam_activations[self.scannerOrs.index(beam)] = activation
        
        for i in range(self.neurs): #get average of beams within sector
            scanner_obs[i] = sum(beam_activations[i*self.n_beams:(i+1)*self.n_beams]) / self.n_beams 
            scanner_obs[i] += rd.randrange(-50, 51, 1) / 10_000 #add noise
            if scanner_obs[i] < 0.: scanner_obs[i] = 0.
            if scanner_obs[i] > 1.: scanner_obs[i] = 1.
            
        return scanner_obs
        
    def observe(self):
            
        IR_obs = self.updateIR()

        green_obs = self.Cam()
        
        blue_obs = self.Cam('blue')
        
        Scanner_obs = self.scanner()
            
        obs = IR_obs + green_obs + blue_obs + Scanner_obs
        
        obs = np.float32(obs)
        
        #print('OBS: ', obs)
        
        return obs
    
    def render(self):
        
        import pygame
        
        # Set up screen
        sw = 850
        sh = 700
        
        def conv(pt, sw=850, sh=700):
            #adjusts coords to pygame referential
            new = [0,0]
            new[0] = pt[0] + sw / 2
            new[1] = -(pt[1] -sh + 25)
            return new
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode([sw, sh], pygame.RESIZABLE)
                
        # Fill the background with white
        self.screen.fill((255, 255, 255))
    
        #draw the walls
        for wall in self.wall_matrix:
            pygame.draw.line(self.screen, (0,0,0), conv(wall[0:2]), conv(wall[2:4]), 1)
        
        #draw the corners
        for cyl in self.cylinder_matrix:
            pygame.draw.circle(self.screen, (0,0,0), conv(cyl[0:2]), self.cylr, 1)
        
        #draw the beacons
        for beacon in self.beacons_on:
            top = [self.beacon_matrix[beacon][0], self.beacon_matrix[beacon][1] + self.beacon_radius]
            bot = [self.beacon_matrix[beacon][0], self.beacon_matrix[beacon][1] - self.beacon_radius]
            pygame.draw.line(self.screen, (0,255,0), conv(bot), conv(top), 1)
        
        #draw blue beacons
        right = [self.blue_beacon[0] + self.beacon_radius, self.blue_beacon[1]]
        left = [self.blue_beacon[0] - self.beacon_radius, self.blue_beacon[1]]
        pygame.draw.line(self.screen, (0,0,255), conv(right), conv(left), 3)
        
        """#draw the goals
        pygame.draw.line(self.screen, (0,0,255), conv(self.goal_matrix[self.scn-1][0:2]), conv(self.goal_matrix[self.scn-1][2:4]), 1)
        """
        
        """#draw initial posotion area
        ix, iy = conv([(0 -self.ix_noise), (self.iy_noise*2 +self.iy_noise)])
        pygame.draw.rect(self.screen, (255,0,0), pygame.Rect(ix, iy, (2*self.ix_noise), (2*self.iy_noise)),  2)
        """
        
        #draw robot
        x, y = conv(self.robotPos) #get robot coords
        
        #drawm the scanner beams
        for beam in self.scannerOrs:
            
            scanner_a = beam * math.pi/180 + self.robotOr
            
            scanner_b = [self.robotPos[0] + (self.robot_radius + self.scanner_range) * math.cos(scanner_a),
                 self.robotPos[1] + (self.robot_radius + self.scanner_range) * math.sin(scanner_a)]
            
            pygame.draw.line(self.screen, (0,0,255), conv(scanner_b), [x,y], 1)
            
        #draw the robot's body
        pygame.draw.circle(self.screen, (0, 0, 0), (x,y), self.robot_radius)
        
        #draw robot's camera fostrums
        for i in range( len (self.camrange)):
            a = self.camrange[i][0] + self.robotOr ###+ ORI ????
            cx = x + self.robot_radius * math.cos(a)
            cy = y + self.robot_radius * math.sin(a)
            if i != 0:
                pygame.draw.line(self.screen, (0,255,0), (x,y), [cx,cy], 1)
            else:
                #mark front of the robot
                pygame.draw.line(self.screen, (255,255,0), (x,y), [cx,cy], 1)
            
        #draw robot's IRs
        
        for i in range(24):
            a = (math.pi *2 /24) * i + self.robotOr
            x0 = x + (self.robot_radius + self.IRmax) * math.cos(a) 
            y0 = y + (self.robot_radius + self.IRmax) * math.sin(a)
            x1 = x + (self.robot_radius + self.IRmin) * math.cos(a)
            y1 = y + (self.robot_radius + self.IRmin) * math.sin(a)
            if i == 0:
                pygame.draw.line(self.screen, (255,0,255), [x1,y1], [x0,y0], 3)
            elif i == 23:
                pygame.draw.line(self.screen, (255,0,255), [x1,y1], [x0,y0], 1)
            elif i % 3 == 0:
                pygame.draw.line(self.screen, (255,0,0), [x1,y1], [x0,y0], 3)
            else:
                pygame.draw.line(self.screen, (255,100,100), [x1,y1], [x0,y0], 1)
        
        pygame.draw.circle(self.screen, (255, 0, 0), (x,y), self.robot_radius+self.IRmax, 1)
        
        pygame.display.flip()
        pygame.event.pump()
        
        if self.record != None:
            # Capture the screen and store the frame
            frame = pygame.surfarray.array3d(self.screen)
            # Convert from (width, height, 3) to (height, width, 3) because pygame's coordinate system is different
            frame = frame.transpose((1, 0, 2))
            self.frames.append(frame)
    
    def step(self, action):
        
        #get the consequences of action
        #action is the output of the neural network controller
        self.move(action)
        
        #evaluete consequences (terminte, truncate and reward)
        self.eval_state()
            
        #get the new state to feed the controller
        self.observation = self.observe()
        
        #log
        if self.terminated or self.truncated:
            self.SCORES.append(self.rsum)
            
        self.envStepCounter += 1
        
        self.info["position"] = self.robotPos
        self.info["orientation"] = self.robotOr
        self.info["iteration"] = self.envStepCounter
        
        if self.render_mode == 'human':
            self.render()
        
        return self.observation, self.reward, self.terminated, self.truncated, self.info
    
    def save_video(self, video_filename= "seqpredprey.mp4" , fps=120):
        from moviepy.editor import ImageSequenceClip
    
        if hasattr(self, 'frames') and self.frames:
            # Create a video from the stored frames
            clip = ImageSequenceClip(self.frames, fps=fps)
            clip.write_videofile(video_filename, codec='libx264')
        else:
            print("No frames to save as video.")
            
    def close(self):
        if self.record != None:
            self.save_video(f'{self.record}.mp4')
            
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

