######################################################################################################################
#
# Test function environment (adapted from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
#
# The contained test functions have been taken from https://arxiv.org/pdf/1308.4008
#
######################################################################################################################

"""
   This file belongs to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
   It includes an implementation of the test function optimization problem used for the experiments reported in the following paper:

   Pagliuca P. (2026). Whether, how and when modern evolutionary strategies can be improved. Journal of Artificial Intelligence and Autonomous Intelligence, vol. 2, issue 3, pp. 425-450.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import sys
import math
from typing import Optional, Tuple, Union

# Length of the vector
VEC_LEN = 1

# Custom floor function
def customFloor(num):
    # Round number to third digit
    numr = round(num, 3)
    # Compute difference and return the number
    diff = num - numr
    if diff >= 0.0:
        return numr
    else:
        return round(numr - 0.001, 3)

class customEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode: Optional[str] = None, options: Optional[dict] = None):
        self.window_size = 480 # Pygame window size
        # Dictionary of the available test functions
        self.funct_dict = {'ackley': self.ackley, 'alpine1': self.alpine1, 'alpine2': self.alpine2, 'chungreynolds': self.chungreynolds, 'cosinemixture': self.cosinemixture, 'csendes': self.csendes, 'deb1': self.deb1, 'deb3': self.deb3, 'dixonproce': self.dixonprice, 'eggholder': self.eggholder, 'exponential': self.exponential, 'griewank': self.griewank, 'michalewicz': self.michalewicz, 'qing': self.qing, 'quartic': self.quartic, 'quintic': self.quintic, 'rana': self.rana, 'rastrigin': self.rastrigin, 'rosenbrock': self.rosenbrock, 'salomon': self.salomon, 'schafferf6': self.schafferF6, 'schumersteiglitz': self.schumersteiglitz, 'schwefel': self.schwefel, 'schwefel12': self.schwefel12, 'schwefel24': self.schwefel24, 'schwefel220': self.schwefel220, 'schwefel222': self.schwefel222, 'schwefel223': self.schwefel223, 'schwefel226': self.schwefel226, 'shubert3': self.shubert3, 'shubert4': self.shubert4, 'sphere': self.sphere, 'step1': self.step1, 'step2': self.step2, 'step3': self.step3, 'stepint': self.stepint, 'stretchedvsinewave': self.stretchedvsinewave, 'styblinskitang': self.styblinskitang, 'sumsquares': self.sumsquares, 'trid': self.trid, 'trigonometric2': self.trigonometric2, 'whitley': self.whitley, 'xinsheyang2': self.xinsheyang2, 'xinsheyang3': self.xinsheyang3, 'xinsheyang4': self.xinsheyang4, 'zakharov': self.zakharov}
        self.vec_len = VEC_LEN
        self.funct_name = 'ackley'
        self.funct = None
        if options is not None:
            # Get the length of the vector
            try:
                self.vec_len = int(options['vec_len'])
            except:
                pass
            # Get the function name
            try:
                self.funct_name = options['funct_name']
            except:
                pass
        
        # Verify if the function name corresponds to an existing function and get the "pointer"      
        try:
            self.funct = self.funct_dict[self.funct_name]
        except:
            # The function name does not exist -> print the available functions and set the Ackley function as default
            print(f"Unknown function with name {self.funct_name}")
            self.printFuncts()
            self.funct = self.funct_dict['ackley']
                
        self.params = None
        self.nparams = 0
        
        self.reward = None
        
        # Observation and action are useless
        high = np.array([np.inf] * self.vec_len, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        high = np.array([np.inf], dtype=np.float32)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def noNetEnv(self):
        return True
        
    def printFuncts(self):
        print("Available functions:")
        for key in self.funct_dict.keys():
            print(f"- {key}")
        
    def setParams(self, params):
        self.params = params
        self.nparams = len(params)

    def reset(self, seed=None):
        # Useless
        observation = None#np.zeros(self.vec_len, dtype=np.float32)
        info = {}

        return observation, info
        
    def ackley(self):
        a = 20.0
        b = 0.2
        c = math.pi * 2.0
        f_sum = 0.0
        f_sum_2 = 0.0
        for i in range(self.nparams):
            f_sum += self.params[i]**2
            f_sum_2 += math.cos(c * self.params[i])
        f_out = -a * math.exp(-b * math.sqrt(f_sum / self.nparams)) - math.exp(f_sum_2 / self.nparams) + a + math.exp(1.0)
        return f_out
        
    def alpine1(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += abs(self.params[i] * math.sin(self.params[i]) + 0.1 * self.params[i])
        return f_out
        
    def alpine2(self):
        f_out = 1.0
        for i in range(self.nparams):
            f_out *= math.sqrt(abs(self.params[i])) * math.sin(self.params[i])
        return f_out
        
    def chungreynolds(self):
        return self.sphere()**2
        
    def cosinemixture(self):
        f_sum = 0.0
        for i in range(self.nparams):
            f_sum += math.cos(5.0 * math.pi * self.params[i])
        f_out = -0.1 * f_sum - self.sphere()
        return f_out
        
    def csendes(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (self.params[i]**6) * (2 + math.sin(1 / self.params[i]))
        return f_out
        
    def deb1(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (math.sin(5.0 * math.pi * self.params[i]))**6
        return -f_out / self.nparams
        
    def deb3(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (math.sin(5.0 * math.pi * (self.params[i]**(3/4) - 0.05)))**6
        return -f_out / self.nparams
        
    def dixonprice(self):
        f_out = (self.params[0] - 1.0)**2
        i = 1
        while i < self.nparams:
            f_out += ((i + 1) * (2.0 * self.params[i]**2 - self.params[i - 1])**2)
            i += 1
        return f_out

    def eggholder(self):
        f_out = 0.0
        for i in range(self.nparams - 1):
            f_out += (-(self.params[i + 1] + 47.0) * math.sin(math.sqrt(abs(self.params[i + 1] + self.params[i] / 2.0 + 47.0))) - self.params[i] * math.sin(math.sqrt(abs(self.params[i] - (self.params[i + 1] + 47.0)))))
        return f_out
        
    def exponential(self):
        return -math.exp(-0.5 * self.sphere())
    
    def griewank(self):
        f_sum = 0.0
        f_prod = 1.0
        for i in range(self.nparams):
            f_sum += self.params[i]**2
            f_prod *= (math.cos(self.params[i] / math.sqrt(i + 1)))
        f_out = 1.0 + f_sum / 4000.0 - f_prod
        return f_out
        
    def michalewicz(self):
        m = 10
        f_sum = 0.0
        for i in range(self.nparams):
            f_sum += math.sin(self.params[i]) * (math.sin(((i + 1) * self.params[i]**2) / math.pi))**(2 * m)
        f_out = -f_sum
        return f_out
        
    def qing(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (self.params[i]**2 - (i + 1))**2
        return f_out
        
    def quartic(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (i + 1) * self.params[i]**4
        return f_out
        
    def quintic(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += abs(self.params[i]**5 - 3.0 * self.params[i]**4 + 4.0 * self.params[i]**3 + 2.0 * self.params[i]**2 - 10.0 * self.params[i] - 4.0)
        return f_out
        
    def rana(self):
        f_out = 0.0
        for i in range(self.nparams - 1):
            t1 = math.sqrt(abs(self.params[i + 1] + self.params[i] + 1))
            t2 = math.sqrt(abs(self.params[i + 1] - self.params[i] + 1))
            f_out += (self.params[i + 1] + 1) * math.cos(t2) * math.sin(t1) + self.params[i] * math.cos(t1) * math.sin(t2)
        return f_out
    
    def rastrigin(self):
        f_out = 10.0 * self.nparams
        for i in range(self.nparams):
            f_out += (self.params[i]**2 - 10.0 * math.cos(2.0 * math.pi * self.params[i]))
        return f_out
    
    def rosenbrock(self):
        f_out = 0.0
        for i in range(self.nparams - 1):
            f_out += 100.0 * ((self.params[i + 1] - self.params[i]**2)**2 + ((1.0 - self.params[i])**2))
        return f_out
        
    def salomon(self):
        return (1.0 - math.cos(2.0 * math.pi * math.sqrt(self.sphere())) + 0.1 * math.sqrt(self.sphere()))
        
    def schafferF6(self):
        f_out = 0.0
        for i in range(self.nparams - 1):
            f_out += (0.5 + (math.sin(math.sqrt(self.params[i]**2 + self.params[i + 1]**2))**2 - 0.5) / ((1.0 + 0.001 * (self.params[i]**2 + self.params[i + 1]**2))**2))
        return f_out
        
    def schumersteiglitz(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += self.params[i]**4
        return f_out
        
    def schwefel(self):
        alpha = 5 # It can be a parameter
        f_out = self.sphere()**alpha
        return f_out
        
    def schwefel12(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_sum = 0.0
            for j in range(i):
                f_sum += self.params[i]
            f_out += f_sum**2
        return f_out
        
    def schwefel24(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += ((self.params[i] - 1.0)**2 + (self.params[i] - self.params[i]**2)**2)
        return f_out
        
    def schwefel220(self):
        f_out = 0.0
        f_sum = 0.0
        for i in range(self.nparams):
            f_sum += abs(self.params[i])
        f_out = -f_sum
        return f_out
        
    def schwefel222(self):
        f_out = 0.0
        f_sum = 0.0
        f_prod = 1.0
        for i in range(self.nparams):
            f_sum += abs(self.params[i])
            f_prod *= abs(self.params[i])
        f_out = f_sum + f_prod
        return f_out
        
    def schwefel223(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += self.params[i]**10
        return f_out
        
    def schwefel226(self):
        f_out = 0.0
        f_sum = 0.0
        for i in range(self.nparams):
            f_sum += self.params[i] * math.sin(math.sqrt(abs(self.params[i])))
        f_out = -f_sum / self.nparams
        return f_out
        
    def shubert3(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_sum = 0.0
            j = 1
            while j <= 5:
                f_sum += j * math.sin((j + 1) * self.params[i] + j)
                j += 1
            f_out += f_sum
        return f_out
        
    def shubert4(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_sum = 0.0
            j = 1
            while j <= 5:
                f_sum += j * math.cos((j + 1) * self.params[i] + j)
                j += 1
            f_out += f_sum
        return f_out
    
    def sphere(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += self.params[i]**2
        return f_out
        
    def step1(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += customFloor(abs(self.params[i]))
        return f_out
    
    def step2(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (customFloor(self.params[i] + 0.5))**2
        return f_out
    
    def step3(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += customFloor(self.params[i]**2)
        return f_out
    
    def stepint(self):
        f_out = 25.0
        for i in range(self.nparams):
            f_out += customFloor(self.params[i])
        return f_out
    
    def stretchedvsinewave(self):
        f_out = 0.0
        for i in range(self.nparams - 1):
            f_1 = (self.params[i + 1]**2 + self.params[i]**2)**0.25
            f_2 = (math.sin(50 * (self.params[i + 1]**2 + self.params[i]**2)**0.1))**2
            f_out += (f_1 + f_2 + 0.1)
        return f_out
        
    def styblinskitang(self):
        f_out = 0.0
        f_1 = 0.0
        f_2 = 0.0
        f_3 = 0.0
        for i in range(self.nparams):
            f_1 = self.params[i]**4
            f_2 = -16.0 * self.params[i]**2
            f_3 = 5.0 * self.params[i]
            f_out += (f_1 + f_2 + f_3)
        f_out /= 2.0
        return f_out
        
    def sumsquares(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_out += (i + 1) * self.params[i]**2
        return f_out
    
    def trid(self):
        f_sum = 0.0
        f_sum_2 = 0.0
        for i in range(self.nparams):
            f_sum += (self.params[i] - 1)**2
            if i >= 1:
                f_sum_2 += (self.params[i] * self.params[i - 1])
        f_out = f_sum - f_sum_2
        return f_out
    
    def trigonometric2(self):
        f_1 = 0.0
        f_2 = 0.0
        f_3 = 0.0
        for i in range(self.nparams):
            f_1 += 8.0 * (math.sin(7.0 * (self.params[i] - 0.9)**2))**2
            f_2 += 6.0 * (math.sin(14.0 * (self.params[i] - 0.9)**2))**2
            f_3 += (self.params[i] - 0.9)**2
        f_out = 1.0 + f_1 + f_2 + f_3
        return f_out
    
    def whitley(self):
        f_out = 0.0
        for i in range(self.nparams):
            f_sum = 0.0
            for j in range(self.nparams):
                f_1 = (100.0 * (self.params[i]**2 - self.params[j])**2 + (1.0 - self.params[j])**2)**2 / 4000.0
                f_2 = math.cos(100.0 * (self.params[i]**2 - self.params[j])**2 + (1.0 - self.params[j])**2 + 1.0)
                f_sum += f_1 - f_2
            f_out += f_sum
        return f_out
    
    def xinsheyang2(self):
        f_sum = 0.0
        f_sum_2 = 0.0
        for i in range(self.nparams):
            f_sum += abs(self.params[i])
            f_sum_2 += math.sin(self.params[i]**2)
        f_out = f_sum * math.exp(-f_sum_2)
        return f_out
    
    def xinsheyang3(self):
        m = 5
        beta = 15
        f_sum = 0.0
        f_sum_2 = 0.0
        f_prod = 1.0
        for i in range(self.nparams):
            f_sum += (self.params[i] / beta)**(2 * m)
            f_sum_2 += self.params[i]**2
            f_prod *= (math.cos(self.params[i]))**2
        f_out = math.exp(-f_sum) - 2.0 * math.exp(-f_sum_2) * f_prod
        return f_out
    
    def xinsheyang4(self):
        f_sum = 0.0
        f_sum_2 = 0.0
        f_sum_3 = 0.0
        for i in range(self.nparams):
            f_sum += (math.sin(self.params[i]))**2
            f_sum_2 += self.params[i]**2
            f_sum_3 += (math.sin(math.sqrt(abs(self.params[i]))))**2
        f_out = (f_sum - math.exp(-f_sum_2)) * math.exp(-f_sum_3)
        return f_out
        
    def zakharov(self):
        f_sum = 0.0
        f_sum_2 = 0.0
        for i in range(self.nparams):
            f_sum += self.params[i]**2
            f_sum_2 += 0.5 * (i + 1) * self.params[i]
        f_out = f_sum + f_sum_2**2 + f_sum_2**4
        return f_out
        
    def step(self, action):
        # Compute the function value and return the reward (-function)
        reward = -self.funct() # We want to maximize reward, but the test function need to be minimized so we invert the sign!!!
            
        # Get observation (useless)
        observation = None
        info = {}
        
        if self.render_mode == "human":
            # Store reward for rendering purposes (i.e., display the reward in a window)
            self.reward = reward
            self.render()
            
        return observation, reward, True, False, info

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        font = pygame.font.SysFont('Arial', 30)
        
        text = font.render(self.funct_name + ": " + str(round(self.reward,3)), True, (0, 0, 0), (0, 255, 0))

        # create a rectangular object for the
        # text surface object
        textRect = text.get_rect()

        # set the center of the rectangular object.
        textRect.center = (self.window_size // 2, self.window_size // 2)
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(text, textRect)
            pygame.event.pump()
            pygame.display.update()
            import time
            time.sleep(1)

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

