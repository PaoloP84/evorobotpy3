######################################################################################################################
#
# N bit-parity environment (adapted from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
#
######################################################################################################################

"""
   This file belongs to https://github.com/PaoloP84/evorobotpy3
   and has been written by Paolo Pagliuca, paolo.pagliuca@istc.cnr.it
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import sys
import math
from typing import Optional, Tuple, Union

NUM_BITS = 1

class customEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode: Optional[str] = None, options: Optional[dict] = None):
        self.window_size = 480 # Pygame window size
        self.nbits = NUM_BITS
        if options is not None:
            # Get the number of bits (if any)
            try:
                self.nbits = int(options['nbits'])
            except:
                pass
        # Compute the number of bit-strings
        self.nbitstrings = 2 ** self.nbits
        # Compute the strings
        self.bitstrings = self.get_all_bit_strings(self.nbits)
        # Parity output
        self.parity = None
        # Counter
        self.cnt = 0
        
        high = np.array([np.inf] * self.nbits, dtype=np.float32)
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
        
    def get_all_bit_strings(self, nbits):
        bit_strings = []
        for i in range(self.nbitstrings):
            bin_string = bin(i)[2:].zfill(nbits)
            bit_strings.append(bin_string)
        return bit_strings

    def _get_obs(self):
        obs = np.zeros(self.nbits, dtype=np.float32)
        # Check counter validity
        assert self.cnt >= 0 and self.cnt < self.nbitstrings, "Invalid value %d".format(cnt)
        # Get the bit-string associated to the counter
        bitstring = self.bitstrings[self.cnt]
        # Check consistency about the length of the bit-string
        assert len(bitstring) == self.nbits, "Mismatch between lengths!!!"
        # Fill the observation array
        bitSum = 0
        for i in range(self.nbits):
            obs[i] = int(bitstring[i])
            bitSum += obs[i]
        if bitSum % 2 == 0:
            self.parity = 1
        else:
            self.parity = 0
        return obs

    def reset(self, seed=None):
        # Get observation
        observation = self._get_obs()
        self.cob = observation
        info = {}

        self.cnt += 1
        if self.cnt == self.nbitstrings:
            self.cnt = 0

        return observation, info
        
    def step(self, action):
        assert len(action) == 1, "Inconsistent action length!!!"
        # Reward and terminated flag are initialized here
        reward = 0.0
        terminated = True # Parity task lasts one step only
        output = None
        if action >= 0.0:
            output = 1
        else:
            output = 0
        self.cac = output
        reward = 1.0 - abs(self.parity - output)              
                
        if self.render_mode == "human":
            self._render_frame()
            
        # Get observation (useless)
        observation = self._get_obs()
        info = {}
            
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
            
    def _render_frame(self):
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
        
        self.font = pygame.font.SysFont('Arial', 15)
        
        size = 20
        for x in range(self.nbits + 2):
            color = (255, 255, 255)
            if x == self.nbits:
                color = (0, 0, 255)
            if x == self.nbits + 1:
                if self.parity == self.cac:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    (int(self.window_size / 2) - (self.nbits - 1 - x) * size, int(self.window_size / 2) - size),
                    (size, size),
                ),
            )
            pygame.draw.line(
                canvas,
                0,
                (int(self.window_size / 2) - (self.nbits - 2 - x) * size, int(self.window_size / 2) - size),
                (int(self.window_size / 2) - (self.nbits - 1 - x) * size, int(self.window_size / 2) - size),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (int(self.window_size / 2) - (self.nbits - 2 - x) * size, int(self.window_size / 2)),
                (int(self.window_size / 2) - (self.nbits - 1 - x) * size, int(self.window_size / 2)),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (int(self.window_size / 2) - (self.nbits - 1 - x) * size, int(self.window_size / 2) - size),
                (int(self.window_size / 2) - (self.nbits - 1 - x) * size, int(self.window_size / 2)),
                width=2,
            )
        
        pygame.draw.line(
            canvas,
            0,
            (int(self.window_size / 2) + 3 * size, int(self.window_size / 2) - size),
            (int(self.window_size / 2) + 3 * size, int(self.window_size / 2)),
            width=2,
        )
            
        
        """
        # Draw the agents
        for loc, col in zip(self.locations, self.colors):
            pygame.draw.circle(
                canvas,
                col,
                (self.locToGrid(loc) + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        """
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            for x in range(self.nbits):
                self.window.blit(self.font.render(str(int(self.cob[x])), True, (0,0,0)), (int(self.window_size / 2) - (self.nbits - 1 - x) * size + int(size / 4), int(self.window_size / 2) - size + int(size / 4)))
            idx = 0
            for x in range(self.nbits, self.nbits + 2):
                if idx == 0:
                    self.window.blit(self.font.render(str(self.parity), True, (0,0,0)), (int(self.window_size / 2) - (self.nbits - 1 - x) * size + int(size / 4), int(self.window_size / 2) - size + int(size / 4)))
                else:
                    self.window.blit(self.font.render(str(self.cac), True, (0,0,0)), (int(self.window_size / 2) - (self.nbits - 1 - x) * size + int(size / 4), int(self.window_size / 2) - size + int(size / 4)))
                idx += 1
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

