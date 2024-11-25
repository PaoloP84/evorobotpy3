######################################################################################################################
#
# Game of Life environment
#
######################################################################################################################

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# Grid size
GRID_SIZE = 10

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

class GameOfLifeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, size=GRID_SIZE):
        self.size = size
        
        # Parameters from which the grid is derived!!!
        self.params = None
        self.grid = None
        
        # Observation space is an encoding for the grid
        # Number of observations
        self.ob_len = self.size ** 2
        # Actions and observations match (1 to 1 mapping)
        act = np.ones(self.ob_len, dtype=np.float32)
        high = np.array([np.inf] * self.ob_len, dtype=np.float32)
        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-high, high)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def noNetEnv(self):
        return True
        
    def setParams(self, params):
        assert len(params) == self.ob_len, "Inconsistent sizes!!!"
        self.params = params
        
    def reset(self, seed=None):
        # Create the grid
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.params[k] > 0.0:
                    self.grid[i,j] = 255
                k += 1
        # Observation is useless
        ob = np.zeros(self.ob_len, dtype=np.float32)
        return ob, {}
        
    def countNonZeros(self):
        num = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i,j] != 0.0:
                    num += 1
        return num
        
    def update(self): # Frame is ignored unless you are using animation
        grid = self.grid.copy()
        N = self.size
        # Update grid
        for a in range(N):
            for b in range(N):
                total = int((grid[a,(b-1)%N]+grid[a,(b+1)%N]+grid[(a-1)%N,b]+grid[(a+1)%N,b]+grid[(a-1)%N,(b-1)%N]+grid[(a-1)%N,(b+1)%N]+grid[(a+1)%N,(b-1)%N]+grid[(a+1)%N,(b+1)%N])/255)
                # Rules for game of life
                if grid[a, b] == 255:
                    if total < 2 or total > 3:
                        self.grid[a,b] = 0
                else:
                    if total == 3:
                        self.grid[a,b] = 255
        # Count the number of ONs (i.e., alive cells)
        num = self.countNonZeros()
        return num

    def step(self, action):
        # Count the number of alive cells
        aliveCells = self.update()
        # Check if there are no alive cells
        terminated = False
        if aliveCells == 0:
            terminated = True
            
        # Compute reward
        reward = float(aliveCells) / float(self.ob_len) # Maximum value is 1
        
        # Dummy observation
        ob = np.zeros(self.ob_len, dtype=np.float32)

        return ob, reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Draw grid
        blockSize = int(WINDOW_WIDTH / GRID_SIZE) #Set the size of the grid block
        i = 0
        for x in range(0, WINDOW_WIDTH, blockSize):
            j = 0
            for y in range(0, WINDOW_HEIGHT, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                if self.grid[i,j] == 0:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)
                else:
                    pygame.draw.rect(self.screen, (0, 255, 255), rect)
                j += 1
            i += 1
            
        if self.render_mode == "human":
            assert self.screen is not None
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )[:, -WINDOW_WIDTH:]
        
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
