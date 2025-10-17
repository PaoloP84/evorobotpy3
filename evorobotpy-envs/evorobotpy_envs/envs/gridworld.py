######################################################################################################################
#
# Grid-world environment (adapted from https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
#
######################################################################################################################

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import sys
from typing import Optional, Tuple, Union

GRID_SIZE = 5

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, options: Optional[dict] = None):
        self.window_size = 512 # Pygame window size
        
        self.size = GRID_SIZE
        self.nobs = 0
        # Check values in options
        if options is not None:
            try:
                self.size = int(options['size'])
            except:
                pass
            try:
                self.nobs = int(options['nobs'])
            except:
                pass
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        """
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        """
        high = np.array([np.inf] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

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

    def _get_obs(self):
        proxy = 0.0
        if self._detect():
            proxy = 1.0
        oblist = []
        for elem in self._agent_location:
            oblist.append(2.0 * (elem / (self.size - 1)) - 1.0)
        for elem in self._target_location:
            oblist.append(2.0 * (elem / (self.size - 1)) - 1.0)
        obs = np.array(oblist, dtype=np.float32)
        obs = np.append(obs, proxy)
        return obs

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
        
    def _detect(self):
        found = False
        i = 0
        while i < self.nobs and not found:
            ob_location = self.obstacles[i]
            dist = np.linalg.norm(
                self._agent_location - ob_location, ord=1
            )
            if dist <= 2:
                found = True
            i += 1
        return found
        
    def _is_out(self):
        if self._agent_location[0] < 0 or self._agent_location[0] >= self.size or self._agent_location[1] < 0 or self._agent_location[1] >= self.size:
            return True
        return False
        
    def _collision(self):
        found = False
        i = 0
        while i < self.nobs and not found:
            ob_location = self.obstacles[i]
            found = np.array_equal(ob_location, self._agent_location)
            i += 1
        return found
        
    def _overlap(self, idx, location):
        found = False
        i = 0
        while i < idx and not found:
            if i != idx:
                ob_location = self.obstacles[i]
                if np.array_equal(ob_location, location):
                    found = True
            i += 1
        return found

    def reset(self, seed=None):
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
            
        # Obstacles
        self.obstacles = []
        for i in range(self.nobs):
            ob_location = self._agent_location
            while np.array_equal(ob_location, self._agent_location) or np.array_equal(ob_location, self._target_location) or self._overlap(i, ob_location):
                ob_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles.append(ob_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # Update agent location
        self._agent_location += direction
        reward = 0.0
        bonus = 0.0
        penalty = 0.0
        # Premature end if agent either goes out of the grid or collides with an obstacle (if any)
        terminated = False
        truncated = False
        if self._is_out() or self._collision():
            truncated = True
            penalty = 1.0
        else:
            # An episode is done if the agent has reached the target
            terminated = np.array_equal(self._agent_location, self._target_location)
            if terminated:
                bonus = 1.0
        reward = bonus - penalty
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

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
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Then we draw the obstacles (if any)
        for i in range(self.nobs):
            ob_location = self.obstacles[i]
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * ob_location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
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

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

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
