"""
Implementation of the double-pole balancing problem (Wieland, 1991)
"""
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import configparser

class DoublePoleEnv(gym.Env):
    """
    ## Description

    This environment corresponds to the version of the double-pole problem described by Wieland (1991).
    Two poles are attached by un-actuated joints to a cart, which moves along a frictionless track.
    The pendulums are placed upright on the cart and the goal is to balance the poles by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(6,)` with the values corresponding to the following positions and velocities:

    | Num | Observation                 | Min                 | Max               |
    |-----|-----------------------------|---------------------|-------------------|
    | 0   | Cart Position               | -4.8                | 4.8               |
    | 1   | Cart Velocity               | -Inf                | Inf               |
    | 2   | Long pole Angle             | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Long pole Angular Velocity  | -Inf                | Inf               |
    | 4   | Short pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 5   | Short pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the poles upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('DoublePole-v0')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, markov: Optional[bool] = True, fixed: Optional[bool] = False, classic: Optional[bool] = False, long_poles: Optional[bool] = False):
        self.markov = markov
        self.fixed = fixed
        self.classic = classic
        self.long_poles = long_poles
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.masspole2 = 0.01
        if self.long_poles:
            self.masspole2 = 0.05
        self.total_mass = self.masspole + self.masspole2 + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.length2 = 0.05
        if self.long_poles:
            self.length2 = 0.25
        self.polemass_length = self.masspole * self.length
        self.polemass_length2 = self.masspole2 * self.length2
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 36 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
        # Read from config file        
        self.readConfig("config.ini")

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        if not self.markov:
            high = np.array(
                [
                    self.x_threshold * 2,
                    self.theta_threshold_radians * 2,
                    self.theta_threshold_radians * 2,
                ],
                dtype=np.float32,
            )

        act = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-act, act)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        
        self.episode = 0
        # Fixed states derived from Pagliuca, Milano and Nolfi (2018)
        self.fixedStates = [[-1.944, 0, 0, 0, 0, 0], [1.944, 0, 0, 0, 0, 0], [0, -1.215, 0, 0, 0, 0], [0, 1.215, 0, 0, 0, 0], [0, 0, -0.10472, 0, 0, 0], [0, 0, 0.10472, 0, 0, 0], [0, 0, 0, -0.135088, 0, 0], [0, 0, 0, 0.135088, 0, 0]]
        self.gstates = None
        
    def readConfig(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        options = config.options("DPOLE")
        for o in options:
            found = 0
            if o == "markov":
                self.markov = bool(config.getint("DPOLE","markov"))
                found = 1
            if o == "fixed":
                self.fixed = bool(config.getint("DPOLE","fixed"))
                found = 1
            if (o == "classic"):
                self.classic = bool(config.getint("DPOLE","classic"))
                found = 1
            if (o == "long_poles"):
                self.long_poles = bool(config.getint("DPOLE","long_poles"))
                found = 1
            if (found == 0):
                print("\033[1mOption %s in section [DPOLE] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()              
        # Set variables
        self.setTask(markov=self.markov, fixed=self.fixed, classic=self.classic, long_poles=self.long_poles)
        
    def setTask(self, markov=True, fixed=False, classic=False, long_poles=False):
        self.markov = markov
        self.fixed = fixed
        self.classic = classic
        self.long_poles = long_poles
        # We need to recompute all the variables
        if self.long_poles:
            self.masspole2 = 0.05
            self.length2 = 0.25
            self.total_mass = self.masspole + self.masspole2 + self.masscart
            self.polemass_length2 = self.masspole2 * self.length2

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot, theta2, theta_dot2 = self.state
        force = self.force_mag if action >= 0.0 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        costheta2 = math.cos(theta2)
        sintheta2 = math.sin(theta2)
        
        mu = 0.000002
        
        m1 = self.masspole * (1.0 - (costheta**2 * 3.0) / 4.0)
        m2 = self.masspole2 * (1.0 - (costheta2**2 * 3.0) / 4.0)

        f1 = self.polemass_length * theta_dot**2 * sintheta
        f2 = self.polemass_length2 * theta_dot2**2 * sintheta2
        
        xacc = (force + f1 + f2) / self.total_mass
        
        thetaacc = - 3.0 * (xacc * costheta + self.gravity * sintheta + (mu * theta_dot) / self.polemass_length) / (4.0 * self.length)
        thetaacc2 = - 3.0 * (xacc * costheta2 + self.gravity * sintheta2 + (mu * theta_dot2) / self.polemass_length2) / (4.0 * self.length2)
        
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot
        theta_dot2 = theta_dot2 + self.tau * thetaacc2
        theta2 = theta2 + self.tau * theta_dot2

        self.state = (x, x_dot, theta, theta_dot, theta2, theta_dot2)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or theta2 < -self.theta_threshold_radians
            or theta2 > self.theta_threshold_radians
        )

        reward = 1.0

        if self.render_mode == "human":
            self.render()
        
        ob = None
        if self.markov:
            ob = np.array(self.state, dtype=np.float32)
        else:
            ob = np.array([x, theta, theta2], dtype=np.float32)
        return ob, reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        if self.classic:
            self.state = (0.0, 0.0, 0.07, 0.0, 0.0, 0.0)
        else:
            if self.fixed:
                # Fixed states
                self.state = tuple(self.fixedStates[self.episode % 8])
                self.episode += 1
            else:
                # Random states
                self.state = []
                self.state.append(self.np_random.uniform(low=-1.944, high=1.944))
                self.state.append(self.np_random.uniform(low=-1.215, high=1.215))
                self.state.append(self.np_random.uniform(low=-0.10472, high=0.10472))
                self.state.append(self.np_random.uniform(low=-0.135088, high=0.135088))
                self.state.append(self.np_random.uniform(low=-0.10472, high=0.10472))
                self.state.append(self.np_random.uniform(low=-0.135088, high=0.135088))
                self.state = tuple(self.state)

        if self.render_mode == "human":
            self.render()
            
        ob = None
        if self.markov:
            ob = np.array(self.state, dtype=np.float32)
        else:
            x, x_dot, theta, theta_dot, theta2, theta_dot2 = self.state
            ob = np.array([x, theta, theta2], dtype=np.float32)
        return np.array(self.state, dtype=np.float32), {}

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
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        polelen2 = scale * (2 * self.length2)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (255, 0, 0))

        # First pole
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx - cartwidth / 4, coord[1] + carty + axleoffset + cartheight / 4)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (0, 255, 255))
        gfxdraw.filled_polygon(self.surf, pole_coords, (0, 255, 255))
        
        # Second pole
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen2 - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[4])
            coord = (coord[0] + cartx + cartwidth / 4, coord[1] + carty + axleoffset + cartheight / 4)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (0, 255, 255))
        gfxdraw.filled_polygon(self.surf, pole_coords, (0, 255, 255))

        """
        gfxdraw.aacircle(
            self.surf,
            int(cartx - cartwidth / 4),
            int(carty + axleoffset + cartheight / 4),
            int(polewidth / 2),
            (128, 128, 128),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx - cartwidth / 4),
            int(carty + axleoffset + cartheight / 4),
            int(polewidth / 2),
            (128, 128, 128),
        )
        
        gfxdraw.aacircle(
            self.surf,
            int(cartx + cartwidth / 4),
            int(carty + axleoffset + cartheight / 4),
            int(polewidth / 2),
            (128, 128, 128),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx + cartwidth / 4),
            int(carty + axleoffset + cartheight / 4),
            int(polewidth / 2),
            (128, 128, 128),
        )
        """
        
        # First wheel
        gfxdraw.aacircle(
            self.surf,
            int(cartx - cartwidth / 4),
            int(carty - cartheight / 2),
            int(polewidth / 2),
            (128, 128, 128),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx - cartwidth / 4),
            int(carty - cartheight / 2),
            int(polewidth / 2),
            (128, 128, 128),
        )
        
        # Second wheel
        gfxdraw.aacircle(
            self.surf,
            int(cartx + cartwidth / 4),
            int(carty - cartheight / 2),
            int(polewidth / 2),
            (128, 128, 128),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx + cartwidth / 4),
            int(carty - cartheight / 2),
            int(polewidth / 2),
            (128, 128, 128),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, int(carty - cartheight/ 2 - polewidth / 2), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
