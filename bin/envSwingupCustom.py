"""
Cart pole swing-up: Original version from:
https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py
adapted to work with evorobotpy
previously aapted from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py
Modified so that done=True when x is outside of -2.4 to 2.4
Reward is also reshaped to be similar to PyBullet/roboschool version
More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""

"""
TO-DO)
can be used in the following way
BUT ot culd be use more directly by defining shapes

        import gym
        from gym import spaces
        from cenvCartpoleswingup import CartPoleSwingUpEnv
        env = CartPoleSwingUpEnv()
        # Define the objects required (they depend on the environment)
        ob = np.arange(3, dtype=np.float32)
        ac = np.arange(1, dtype=np.float32)
        from policy import GymPolicy
        policy = GymPolicy(env, 3, 1, -1.0, 1.0, ob, ac, filename, cseed, nrobots, heterogeneous, test)
"""

from typing import TYPE_CHECKING, List, Optional

import logging
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding
import numpy as np

logger = logging.getLogger(__name__)

class customEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6 # pole's length
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        self.dt = 0.01  # seconds between state updates
        self.b = 0.1  # friction coefficient

        self.t = 0 # timestep
        self.t_limit = 1000

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        
        self.timer = 0
        
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Valid action
        action = np.clip(action, -1.0, 1.0)[0]
        action *= self.force_mag

        state = self.state
        x, x_dot, theta, theta_dot = state

        s = math.sin(theta)
        c = math.cos(theta)

        xdot_update = (-2*self.m_p_l*(theta_dot**2)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c**2)
        thetadot_update = (-3*self.m_p_l*(theta_dot**2)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c**2)
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt

        self.state = (x,x_dot,theta,theta_dot)
        
        self.scroll = x

        done = False
        if  x < -self.x_threshold or x > self.x_threshold:
          done = True

        self.t += 1

        if self.t >= self.t_limit:
          done = True

        reward_theta = (np.cos(theta)+1.0)/2.0
        reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))

        reward = reward_theta*reward_x

        obs = np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])

        return obs, reward, done, False, {}

    def reset(self, seed=None):
        self.scroll = 0.0
        #self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_done = None
        self.t = 0 # timestep
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def render(self, mode='human', close=False):
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
        
        screen_width = 600
        screen_height = 600 # before was 400
        world_width = 5  # max visible position of cart
        scale = screen_width/world_width
        carty = screen_height/2 # TOP OF CART
        polewidth = 6.0
        polelen = scale*self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0
        
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Draw surface
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
                
        pygame.transform.scale(self.surf, (scale, scale))
        
        if self.state is None: return None

        x = self.state
        
        # Draw cart
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (255, 0, 0))
        
        # Draw pole
        l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                -polelen - polewidth / 2,
                -polewidth / 2,
            )
        
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(x[2])#-x[2]
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (0, 0, 255))
        gfxdraw.filled_polygon(self.surf, pole_coords, (0, 0, 255))

        # Draw axle
        gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (0, 0, 0),
            )
        gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (0, 0, 0),
            )
            
        # Draw wheels
        # Left
        gfxdraw.aacircle(
                self.surf,
                int(cartx - cartwidth / 2),
                int(carty + cartheight / 2),
                int(polewidth / 2),
                (0, 0, 0),
            )
        gfxdraw.filled_circle(
                self.surf,
                int(cartx - cartwidth / 2),
                int(carty + cartheight / 2),
                int(polewidth / 2),
                (0, 0, 0),
            )
        # Right
        gfxdraw.aacircle(
                self.surf,
                int(cartx + cartwidth / 2),
                int(carty + cartheight / 2),
                int(polewidth / 2),
                (0, 0, 0),
            )
        gfxdraw.filled_circle(
                self.surf,
                int(cartx + cartwidth / 2),
                int(carty + cartheight / 2),
                int(polewidth / 2),
                (0, 0, 0),
            )
            
        # Draw track
        #gfxdraw.hline(self.surf, int(screen_width/2 - self.x_threshold*scale), int(screen_width/2 + self.x_threshold*scale), int(carty + cartheight / 2 + polewidth / 2), color=(0,0,0),)
        self.track = pygame.draw.line(self.surf, color=(0,0,0), start_pos=(screen_width/2 - self.x_threshold*scale,carty + cartheight / 2 + polewidth / 2), end_pos=(screen_width/2 + self.x_threshold*scale,carty + cartheight / 2 + polewidth / 2), width=1)
        
        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["video.frames_per_second"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -screen_width:]
        
  
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

