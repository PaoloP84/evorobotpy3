import sys, math, time
import numpy as np

from typing import TYPE_CHECKING, List, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e


if TYPE_CHECKING:
    import pygame

# Air hockey

FPS    = 50
SCALE  = 25.0

# Maximum velocity and force allowed to agents
MAX_VELOCITY = 50.0 # To be tuned!!!
MAX_FORCE = 500.0 # To be tuned!!!
K_FORCE = 20.0 # 4.0

# Field of view (angle)
FOV = math.pi# / 4.0

# Field sizes and diag
FIELD_WIDTH = 1000
FIELD_HEIGHT = 500
FIELD_DIAG = math.sqrt(math.pow(FIELD_WIDTH, 2.0) + math.pow(FIELD_HEIGHT, 2.0))
# Tolerance for going in the opponent's half of field
FIELD_EDGE = 100

# Maximum distance
MAX_DIST = FIELD_DIAG/SCALE

# Start width (for rendering)
START_W = 50/SCALE
# Start height (for rendering)
START_H = 50/SCALE

# Window sizes
VIEWPORT_W = 2000
VIEWPORT_H = 600

# Sizes of agents (paddles) and hockey disk (puck)
PADDLE_RADIUS = 1.0
PUCK_RADIUS = 0.5
# Puck density
PUCK_DENSITY = 0.5
# Puck velocity
PUCK_VELOCITY = 50.0
# Agent density
PADDLE_DENSITY = 3.0
# Agent damping
PADDLE_DAMPING = 0.5

# None value
NONE = -1

# Convert angle in range [-180°,180°]
def setAngleInRange(angle):
    a = angle
    # Angles belong to range [-180°,180°]
    if a > math.pi:
        a -= (2.0 * math.pi)
    if a < -math.pi:
        a += (2.0 * math.pi)
    return a

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        pass
    def EndContact(self, contact):
        pass

class BaselinePolicy:
    def __init__(self, nobs, nacts):
        # Set number of observations and actions
        self.nobs = nobs
        self.nacts = nacts

        self.rs = np.random.RandomState()
        
        # store current inputs and outputs
        self.ob = np.zeros(self.nobs)
        self.ac = np.zeros(self.nacts)

        # Network weights and biases
        self.weight = self.rs.uniform(-0.1, 0.1, (self.nobs * self.nacts)).reshape(self.nacts, self.nobs)
        self.bias = self.rs.uniform(-0.1, 0.1, self.nacts)
        # Reshape weights
        self.weight.reshape((self.nacts, self.nobs))
    def reset(self, ob):
        # Consistency check
        assert len(self.ob) == len(ob), "Passed a wrong number of inputs!!!"
        self.ob = np.array(ob)
    def forward(self):
        self.ac = np.tanh(np.dot(self.weight, self.ob) + self.bias)
    def predict(self, obs):
        self.reset(obs)
        self.forward()
        return self.ac

class customEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=[0.0,0.0], doSleep=True) # Gravity is set to 0 in this environment (game)
        self.field = None
        self.puck = None
        self.puckDensity = PUCK_DENSITY
        self.paddles = []
        self.paddleDensity = PADDLE_DENSITY
        self.nagents = 2
        # Opponent
        self.opponents = np.zeros(self.nagents, dtype=np.int64)
        self.opponents[0] = 1 # The agent is 0, opponent is 1

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = 1.0)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = 1.0,
                    categoryBits=0x0001,
                )

        # Maximum number of steps
        self.nsteps = 2000
        self.halfsteps = int(self.nsteps / 2)
        
        # Number of observations
        self.ob_len = 20 # Agent position (x,y) + distance and angle from opponent + distance and angle from puck + puck linear velocity (vx,vy)
        # Number of actions
        self.ac_len = 2
        
        self.scroll = START_W - VIEWPORT_W/SCALE/5

        act = np.ones(self.ac_len, dtype=np.float32)
        high = np.array([np.inf]*self.ob_len, dtype=np.float32)
        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-high, high)

        self.timer = 0

        # Opponent policy
        self.policy = BaselinePolicy(self.ob_len, self.ac_len) # the “bad guy”
        # Other observation
        self.otherObs = None
        # Other action
        self.otherAction = None
        
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.field: return
        self.world.contactListener = None
        for f in self.field:
            self.world.DestroyBody(f)
        self.field = []
        self.world.DestroyBody(self.puck)
        self.puck = None
        for paddle in self.paddles:
            self.world.DestroyBody(paddle)
        self.paddles = []

    def _generate_field(self):
        # Field
        self.field = []
        # Walls
        # Left vertical
        poly = [
            (START_W,   START_H),
            (START_W,   START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W-20/SCALE,    START_H),
            (START_W-20/SCALE,    START_H+(FIELD_HEIGHT/2-100)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        wall1 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        wall1.color1 = (0,0,0)
        wall1.color2 = (0,0,0)
        self.field.append(wall1)
        poly = [
            (START_W,   START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W,   START_H+FIELD_HEIGHT/SCALE),
            (START_W-20/SCALE,    START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W-20/SCALE,    START_H+FIELD_HEIGHT/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        wall2 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        wall2.color1 = (0,0,0)
        wall2.color2 = (0,0,0)
        self.field.append(wall2)
        # Left goal
        poly = [
            (START_W-80/SCALE,   START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W-80/SCALE,   START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W-60/SCALE,    START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W-60/SCALE,    START_H+(FIELD_HEIGHT/2+100)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        goal1 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        goal1.color1 = (0,0,0)
        goal1.color2 = (0,0,0)
        self.field.append(goal1)
        poly = [
            (START_W-80/SCALE,   START_H+(FIELD_HEIGHT/2-120)/SCALE),
            (START_W-80/SCALE,   START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W-20/SCALE,    START_H+(FIELD_HEIGHT/2-120)/SCALE),
            (START_W-20/SCALE,    START_H+(FIELD_HEIGHT/2-100)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        bottomgoal1 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        bottomgoal1.color1 = (0,0,0)
        bottomgoal1.color2 = (0,0,0)
        self.field.append(bottomgoal1)
        poly = [
            (START_W-80/SCALE,   START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W-80/SCALE,   START_H+(FIELD_HEIGHT/2+120)/SCALE),
            (START_W-20/SCALE,    START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W-20/SCALE,    START_H+(FIELD_HEIGHT/2+120)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        topgoal1 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        topgoal1.color1 = (0,0,0)
        topgoal1.color2 = (0,0,0)
        self.field.append(topgoal1)
        # Right vertical
        poly = [
            (START_W+FIELD_WIDTH/SCALE,   START_H),
            (START_W+FIELD_WIDTH/SCALE,   START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H+(FIELD_HEIGHT/2-100)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        wall3 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        wall3.color1 = (0,0,0)
        wall3.color2 = (0,0,0)
        self.field.append(wall3)
        poly = [
            (START_W+FIELD_WIDTH/SCALE,   START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W+FIELD_WIDTH/SCALE,   START_H+FIELD_HEIGHT/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H+FIELD_HEIGHT/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        wall4 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        wall4.color1 = (0,0,0)
        wall4.color2 = (0,0,0)
        self.field.append(wall4)
        # Right goal
        poly = [
            (START_W+(FIELD_WIDTH+60)/SCALE,   START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W+(FIELD_WIDTH+60)/SCALE,   START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W+(FIELD_WIDTH+80)/SCALE,    START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W+(FIELD_WIDTH+80)/SCALE,    START_H+(FIELD_HEIGHT/2+100)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        goal2 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        goal2.color1 = (0,0,0)
        goal2.color2 = (0,0,0)
        self.field.append(goal2)
        poly = [
            (START_W+(FIELD_WIDTH+20)/SCALE,   START_H+(FIELD_HEIGHT/2-120)/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,   START_H+(FIELD_HEIGHT/2-100)/SCALE),
            (START_W+(FIELD_WIDTH+80)/SCALE,    START_H+(FIELD_HEIGHT/2-120)/SCALE),
            (START_W+(FIELD_WIDTH+80)/SCALE,    START_H+(FIELD_HEIGHT/2-100)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        bottomgoal2 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        bottomgoal2.color1 = (0,0,0)
        bottomgoal2.color2 = (0,0,0)
        self.field.append(bottomgoal2)
        poly = [
            (START_W+(FIELD_WIDTH+20)/SCALE,   START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,   START_H+(FIELD_HEIGHT/2+120)/SCALE),
            (START_W+(FIELD_WIDTH+80)/SCALE,    START_H+(FIELD_HEIGHT/2+100)/SCALE),
            (START_W+(FIELD_WIDTH+80)/SCALE,    START_H+(FIELD_HEIGHT/2+120)/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        topgoal2 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        topgoal2.color1 = (0,0,0)
        topgoal2.color2 = (0,0,0)
        self.field.append(topgoal2)
        # Bottom horizontal
        poly = [
            (START_W-20/SCALE,   START_H-20/SCALE),
            (START_W-20/SCALE,   START_H),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H-20/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H)
            ]
        self.fd_polygon.shape.vertices=poly
        wall5 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        wall5.color1 = (0,0,0)
        wall5.color2 = (0,0,0)
        self.field.append(wall5)
        # Up horizontal
        poly = [
            (START_W-20/SCALE,   START_H+FIELD_HEIGHT/SCALE),
            (START_W-20/SCALE,   START_H+FIELD_HEIGHT/SCALE+20/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H+FIELD_HEIGHT/SCALE),
            (START_W+(FIELD_WIDTH+20)/SCALE,    START_H+FIELD_HEIGHT/SCALE+20/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        wall6 = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        wall6.color1 = (0,0,0)
        wall6.color2 = (0,0,0)
        self.field.append(wall6)
        
    # Return distance and angle from another object
    def distanceAndAngle(self, paddleId, other, radius):
        mx, my = self.paddles[paddleId].position
        ox, oy = other.position
        d = math.sqrt(math.pow((mx - ox), 2) + math.pow((my - oy), 2)) - PADDLE_RADIUS - radius
        a = math.atan2((oy - my), (ox - mx)) - self.paddles[paddleId].angle
        # Set angle in range [-pi,pi]
        a = setAngleInRange(a)
        if abs(d) < 1e-6:
            d = 0.0
        if abs(a) < 1e-6:
            a = 0.0
        return d, a

    # Check whether or not the puck is in goal
    def isInGoal(self):
        goal = False
        paddle = NONE
        x, y = self.puck.position
        # Check left goal first
        if (x <= (START_W - 20/SCALE - 2 * PUCK_RADIUS/SCALE)) and (y >= START_H+(FIELD_HEIGHT/2-50)/SCALE and y <= START_H+(FIELD_HEIGHT/2+50)/SCALE):
            goal = True
            paddle = 1
        # Check right goal then
        elif (x >= (START_W + FIELD_WIDTH/SCALE + 20/SCALE + 2 * PUCK_RADIUS/SCALE)) and (y >= START_H+(FIELD_HEIGHT/2-50)/SCALE and y <= START_H+(FIELD_HEIGHT/2+50)/SCALE):
            goal = True
            paddle = 0
        return goal, paddle

    # Check whether or not paddle hit the puck
    def hitPuck(self, paddleId):
        hit = False
        # Compute distance between paddle and puck
        d, a = self.distanceAndAngle(paddleId, self.puck, PUCK_RADIUS)
        if d <= 0.0:
            hit = True
        return hit

    # Check whether or not paddle hit the other paddle
    def hitPaddle(self, paddleId):
        hit = False
        # Compute distance between paddles
        oId = self.opponents[paddleId]
        d, a = self.distanceAndAngle(paddleId, self.paddles[oId], PADDLE_RADIUS)
        if d <= 0.0:
            hit = True
        return hit

    # Check whether or not the paddle is out of the field
    def isOut(self, paddleId):
        out = False
        x, y = self.paddles[paddleId].position
        if x < START_W-20/SCALE or x > START_W+(FIELD_WIDTH+20)/SCALE or \
            y < START_H-20/SCALE or y > START_H+(FIELD_HEIGHT+20)/SCALE:
            out = True
        return out

    def setOtherAction(self, action):
        self.otherAction = action

    def reset(self, seed=None):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.scroll = 0.0
        
        self.timer = 0

        # Generate field
        self._generate_field()

        np.random.seed(seed)

        # Extract randomly a number in the range [0,1]
        rn = np.random.uniform(0.0,1.0)
        # Flag whether the agent plays on the left side or on the right side
        if rn >= 0.5:
            self.playLeft = True
        else:
            self.playLeft = False

        # Variables: SCALE = 25, START_W = 0, START_H = 0, FIELD_WIDTH = 1000, FIELD_HEIGHT = 500

        puckAngle = 0.0 # Useless
        # Puck
        self.puck = self.world.CreateDynamicBody(
            position=(START_W + 250/SCALE, START_H + FIELD_HEIGHT/2/SCALE),
            angle=puckAngle,
            fixtures = fixtureDef(shape = circleShape(radius=PUCK_RADIUS), friction=0.0, density=self.puckDensity, restitution=1.0)
            )
        self.puck.linearVelocity = (0,0)
        self.puck.color1 = (0,0,0)
        self.puck.color2 = (0,0,0)
        # Set damping for the puck (both linear and angular)
        self.puck.linearDamping = 0.5
        self.puck.angularDamping = 0.5
        # Paddles
        self.paddles = []
        # Left paddle
        paddle1 = self.world.CreateDynamicBody(
            position=(START_W + 100/SCALE, START_H + FIELD_HEIGHT/2/SCALE),
            angle=0.0,
            fixtures = fixtureDef(shape = circleShape(radius=PADDLE_RADIUS), density=3.0, restitution=1.0),
            fixedRotation = True
            )
        # Set color to red
        paddle1.color1 = (255,0,0)
        paddle1.color2 = (255,0,0)
        paddle1.linearDamping = PADDLE_DAMPING
        paddle1.angularDamping = PADDLE_DAMPING
        self.paddles.append(paddle1)
        # Right paddle
        paddle2 = self.world.CreateDynamicBody(
            position=(START_W + (FIELD_WIDTH - 100)/SCALE, START_H + FIELD_HEIGHT/2/SCALE),
            angle=math.pi,
            fixtures = fixtureDef(shape = circleShape(radius=PADDLE_RADIUS), density=3.0, restitution=1.0),
            fixedRotation = True
            )
        # Set color to blue
        paddle2.color1 = (0,0,255)
        paddle2.color2 = (0,0,255)
        paddle2.linearDamping = PADDLE_DAMPING
        paddle2.angularDamping = PADDLE_DAMPING
        self.paddles.append(paddle2)
        
        self.drawlist = self.field + [self.puck] + self.paddles
        
        fakeAction = [0.0 for _ in range(self.ac_len * self.nagents)]

        # Reward
        self.reward = np.zeros(self.nagents, dtype=np.float64)
        # Set hit flag to False (at the beginning of the episode the puck is not hit)
        self.hit = np.full(self.nagents, False)
        # Reset step
        self.cstep = 0

        return self.step(np.array(fakeAction))[0], {} # self._step

    def step(self, action): # _step
        # Apply actions
        for i in range(self.nagents):
            act = []
            # Agent position
            px, py = self.paddles[i].position
            if i == 0:
                act = action[0:self.ac_len]
                # Compute forces
                fx = (((float(act[0]) * MAX_VELOCITY) - self.paddles[i].linearVelocity.x) / MAX_VELOCITY) * MAX_FORCE
                fy = (((float(act[1]) * MAX_VELOCITY) - self.paddles[i].linearVelocity.y) / MAX_VELOCITY) * MAX_FORCE
                # Compute the distance of the agent from the half of field
                if px > (START_W + (FIELD_WIDTH/2-FIELD_EDGE)/SCALE):
                    dx = (px - (START_W + (FIELD_WIDTH/2-FIELD_EDGE)/SCALE)) / (START_W + (FIELD_WIDTH/2-FIELD_EDGE)/SCALE) * 4.0
                    # Put a horizontal force pushing the agent toward its half of field
                    fx += -MAX_FORCE * 2.0 * dx
            else:
                act = action[self.ac_len:]
                # Invert actions for right agent
                act[0] = (2.0 - (act[0] + 1.0)) - 1.0
                act[1] = (2.0 - (act[1] + 1.0)) - 1.0
                # Compute forces
                fx = (((float(act[0]) * MAX_VELOCITY) - self.paddles[i].linearVelocity.x) / MAX_VELOCITY) * MAX_FORCE
                fy = (((float(act[1]) * MAX_VELOCITY) - self.paddles[i].linearVelocity.y) / MAX_VELOCITY) * MAX_FORCE
                # Compute the distance of the agent from the half of field
                if px < (START_W + (FIELD_WIDTH/2+FIELD_EDGE)/SCALE):
                    dx = ((START_W + (FIELD_WIDTH/2+FIELD_EDGE)/SCALE) - px) / (START_W + (FIELD_WIDTH/2+FIELD_EDGE)/SCALE) * 4.0
                    # Put a horizontal force pushing the agent toward its half of field
                    fx += MAX_FORCE * 2.0 * dx
            # Apply forces to agent
            self.paddles[i].ApplyForce(force=(fx,fy), point=(px,py), wake=True)
        # Perform a world step
        self.world.Step(1.0/FPS, 6*30, 2*30)

        self.scroll = START_W - VIEWPORT_W/SCALE/5
        
        # Fill the agent observation
        obs = []
        for i in range(self.nagents):
            o = self.opponents[i]
            # Get relative distance and angle from opponent
            d1, a1 = self.distanceAndAngle(i, self.paddles[o], PADDLE_RADIUS)
            # Get relative distance and angle from puck
            d2, a2 = self.distanceAndAngle(i, self.puck, PUCK_RADIUS)
            # Puck velocity
            vx, vy = float(NONE), float(NONE)
            # Agent position
            x, y = float(NONE), float(NONE)
            if i == 0:
                x = (self.paddles[i].position.x - START_W) / (FIELD_WIDTH/SCALE)
                y = (((self.paddles[i].position.y - START_H) / (FIELD_HEIGHT/SCALE)) * 2.0) - 1.0
                vx, vy = self.puck.linearVelocity
            else:
                x = (FIELD_WIDTH/SCALE - (self.paddles[i].position.x - START_W)) / (FIELD_WIDTH/SCALE)
                y = (((FIELD_HEIGHT/SCALE - (self.paddles[i].position.y - START_H)) / (FIELD_HEIGHT/SCALE)) * 2.0) - 1.0
                vx, vy = -self.puck.linearVelocity # Puck velocity is inverted
            # Add agent observation
            obs += [x, y, d1 / MAX_DIST, a1 / FOV, d2 / MAX_DIST, a2 / FOV, vx / (PUCK_VELOCITY * 2.0), vy / (PUCK_VELOCITY * 2.0)] # To be normalized in ranges
        
        # Reward
        reward = 0.0 # This variable refers to left agent only
        # Check whether or not the paddle hit the puck
        for i in range(self.nagents):
            self.hit[i] = self.hitPuck(i)
            if self.hit[i]:
                # If the paddle hit the puck, it receives a bonus of 0.001
                self.reward[i] += 0.001
                if i == 0:
                    reward += 0.001
            
        # Check whether or not the paddle marked a goal
        done = False
        goal, paddle = self.isInGoal()
        if goal:
            # Get the paddle opponent
            oId = self.opponents[paddle]
            # Opponent recevies a malus
            self.reward[oId] = -100.0
            if paddle == 1:
                reward = -100.0
            # The agent is rewarded only if it hit the puck at least once
            self.reward[paddle] += 1.0
            if paddle == 0:
                # If the agent 0 marked, reward is +100
                reward += 100.0
                # Change puck color (for rendering only)
                self.puck.color1 = (0,255,0)
                self.puck.color2 = (0,255,0)
            else:
                # Change puck color (for rendering only)
                self.puck.color1 = (255,0,0)
                self.puck.color2 = (255,0,0)
            # In case of goal, episode stops
            done = True

        # Other penalties (for going out of the field or for hitting the opponent)
        for i in range(self.nagents):
            # Check whether or not the paddle is out of the field
            out = self.isOut(i)
            if out:
                # Episode stops
                done = True
                # Agent is penalized
                self.reward[i] -= 100.0
                if i == 0:
                    reward -= 100.0

            # Check whether or not there is a contact between paddles
            hit = self.hitPaddle(i)
            if hit:
                self.reward[i] -= 0.01 # Penalty for contacts
                if i == 0:
                    reward -= 0.01
                    
        # Update step counter
        self.cstep += 1

        self.timer += 1

        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def render(self, mode='human', close=False): # _render
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
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        pygame.draw.polygon(
            self.surf, 
            points = [
            (self.scroll * SCALE, 0),
            (self.scroll * SCALE + VIEWPORT_W, 0),
            (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
            (self.scroll * SCALE, VIEWPORT_H),
            ], 
            color=(230, 230, 255) 
        )
        
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]
            
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

