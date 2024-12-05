import sys, math, time
import numpy as np

from typing import TYPE_CHECKING, List, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding

# Multiple agents task

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

FPS    = 50
SCALE  = 25.0

# TO DO:
# agents da dynamicBody a kinematicBody (come gestisce le collisioni però)?
# set velocities: setLinearVelocity((vx, vy)) e setAngularVelocity(theta)? Quindi rete neurale con 3 output???
# sensor inputs of the agents: che usiamo?
# dato field-of-view (es. 90°) e distanza massima di visibilità, potremmo:
# - se vede il compagno blu in un settore, prendiamo quanto ne vede
# - se vede un oggetto in un settore, prendiamo quanto ne vede
# - in caso di sovrapposizione, prende quello più vicino?
# - colore oggetto rilevato (3 neuroni R, G, B che possono assumere valori tra 0 e 1)?

# Field sizes and diag
ARENA_WIDTH = 500
ARENA_HEIGHT = 500
ARENA_DIAG = math.sqrt(math.pow(ARENA_WIDTH, 2.0) + math.pow(ARENA_HEIGHT, 2.0))

# Maximum distance
MAX_DIST = 250/SCALE

# Start width (for rendering)
START_W = 375/SCALE
# Start height (for rendering)
START_H = 75/SCALE

# Window sizes
VIEWPORT_W = 1280
VIEWPORT_H = 640

# Sizes of agent and object
AGENT_RADIUS = 0.75
OBJECT_RADIUS = 0.5
# Thresold
THRESHOLD = 0.25
# Object density
OBJECT_DENSITY = 1.5
# Agent density
AGENT_DENSITY = 3.0
# Agent damping
AGENT_DAMPING = 0.0

# Constants
MAX_VEL = 0.2
MAX_ROT = math.pi / 20.0
MAX_DIFF = 2.0 * MAX_VEL
# Field of view (angle)
FOV = math.pi / 2.0
# Number of sectors
NUM_SECTORS = 6

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
        """
        if self.env.target==contact.fixtureA.body or self.env.target==contact.fixtureB.body:
            self.env.target.ground_contact = True
        for agent in self.env.agents:
            if agent in [contact.fixtureA.body, contact.fixtureB.body]:
                agent.ground_contact = True
        """
        pass
    def EndContact(self, contact):
        """
        if self.env.object==contact.fixtureA.body or self.env.object==contact.fixtureB.body:
            self.env.object.ground_contact = False
        for agent in self.env.agents:
            if agent in [contact.fixtureA.body, contact.fixtureB.body]:
                agent.ground_contact = False
        """
        pass

class customEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=[0.0,0.0], doSleep=True) # Gravity is set to 0 in this environment (game)
        self.arena = None
        self.target = []
        self.object = None
        self.objectDensity = OBJECT_DENSITY
        self.agent = None
        self.agentDensity = AGENT_DENSITY
        self.drawlist = []

        self.prev_shaping = None

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
        self.nsteps = 1000
        
        # Number of observations
        self.ob_len = NUM_SECTORS # For each sector the proportion of target detected
        # Number of actions
        self.ac_len = 2 # left and right wheel speeds

        act = np.ones(self.ac_len, dtype=np.float32)
        high = np.array([np.inf]*self.ob_len, dtype=np.float32)
        self.action_space = spaces.Box(-act,act)
        self.observation_space = spaces.Box(-high, high)

        self.timer = 0
        
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.arena: return
        self.world.contactListener = None
        for f in self.arena:
            self.world.DestroyBody(f)
        self.arena = []
        self.world.DestroyBody(self.object)
        self.object = None
        self.world.DestroyBody(self.agent)
        self.agent = None
        # Maybe useless
        self.drawlist = []

    def _generate_arena(self):
        # Field
        self.arena = []
        # Walls
        # Left vertical
        poly = [
            (START_W,   START_H),
            (START_W,   START_H+ARENA_HEIGHT/SCALE),
            (START_W-20/SCALE,    START_H),
            (START_W-20/SCALE,    START_H+ARENA_HEIGHT/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        leftWall = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        leftWall.color1 = (0,0,0)
        leftWall.color2 = (0,0,0)
        self.arena.append(leftWall)
        # Right vertical
        poly = [
            (START_W+ARENA_WIDTH/SCALE,   START_H),
            (START_W+ARENA_WIDTH/SCALE,   START_H+ARENA_HEIGHT/SCALE),
            (START_W+(ARENA_WIDTH+20)/SCALE,    START_H),
            (START_W+(ARENA_WIDTH+20)/SCALE,    START_H+ARENA_HEIGHT/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        rightWall = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        rightWall.color1 = (0,0,0)
        rightWall.color2 = (0,0,0)
        self.arena.append(rightWall)
        # Bottom horizontal
        poly = [
            (START_W-20/SCALE,   START_H-20/SCALE),
            (START_W-20/SCALE,   START_H),
            (START_W+(ARENA_WIDTH+20)/SCALE,    START_H-20/SCALE),
            (START_W+(ARENA_WIDTH+20)/SCALE,    START_H)
            ]
        self.fd_polygon.shape.vertices=poly
        bottomWall = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        bottomWall.color1 = (0,0,0)
        bottomWall.color2 = (0,0,0)
        self.arena.append(bottomWall)
        # Up horizontal
        poly = [
            (START_W-20/SCALE,   START_H+ARENA_HEIGHT/SCALE),
            (START_W-20/SCALE,   START_H+ARENA_HEIGHT/SCALE+20/SCALE),
            (START_W+(ARENA_WIDTH+20)/SCALE,    START_H+ARENA_HEIGHT/SCALE),
            (START_W+(ARENA_WIDTH+20)/SCALE,    START_H+ARENA_HEIGHT/SCALE+20/SCALE)
            ]
        self.fd_polygon.shape.vertices=poly
        topWall = self.world.CreateStaticBody(
            fixtures = self.fd_polygon)
        topWall.color1 = (0,0,0)
        topWall.color2 = (0,0,0)
        self.arena.append(topWall)
        
    # Return distance and angle from another object
    def distanceAndAngle(self, other, radius):
        mx, my = self.agent.position
        ox, oy = other.position
        d = math.sqrt(math.pow((mx - ox), 2) + math.pow((my - oy), 2)) - AGENT_RADIUS - radius
        a = math.atan2((oy - my), (ox - mx)) - self.agent.angle
        # Set angle in range [-pi,pi]
        a = setAngleInRange(a)
        if abs(d) < 1e-6:
            d = 0.0
        if abs(a) < 1e-6:
            a = 0.0
        return d, a
        
    def calcAngleBetweenTangents(self, other, radius, angle):
        mx, my = self.agent.position
        ox, oy = other.position
        # Compute tangent points
        p1x = ox + radius * math.cos(angle + math.pi / 2.0)
        p1y = oy + radius * math.sin(angle + math.pi / 2.0)
        p2x = ox + radius * math.cos(angle - math.pi / 2.0)
        p2y = oy + radius * math.sin(angle - math.pi / 2.0)
        # Compute angular coefficients of tangent rects
        m1 = (p1y - my) / (p1x - mx)
        m2 = (p2y - my) / (p2x - mx)
        # Get angle between the two tangent rects
        tan_ang = abs((m1 - m2) / (1 + m1 * m2))
        ang = math.atan(tan_ang)
        return ang

    def calcSector(self, other, radius, relAngle):
        # Compute the angle between tangent rects to the other object
        ang = self.calcAngleBetweenTangents(other, radius, relAngle)
        # Compute extremes and set in boundaries if necessary
        minAng = relAngle - ang
        if minAng < -FOV / 2.0:
            minAng = -FOV / 2.0
        maxAng = relAngle + ang
        if maxAng > FOV / 2.0:
            maxAng = FOV / 2.0
        # Look for the sector(s)
        sector = []
        sectorSize = FOV / float(NUM_SECTORS)
        start = -FOV / 2.0
        end = start + sectorSize
        if abs(end) < 1e-6:
            end = 0.0
        # We check first if the angle is inside a specific sectors (extremes excluded!)
        # Do first for min angle
        angle = minAng
        found = False
        i = 0
        while i < NUM_SECTORS and not found:
            if angle > start and angle < end:
                sector.append(i)
                found = True
            # Update sector range
            start = end
            if abs(start) < 1e-6:
                start = 0.0
            end += sectorSize
            if abs(end) < 1e-6:
                end = 0.0
            i += 1
        if not found:
            # The angle is precise
            if angle == -FOV / 2.0:
                sector.append(0)
            elif angle == FOV / 2.0:
                sector.append(NUM_SECTORS - 1)
            else:
                # None of the extremes match the angle, we must fill two sectors
                extreme = -FOV / 2.0 + sectorSize
                if abs(extreme) < 1e-6:
                    extreme = 0.0
                i = 0
                j = 1
                while extreme <= (FOV / 2.0 - sectorSize) and not found:
                    if angle == extreme:
                        sector.append(i)
                        sector.append(j)
                        found = True
                    extreme += sectorSize
                    if abs(extreme) < 1e-6:
                        extreme = 0.0
                    i += 1
                    j += 1
        # Repeat the same procedure for max angle
        angle = maxAng
        found = False
        i = 0
        while i < NUM_SECTORS and not found:
            if angle > start and angle < end:
                sector.append(i)
                found = True
            # Update sector range
            start = end
            if abs(start) < 1e-6:
                start = 0.0
            end += sectorSize
            if abs(end) < 1e-6:
                end = 0.0
            i += 1
        if not found:
            # The angle is precise
            if angle == -FOV / 2.0:
                sector.append(0)
            elif angle == FOV / 2.0:
                sector.append(NUM_SECTORS - 1)
            else:
                # None of the extremes match the angle, we must fill two sectors
                extreme = -FOV / 2.0 + sectorSize
                if abs(extreme) < 1e-6:
                    extreme = 0.0
                i = 0
                j = 1
                while extreme <= (FOV / 2.0 - sectorSize) and not found:
                    if angle == extreme:
                        sector.append(i)
                        sector.append(j)
                        found = True
                    extreme += sectorSize
                    if abs(extreme) < 1e-6:
                        extreme = 0.0
                    i += 1
                    j += 1
        return sector, minAng, maxAng

    def setNSteps(self, nsteps):
        self.nsteps = nsteps

    def reset(self, seed=None):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.scroll = 0.0
        
        self.timer = 0

        # Generate arena
        self._generate_arena()

        # Agents
        positions = []
        ok = False
        cpos = None
        while not ok:
            px = self.np_random.uniform(START_W + 50 / SCALE, START_W + (ARENA_WIDTH - 50)/SCALE)
            py = self.np_random.uniform(START_H + 50 / SCALE, START_H + (ARENA_HEIGHT - 50)/SCALE)
            cpos = (px, py)
            ok = True
            # Check distance from objects
            for opos in positions:
                dist = math.sqrt(math.pow((cpos[0] - opos[0]), 2.0) + math.pow((cpos[1] - opos[1]), 2.0))
                if dist <= (150 / SCALE):
                    ok = False
        positions.append(cpos)
        px, py = cpos
        ang = self.np_random.uniform(-math.pi, math.pi)
        self.agent = self.world.CreateDynamicBody(
            position=(px, py),
            angle=ang,
            fixtures = fixtureDef(shape = circleShape(radius=AGENT_RADIUS), density=self.agentDensity, restitution=1.0),
            fixedRotation = False
        )
        # Set color to blue
        self.agent.color1 = (0,0,255)
        self.agent.color2 = (0,0,255)
        self.agent.linearDamping = AGENT_DAMPING
        self.agent.angularDamping = AGENT_DAMPING

        # Object
        ok = False
        cpos = None
        while not ok:
            px = self.np_random.uniform(START_W + 50 / SCALE, START_W + (ARENA_WIDTH - 50)/SCALE)
            py = self.np_random.uniform(START_H + 50 / SCALE, START_H + (ARENA_HEIGHT - 50)/SCALE)
            cpos = (px, py)
            ok = True
            for opos in positions:
                dist = math.sqrt(math.pow((cpos[0] - opos[0]), 2.0) + math.pow((cpos[1] - opos[1]), 2.0))
                if dist <= (150 / SCALE):
                    ok = False
        positions.append(cpos)
        self.target = cpos
        # Create a small cylinder at the center of the target area
        self.object = self.world.CreateStaticBody(
            position=(px,py),
            angle=0.0,
            fixtures = fixtureDef(shape = circleShape(radius=OBJECT_RADIUS), friction=0.0, density=self.objectDensity, restitution=1.0)
        )
        self.object.linearVelocity = (0,0)
        self.object.color1 = (255,0,0)
        self.object.color2 = (0,0,0)
        
        self.drawlist = self.arena + [self.object] + [self.agent]

        # step
        self.cstep = 0
        
        if self.render_mode == "human":
            self.render()
            
        return self.getObs(), {}

    def step(self, action): # _step
        # Agent position
        px, py = self.agent.position
        # Compute "motors" (i.e., wheel speeds)
        motorLeft = MAX_VEL * float(action[0])
        motorRight = MAX_VEL * float(action[1])
        # Compute the agent motion
        # Difference between two motors
        diff = motorLeft - motorRight
        # Get the absolute value
        absdiff = abs(diff)
        # Compute the rotation
        rot = MAX_ROT * absdiff / MAX_DIFF
        # Get the angle
        angle = self.agent.angle
        # Update angle based on difference (negative -> turn right, positive -> turn left)
        if diff < 0.0:
            # If the difference is negative (i.e., right motor > left motor), the robot turns left
            angle += rot
        elif diff > 0.0:
            # If the difference is positive (i.e., left motor > right motor), the robot turns right
            angle -= rot
        # Set angle in range [-180°,180°]
        angle = setAngleInRange(angle)
        # Average velocity
        avgVel = (motorLeft + motorRight) / 2.0
        # Update position
        px += avgVel * math.cos(angle)
        py += avgVel * math.sin(angle)
        # Collisions with walls
        if px <= START_W + AGENT_RADIUS or px >= START_W + ARENA_WIDTH/SCALE - AGENT_RADIUS or py <= START_H + AGENT_RADIUS or py >= START_H + ARENA_HEIGHT/SCALE - AGENT_RADIUS:
            # Restore previous position
            px, py = self.agent.position
            # Change agent's orientation based on the rotation performed
            if diff < 0.0:
                # Agent rotated on the right -> make it go on the left
                angle += math.pi / 2.0
            elif diff > 0.0:
                # Agent rotated on the left -> make it go on the right
                angle -= math.pi / 2.0
            else:
                # Agent arrived at wall frontally -> invert its direction
                angle += math.pi
            angle = setAngleInRange(angle)
        # Collision with objects
        d, _ = self.distanceAndAngle(self.object, OBJECT_RADIUS)
        if d < 1e-6:
            # Simply restore old position
            px, py = self.agent.position
        # Update position and orientation of current agent
        self.agent.angle = angle
        self.agent.position = (px, py)        
        # Perform a world step
        self.world.Step(1.0/FPS, 6*30, 2*30)
        
        reward = 0.0
        # Check whether the agent with the target
        px, py = self.agent.position
        ox, oy = self.target
        d = math.sqrt(math.pow((px - ox), 2) + math.pow((py - oy), 2))
        if d <= AGENT_RADIUS + OBJECT_RADIUS + THRESHOLD:
            reward = 1.0
        
        # Update step counter
        self.cstep += 1
        
        # Get new observations
        obs = self.getObs()

        # Check whether the episode must be ended
        done = False

        # Check the number of performed steps
        if self.cstep >= self.nsteps:
            done = True

        self.timer += 1
        
        if self.render_mode == "human":
            self.render()
        
        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def getObs(self):
        # Fill the agent observation
        obs = [0.0 for _ in range(NUM_SECTORS + 1)]
        # Check whether robot detects any object
        dist, relang = self.distanceAndAngle(self.object, OBJECT_RADIUS)
        if dist <= MAX_DIST and (relang >= -FOV / 2.0 and relang <= FOV / 2.0):
            # Store the sector
            objsector, objminang, objmaxang = self.calcSector(self.object, OBJECT_RADIUS, relang)
            for i in objsector:
                obs[i] = (1.0 - dist / MAX_DIST)
        return obs

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
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )
        
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    width = 0
                    if obj.color1 == (255, 0, 0):
                        width = 1
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                        width=width,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                        width=width,
                    )
                    if obj.color1 == (0, 0, 255) and obj.color2 == (0, 0, 255):
                        # Agent -> draw visual cone
                        center = trans * f.shape.pos * SCALE
                        # Start position is fixed
                        sx = center.x
                        sy = center.y
                        angle = obj.angle
                        start_ang = -FOV / 2.0
                        end_ang = FOV / 2.0
                        incr_ang = FOV / 6.0
                        # Loop over angles
                        while start_ang <= end_ang:
                            ex = sx + MAX_DIST * SCALE * math.cos(angle + start_ang)
                            ey = sy + MAX_DIST * SCALE * math.sin(angle + start_ang)
                            pygame.draw.aaline(
                                self.surf,
                                start_pos=(sx, sy),
                                end_pos=(ex, ey),
                                color=(255, 255, 0),
                            )
                            start_ang += incr_ang
                        # Draw agent orientation
                        ex = sx + AGENT_RADIUS * SCALE * math.cos(angle)
                        ey = sy + AGENT_RADIUS * SCALE * math.sin(angle)
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=(sx, sy),
                            end_pos=(ex, ey),
                            color=(0, 0, 0),
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

