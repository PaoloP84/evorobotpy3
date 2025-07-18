######################################################################################################################
#
# Conway's game of life environment 
#
# Appeared in public in: 
# Gardner, M. Mathematical games-The fantastic combinations of John Conway’s new solitaire game, Life, 1970. Scientific American, October, 120-123.
#
######################################################################################################################

import sys, math, time, os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

# Size of grid, which is squared
GRID_SIZE = 100
# Frames per second
FPS = 1

class GameOfLifeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.seed()
        self.viewer = None
        # Figure to see animation (rendering)
        self.fig = None
        self.ax = None

        # Maximum number of steps
        self.nsteps = 100
        
        # Number of observations
        self.ob_len = GRID_SIZE ** 2 # For each sector the proportion of object/agent detected + blue color + red color
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
        for obj in self.objects:
            self.world.DestroyBody(obj)
        self.objects = []
        for agent in self.agents:
            self.world.DestroyBody(agent)
        self.agents = []
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
    def distanceAndAngle(self, agentId, other, radius):
        mx, my = self.agents[agentId].position
        ox, oy = other.position
        d = math.sqrt(math.pow((mx - ox), 2) + math.pow((my - oy), 2)) - AGENT_RADIUS - radius
        a = math.atan2((oy - my), (ox - mx)) - self.agents[agentId].angle
        # Set angle in range [-pi,pi]
        a = setAngleInRange(a)
        if abs(d) < 1e-6:
            d = 0.0
        if abs(a) < 1e-6:
            a = 0.0
        return d, a

    def calcSector(self, angle):
        sector = []
        sectorSize = FOV / float(NUM_SECTORS)
        start = -FOV / 2.0
        end = start + sectorSize
        if abs(end) < 1e-6:
            end = 0.0
        # We check first if the angle is inside a specific sectors (extremes excluded!)
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
        return sector

    def setNSteps(self, nsteps):
        self.nsteps = nsteps
        
    def setNAgents(self, nagents):
        self.nagents = nagents

    def reset(self, seed=None):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.scroll = 0.0
        
        self.timer = 0

        # Generate arena
        self._generate_arena()

        # Agents
        self.agents = []
        positions = []
        for _ in range(self.nagents):
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
            agent = self.world.CreateDynamicBody(
                position=(px, py),
                angle=ang,
                fixtures = fixtureDef(shape = circleShape(radius=AGENT_RADIUS), density=self.agentDensity, restitution=1.0),
                fixedRotation = False
                )
            # Set color to blue
            agent.color1 = (0,0,255)
            agent.color2 = (0,0,255)
            agent.linearDamping = AGENT_DAMPING
            agent.angularDamping = AGENT_DAMPING
            self.agents.append(agent)

        objectAngle = 0.0 # Useless
        # Object
        self.objects = []
        for elem in range(NUM_OBJECTS):
            ok = False
            cpos = None
            while not ok:
                px = self.np_random.uniform(START_W + 50 / SCALE, START_W + (ARENA_WIDTH - 50)/SCALE)
                py = self.np_random.uniform(START_H + 50 / SCALE, START_H + (ARENA_HEIGHT - 50)/SCALE)
                cpos = (px, py)
                ok = True
                for opos in positions:
                    dist = math.sqrt(math.pow((cpos[0] - opos[0]), 2.0) + math.pow((cpos[1] - opos[1]), 2.0))
                    if dist <= (100 / SCALE):
                        ok = False
            positions.append(cpos)
            px, py = cpos
            obj = self.world.CreateStaticBody(
                position=(px, py),
                angle=objectAngle,
                fixtures = fixtureDef(shape = circleShape(radius=OBJECT_RADIUS), friction=0.0, density=self.objectDensity, restitution=1.0)
                )
            obj.linearVelocity = (0,0)
            if elem % 2 == 0:
                obj.color1 = (255,0,0)
                obj.color2 = (255,0,0)
            else:
                obj.color1 = (0,255,0)
                obj.color2 = (0,255,0)
            self.objects.append(obj)
        
        self.drawlist = self.arena + self.objects + self.agents

        # step
        self.cstep = 0
        
        if self.render_mode == "human":
            self.render()
            
        return self.getObs(), {}

    def step(self, action): # _step
        # Apply actions
        for i in range(self.nagents):
            act = []
            # Agent position
            px, py = self.agents[i].position
            if i == 0:
                act = action[0:self.ac_len]
            else:
                act = action[self.ac_len:]
            # Compute "motors" (i.e., wheel speeds)
            motorLeft = MAX_VEL * float(act[0])
            motorRight = MAX_VEL * float(act[1])
            # Compute the agent motion
            # Difference between two motors
            diff = motorLeft - motorRight
            # Get the absolute value
            absdiff = abs(diff)
            # Compute the rotation
            rot = MAX_ROT * absdiff / MAX_DIFF
            # Get the angle
            angle = self.agents[i].angle
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
            px += avgVel * math.cos(angle)#(MAX_DIFF - absdiff) * math.cos(angle)
            py += avgVel * math.sin(angle)#(MAX_DIFF - absdiff) * math.sin(angle)
            # Collisions with walls
            if px <= START_W + AGENT_RADIUS or px >= START_W + ARENA_WIDTH/SCALE - AGENT_RADIUS or py <= START_H + AGENT_RADIUS or py >= START_H + ARENA_HEIGHT/SCALE - AGENT_RADIUS:
                # Restore previous position
                px, py = self.agents[i].position
                # Change agent's orientation based on the rotation performed
                if diff < 0.0:
                    # Agent rotated on the right -> make it go on the left
                    angle += math.pi / 2.0
                elif diff > 0.0:
                    # Agent rotated on the left -> make it go on the right
                    angle -= math.pi / 2.0
                else:
                    # Agent arrived at wall frontally -> invert its direction
                    angle = += math.pi#-angle
                """
                # Random rotation (either left or right) of 90°
                coin = self.np_random.uniform(0.0, 1.0)
                if coin >= 0.5:
                    angle += math.pi / 2.0
                else:
                    angle -= math.pi / 2.0
                """
                angle = setAngleInRange(angle)
            # Collision with other peers
            for j in range(self.nagents):
                if i != j:
                    ox, oy = self.agents[j].position
                    dist = math.sqrt((px-ox)*(px-ox)+(py-oy)*(py-oy))
                    dist -= 2.0 * AGENT_RADIUS
                    # Check whether a collision with another agent happens
                    if dist <= 5e-6: # TO BE FURTHER FIXED
                        # Restore previous position
                        px, py = self.agents[i].position
                        # Invert angle
                        #angle = -angle
                        # Change agent's orientation based on the rotation performed
                        if diff < 0.0:
                            # Agent rotated on the right -> make it go on the left
                            angle += math.pi / 2.0
                        elif diff > 0.0:
                            # Agent rotated on the left -> make it go on the right
                            angle -= math.pi / 2.0
                        else:
                            # Agent arrived at wall frontally -> invert its direction
                            angle += math.pi#-angle
                        angle = setAngleInRange(angle)
            # Update position and orientation of current agent
            self.agents[i].angle = angle
            self.agents[i].position = (px,py)        

        # Perform a world step
        self.world.Step(1.0/FPS, 6*30, 2*30)
        
        reward = 0.0
        fault = False
        # Check whether the agents collided with the objects
        for i in range(self.nagents):
            cobj = 0
            for o in self.objects:
                d, _ = self.distanceAndAngle(i, o, OBJECT_RADIUS)
                """
                if d <= 0.25:
                    print(i, cobj, d)
                """
                cobj += 1
                if d < 1e-6: # TO BE FIXED
                    if o.color1 == (0, 255, 0):
                        # Green objects are good
                        reward += 1.0
                    else:
                        # Red objects are bad, episode stops!!!
                        reward -= 10.0
                        fault = True
                    # Remove the object from the world
                    self.objects.remove(o)
                    self.world.DestroyBody(o)
        
        
        # Update step counter
        self.cstep += 1
        
        obs = self.getObs()
        
        #print(obs)
        #time.sleep(2)

        # Check whether the episode must be ended
        done = False
        
        # Check whether at least one of the agents touched a red object  
        if fault:
            # If one agent hit the red item, episode stops!!!
            done = True

        # Check the number of performed steps
        if self.cstep >= self.nsteps:
            done = True

        self.timer += 1
        
        if self.render_mode == "human":
            self.render()
        
        return np.array(obs, dtype=np.float32), reward, done, False, {}

    def getObs(self):
        # Fill the agent observation
        obs = []
        for i in range(self.nagents):
            px, py = self.agents[i].position
            angle = self.agents[i].angle
            for oid in range(self.nagents):
                if i != oid:
                    # Check whether robot detects its peers
                    robotsectors = []
                    robotdists = []
                    robotdist, robotrelang = self.distanceAndAngle(i, self.agents[oid], AGENT_RADIUS)
                    if robotdist <= MAX_DIST and (robotrelang >= -FOV / 2.0 and robotrelang <= FOV / 2.0):
                        # Store the sector
                        robotsector = self.calcSector(robotrelang)
                        robotsectors.append(robotsector)
                        robotdists.append(robotdist)
            # Check whether robot detects any object
            objsectors = []
            objdists = []
            objcolors = []
            objid = 0
            for obj in self.objects:
                objdist, objrelang = self.distanceAndAngle(i, obj, OBJECT_RADIUS)
                if objdist <= MAX_DIST and (objrelang >= -FOV / 2.0 and objrelang <= FOV / 2.0):
                    # Store the sector
                    objsector = self.calcSector(objrelang)
                    objsectors.append(objsector)
                    objdists.append(objdist)
                    objcolors.append(obj.color1)
                objid += 1
            # TO BE ADDED (check wall detection)?
            # Check if sectors overlap
            sectors = []
            dists = []
            colors = [(0, 0, 0)] * NUM_SECTORS
            for robs, dist in zip(robotsectors, robotdists):
                for s in robs:
                    if s not in sectors:
                        sectors.append(s)
                        dists.append(dist)
                        colors[s] = (0, 0, 255)
                    else:
                        idx = sectors.index(s)
                        cdist = dists[idx]
                        if dist < cdist:
                            dists[idx] = dist
                            colors[s] = (0, 0, 255)
            for objs, dist, color in zip(objsectors, objdists, objcolors):
                for s in objs:
                    if s not in sectors:
                        sectors.append(s)
                        dists.append(dist)
                        colors[s] = color
                    else:
                        idx = sectors.index(s)
                        cdist = dists[idx]
                        if dist < cdist:
                            dists[idx] = dist
                            colors[s] = color
            # Finally fill observations
            objlist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for s, d in zip(sectors, dists):
                if colors[s] == (255, 0, 0):
                    # Red
                    objlist[s] = (1.0 - d / MAX_DIST)
                elif colors[s] == (0, 255, 0):
                    # Green
                    objlist[NUM_SECTORS + s] = (1.0 - d / MAX_DIST)
                else:
                    # Blue
                    objlist[2 * NUM_SECTORS + s] = (1.0 - d / MAX_DIST)
            obs += objlist
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
                        # Draw the robot diameter orthogonally to its orientation
                        normAng = angle + math.pi / 2.0
                        normAng = setAngleInRange(normAng)
                        ex = sx + AGENT_RADIUS * SCALE * math.cos(normAng)
                        ey = sy + AGENT_RADIUS * SCALE * math.sin(normAng)
                        sx = sx - AGENT_RADIUS * SCALE * math.cos(normAng)
                        sy = sy - AGENT_RADIUS * SCALE * math.sin(normAng)
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

