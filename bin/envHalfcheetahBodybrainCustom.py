import sys, math
import numpy as np

from typing import TYPE_CHECKING, List, Optional

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

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
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 100.0

VIEWPORT_W = 1500
VIEWPORT_H = 600

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 40    # in steps
FRICTION = 2.5

LOCAL_SEED = 336699

NSTEPS = 100

MOD_RATE = 0.2

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
        if self.env.torso==contact.fixtureA.body or self.env.torso==contact.fixtureB.body:
            self.env.torso.ground_contact = True
        for seg in self.env.segs:
            if seg in [contact.fixtureA.body, contact.fixtureB.body]:
                seg.ground_contact = True
    def EndContact(self, contact):
        for seg in self.env.segs:
            if seg in [contact.fixtureA.body, contact.fixtureB.body]:
                seg.ground_contact = False

class customEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    # Task parameters
    numSegs = 7 # Number of segment does not include torso
    segSpeed = 5.0
    segWidth = 24.0
    segHeight = 6.0
    segDensity = 5.0
    jointRange = math.pi / 2.0
    initHeightFactor = 2.0
    torques = [120.0, 90.0, 60.0, 140.0, 60.0, 30.0]
    labels = ['head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
    # Body sizes
    body_param = [60.0, 6.0, 16.8, 6.0, 17.8, 6.0, 18.8, 6.0, 12.2, 6.0, 16.7, 6.0, 13.3, 6.0, 10.0, 6.0] # From half_cheetah.xml (pybullet_data/mjcf)
    # Joint angular ranges
    joint_param = [-0.52, 1.05, -0.785, 0.785, -0.4, 0.785, -1.5, 0.8, -1.2, 1.1, -3.1, -0.3] # From half_cheetah.xml (pybullet_data/mjcf)
    # Segment angles
    angle_param = [-1.002, -2.678, -1.271, -2.099, -0.945, -0.999] # From half_cheetah.xml (pybullet_data/mjcf)

    def __init__(self, render_mode: Optional[str] = None):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.torso = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )

        # Morphology parameters
        self.nsizes = 2 + (2 * self.numSegs)
        self.nangleranges = (2 * (self.numSegs - 1))
        self.nangles = (2 + self.numSegs)
        self.ndensities = (2 + self.numSegs)
        self.njoints = self.numSegs # number of joints matches number of segments (there is one joint between each pair of segments including torso)
        self.nparams = (self.nsizes + self.nangleranges + self.nangles + self.ndensities) # Torso width and height + 2 for each segment (width and height) + 2 for each joint (asymmetric angular ranges) except for the head + torso angle + one for each segment angle + torso density + one for each segment density. N.B.: number of joints matches number of segments (there is one joint between each pair of segments including torso)
        self.params = np.zeros(self.nparams, dtype=np.float64)#np.ones(self.nparams, dtype=np.float64)
        self.factors = np.ones(self.nparams, dtype=np.float64) # 2 for each segment (width and height) + 2 for each joint (asymmetric angular ranges) + torso angle + segment angle
        self.rate = 0.0
        # Set default number of steps
        self.nsteps = NSTEPS
        # Test flag
        self.test = False

        # Number of observations
        self.ob_len = 5 + (self.njoints * 2) + self.numSegs # 5 for torso, njoints * 2 for each joint, contact flag for each segment
        # Number of actions
        self.ac_len = self.njoints - 1# 1 action for each joint
        act = np.ones(self.ac_len, dtype=np.float32)
        high = np.array([np.inf] * self.ob_len, dtype=np.float32)
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
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.torso)
        self.torso = None
        for seg in self.segs:
            self.world.DestroyBody(seg)
        self.segs = []
        self.joints = []

    def _generate_terrain(self):
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS/2, TERRAIN_GRASS)

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def getNumParams(self):
        return self.nparams

    def setParams(self, params, rate=None):
        self.params = params
        # And the rate
        if rate is None:
            self.rate = MOD_RATE
        else:
            self.rate = rate

    def setTest(self):
        # Set test flag
        self.test = True

    def getMinY(self, y, size, a):
        y1 = y - size * math.sin(a)
        y2 = y + size * math.sin(a)
        miny = y1
        if y2 < miny:
            miny = y2
        return miny

    def lowestHeight(self):
        # Scale factor
        U = 1.0 / SCALE
        # Torso
        lowest = self.torso.position[1]
        j = 2
        # Other segments
        for i in range(self.numSegs):
            cy = self.segs[i].position[1]
            if cy < lowest:
                lowest = cy
            j += 2
        return lowest

    def computeFactor(self, val, idx):
        # Parameters list:
        # - torso width and height
        # - segment width and height (one pair for each segment)
        # - joint angular ranges (two for each joint)
        # - torso angle
        # - segment angles
        # - torso density
        # - segment density (one for each segment)
        factor = 0.0
        if idx < self.nsizes:
            # Segment sizes
            # Body sizes (width and height of segments) are scaled depending on the parameter
            # (1 + p * rate)
            factor = (val * (1.0 + np.tanh(self.params[idx]) * self.rate))
        elif idx >= (self.nsizes + self.nangleranges + self.nangles):
            # Segment densities
            # Density factors are computed as tanh(p), where p is the density parameter
            factor = (val * (1.0 + np.tanh(self.params[idx]) * self.rate))
        else:
            # Joint ranges and segment angles
            # Angle factors must belong to range [-1,1]. Values
            # outside boundaries are cut
            param = self.params[idx]
            if param < -1.0:
                param = -1.0
            if param > 1.0:
                param = 1.0
            factor = (param * val)
        return factor

    def reset(self, seed=None):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0

        self.timer = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain()
        self._generate_clouds()

        # Scale factor
        U = 1.0 / SCALE # 1/30
        
        # Copy of default body params scaled by morphological factors
        bodyParams = np.zeros(len(self.body_param), dtype=np.float64)
        for i in range(len(self.body_param)):
            bodyParams[i] = self.computeFactor(self.body_param[i], i)

        # Compute the leg heights
        bleg_h = bodyParams[1] * U + bodyParams[5] * U + bodyParams[7] * U + bodyParams[9] * U
        fleg_h = bodyParams[1] * U + bodyParams[11] * U + bodyParams[13] * U + bodyParams[15] * U
        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        # We use the longest leg to set the initial height of the cheetah
        max_h = bleg_h
        if fleg_h > max_h:
            max_h = fleg_h
        init_y = TERRAIN_HEIGHT + (self.initHeightFactor * max_h)

        self.segs = []
        self.joints = []

        # Torso sizes
        torsoWidth = bodyParams[0] * U
        torsoHeight = bodyParams[1] * U
        torsoAngle = 0.0 + self.computeFactor(math.pi / 4.0, (self.nsizes + self.nangleranges))
        # Create first segment (i.e., torso) out of the for loop
        self.torso = self.world.CreateDynamicBody(
                position = (init_x, init_y),
                angle = torsoAngle,
                fixtures = fixtureDef(
                            shape=polygonShape(box=(torsoWidth/2, torsoHeight/2)),
                            density=self.computeFactor(self.segDensity, self.nsizes + self.nangleranges + self.nangles),
                            restitution=0.0,
                            categoryBits=0x0020,
                            maskBits=0x001)
               )
        self.torso.color1 = (128,102,230)
        self.torso.color2 = (77,77,128)
        self.torso.ground_contact = False
        
        # Head sizes
        headWidth = bodyParams[2] * U
        headHeight = bodyParams[3] * U
        headAngle = math.pi / 4.0 + self.computeFactor(math.pi / 4.0, (self.nsizes + self.nangleranges) + 1)

        # Create head
        head = self.world.CreateDynamicBody(
                    position = (init_x + torsoWidth/2 * math.cos(torsoAngle) + headWidth/2 * math.cos(headAngle), init_y + torsoWidth/2 * math.sin(torsoAngle) + headWidth/2 * math.sin(headAngle)),
                    angle = headAngle,
                    fixtures = fixtureDef(
                                shape=polygonShape(box=(headWidth/2, headHeight/2)),
                                density=self.computeFactor(self.segDensity, self.nsizes + self.nangleranges + self.nangles + 1),
                                restitution=0.0,
                                categoryBits=0x0020,
                                maskBits=0x001)
                     )
        head.color1 = (255,26,51)
        head.color2 = (204,51,153)
        # Joint
        rjd = revoluteJointDef(
                bodyA=self.torso,
                bodyB=head,
                localAnchorA=(torsoWidth/2,0),
                localAnchorB=(-headWidth/2,-headHeight/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=0.0,
                motorSpeed = 0,
                lowerAngle = 0.0,
                upperAngle = 0.0,
               )
        head.ground_contact = False
        # Append segment to list of segments
        self.segs.append(head)
        # Append joint to list of joints
        self.joints.append(self.world.CreateJoint(rjd))

        # Indices for body params and joint params
        bi = 4
        ji = 0
        # Index for angles
        ai = 0
        h = self.nsizes # Joint ranges
        z = (self.nsizes + self.nangleranges + 2) # Relative angles
        w = (self.nsizes + self.nangleranges + self.nangles + 2) # Densities
        # Create front and back segments (thigh, shin and foot)
        for i in [-1,+1]:
            # Thigh
            thighWidth = bodyParams[bi] * U
            thighHeight = bodyParams[bi + 1] * U
            # Joint ranges
            thighLowerAngle = self.joint_param[ji] * (1.0 + self.computeFactor(self.rate, h))
            thighUpperAngle = self.joint_param[ji + 1] * (1.0 + self.computeFactor(self.rate, h + 1))
            thighAngle = self.angle_param[ai] + self.computeFactor(self.rate, z)
            thighAngle = setAngleInRange(thighAngle)
            # (x,y) coords for thigh
            thigh_x = init_x + (i * torsoWidth / 2) * math.cos(torsoAngle) + thighWidth / 2.0 * math.cos(thighAngle)
            thigh_y = init_y + (i * torsoWidth / 2) * math.sin(torsoAngle) + thighWidth / 2.0 * math.sin(thighAngle)
            thigh = self.world.CreateDynamicBody(
                    position = (thigh_x, thigh_y),
                    angle = thighAngle,
                    fixtures = fixtureDef(
                                shape=polygonShape(box=(thighWidth/2, thighHeight/2)),
                                density=self.computeFactor(self.segDensity, w),
                                restitution=0.0,
                                categoryBits=0x0020,
                                maskBits=0x001)
                     )
            thigh.color1 = (255,26,51)
            thigh.color2 = (204,51,153)
            # Joint
            rjd = revoluteJointDef(
                    bodyA=self.torso,
                    bodyB=thigh,
                    localAnchorA=((i * torsoWidth / 2), 0),
                    localAnchorB=(-thighWidth/2,0),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=MOTORS_TORQUE,
                    motorSpeed = i,
                    lowerAngle = thighLowerAngle,
                    upperAngle = thighUpperAngle,
                   )
            thigh.ground_contact = False
            # Append segment to list
            self.segs.append(thigh)
            # Append joint to list of joints
            self.joints.append(self.world.CreateJoint(rjd))
            # Update indices
            bi += 2
            ji += 2
            ai += 1
            h += 2
            z += 1
            w += 1
            # Shin
            shinWidth = bodyParams[bi] * U
            shinHeight = bodyParams[bi + 1] * U
            # Joint ranges
            shinLowerAngle = self.joint_param[ji] * (1.0 + self.computeFactor(self.rate, h))
            shinUpperAngle = self.joint_param[ji + 1] * (1.0 + self.computeFactor(self.rate, h + 1))
            shinAngle = self.angle_param[ai] * (1.0 + self.computeFactor(self.rate, z))
            shinAngle = setAngleInRange(shinAngle)
            # (x,y) coords for shin
            shin_x = thigh_x + thighWidth / 2.0 * math.cos(thighAngle) + shinWidth / 2.0 * math.cos(shinAngle)
            shin_y = thigh_y + thighWidth / 2.0 * math.sin(thighAngle) + shinWidth / 2.0 * math.sin(shinAngle)
            shin = self.world.CreateDynamicBody(
                    position = (shin_x, shin_y),
                    angle = shinAngle,
                    fixtures = fixtureDef(
                                shape=polygonShape(box=(shinWidth/2, shinHeight/2)),
                                density=self.computeFactor(self.segDensity, w),
                                restitution=0.0,
                                categoryBits=0x0020,
                                maskBits=0x001)
                     )
            shin.color1 = (128,102,230)
            shin.color2 = (77,77,128)
            # Joint
            rjd = revoluteJointDef(
                    bodyA=thigh,
                    bodyB=shin,
                    localAnchorA=(thighWidth/2,0),
                    localAnchorB=(-shinWidth/2,0),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=MOTORS_TORQUE,
                    motorSpeed = i,
                    lowerAngle = shinLowerAngle,
                    upperAngle = shinUpperAngle,
                   )
            shin.ground_contact = False
            # Append segment to list of segments
            self.segs.append(shin)
            # Append joint to list of joints
            self.joints.append(self.world.CreateJoint(rjd))
            # Update indices
            bi += 2
            ji += 2
            ai += 1
            h += 2
            z += 1
            w += 1
            # Foot
            footWidth = bodyParams[bi] * U
            footHeight = bodyParams[bi + 1] * U
            # Joint ranges
            footLowerAngle = self.joint_param[ji] * (1.0 + self.computeFactor(self.rate, h))
            footUpperAngle = self.joint_param[ji + 1] * (1.0 + self.computeFactor(self.rate, h + 1))
            footAngle = self.angle_param[ai] * (1.0 + self.computeFactor(self.rate, z))
            footAngle = setAngleInRange(footAngle)
            # (x,y) coords for foot
            foot_x = shin_x + shinWidth / 2.0 * math.cos(shinAngle) + footWidth / 2.0 * math.cos(footAngle)
            foot_y = shin_y + shinWidth / 2.0 * math.sin(shinAngle) + footWidth / 2.0 * math.sin(footAngle)
            foot = self.world.CreateDynamicBody(
                    position = (foot_x, foot_y),
                    angle = footAngle,
                    fixtures = fixtureDef(
                                shape=polygonShape(box=(footWidth/2, footHeight/2)),
                                density=self.computeFactor(self.segDensity, w),
                                restitution=0.0,
                                categoryBits=0x0020,
                                maskBits=0x001)
                     )
            foot.color1 = (255,26,51)
            foot.color2 = (204,51,153)
            # Joint
            rjd = revoluteJointDef(
                    bodyA=shin,
                    bodyB=foot,
                    localAnchorA=(shinWidth/2,0),
                    localAnchorB=(i * footWidth/2,0),#(-footWidth/2,0),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=MOTORS_TORQUE,
                    motorSpeed = i,
                    lowerAngle = footLowerAngle,
                    upperAngle = footUpperAngle,
                   )
            foot.ground_contact = False
            # Append segment to list of segments
            self.segs.append(foot)
            # Append joint to list of joints
            self.joints.append(self.world.CreateJoint(rjd))
            # Update indices
            bi += 2
            ji += 2
            ai += 1
            h += 2
            z += 1
            w += 1
            
        # Check number of joints
        assert len(self.joints) == self.njoints

        # List of objects to display
        self.drawlist = self.terrain + self.segs + [self.torso]

        fakeAction = [0.0 for _ in range(self.ac_len)]

        self.cstep = 0

        # Flags whether the robot is on the ground (i.e., at least one segment touches the ground)
        self.onGround = False
        """
        if self.render_mode == "human":
            self.render()
        """
        return self.step(np.array(fakeAction))[0], {} # self._step

    def step(self, action): # _step
        self.cstep += 1
        # Head joint (fixed)
        self.joints[0].motorSpeed = 0.0
        self.joints[0].maxMotorTorque = 0.0
        # Apply actions
        j = 1 # Head is fixed
        for i in range(len(action)):
            self.joints[j].motorSpeed = float(self.segSpeed * np.sign(action[i]))
            self.joints[j].maxMotorTorque = float(self.torques[i] * np.clip(np.abs(action[i]), 0, 1))
            j += 1
        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.torso.position
        vel = self.torso.linearVelocity

        # Add torso info
        state = [
            self.torso.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.torso.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            1.0 if self.torso.ground_contact else 0.0
            ]
        # Add joint info
        joint_state = []
        for i in range(self.njoints):
            joint_state += [(self.joints[i].angle)]
            joint_state += [self.joints[i].speed / self.segSpeed]
        state += joint_state
        # Add contact info
        seg_contact_state = []
        for i in range(self.numSegs):
            seg_contact_state += [1.0 if self.segs[i].ground_contact else 0.0]
        state += seg_contact_state

        # Check state array size
        assert len(state) == (5 + (self.njoints * 2) + self.numSegs)

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        #shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        progress = 0.0
        if self.prev_shaping is not None:
            progress = shaping - self.prev_shaping#reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward += progress

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            terminated = True

        # Check whether or not at least one segment touched the ground
        if self.torso.ground_contact:
            self.onGround = True
        else:
            for i in range(self.numSegs):
                if self.segs[i].ground_contact:
                    self.onGround = True
        
        # Security check
        if self.onGround:
            # Check whether or not the embryo has one segment below the terrain
            lowest = self.lowestHeight()
            if lowest < TERRAIN_HEIGHT:
                if self.test:
                    print("### LOWEST UNDER THRESHOLD ###")
                    print(lowest, TERRAIN_HEIGHT)
                terminated = True

        # Check stop conditions
        # Height check
        heightCheck = False
        if self.torso.position[1] < TERRAIN_HEIGHT + 0.3:
            heightCheck = True
            if self.test:
                print("### TORSO UNDER THRESHOLD ###")
                print(self.torso.position[1], TERRAIN_HEIGHT)
        # Torso angle out of boundaries
        torsoCheck = False
        if abs(self.torso.angle) > 1.0:
            torsoCheck = True
            if self.test:
                print("### TORSO ANGLE OUT OF RANGES ###")
                print(self.torso.angle)
        #"""
        if heightCheck or torsoCheck:
            terminated = True
        #"""
        # Check if other segments than feet touch the ground
        nTouchSegs = 0
        i = 0
        while i < self.numSegs:
            if i != 3 and i != 6:
                if self.segs[i].ground_contact:
                    nTouchSegs += 1

            i += 1
        # Now check if the torso touches the ground
        if self.torso.ground_contact:
            nTouchSegs += 1
        #reward -= (2.0 * nTouchSegs) # Penalty for other segments than feet touching the ground
        if nTouchSegs > 0:
            terminated = True

        # Compute joints at limit (to check the use of angle instead of position).
        # Definition taken from robot_locomotors.py and robot_bases.py files of pybullet
        nJointsAtLimit = 0
        for i in range(self.njoints):
            midAngle = 0.5 * (self.joints[i].lowerLimit + self.joints[i].upperLimit)
            try:
                jRelAng = (2.0 * (self.joints[i].angle - midAngle) / (self.joints[i].upperLimit - self.joints[i].lowerLimit))
            except:
                # In case of division by zero, we set jRelAng to 0
                jRelAng = 0.0
            if abs(jRelAng) > 0.99:#self.joints[i].angle < self.joints[i].lowerLimit or self.joints[i].angle > self.joints[i].upperLimit:
                nJointsAtLimit += 1
        reward -= (0.1 * nJointsAtLimit)

        if self.test:
            if terminated:
                print(self.cstep, self.nsteps, (self.nsteps - self.cstep))

        self.timer += 1
        """
        if self.render_mode == "human":
            self.render()
        """
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

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

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

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
        """
        flagy1 = TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        x = TERRAIN_STEP * 3 * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )
        """
        x = self.terrain_x[3] * SCALE
        flagy1 = self.terrain_y[3] * SCALE#TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        # Second flag marking the goal
        x = self.terrain_x[len(self.terrain_x)-3] * SCALE
        flagy1 = self.terrain_y[len(self.terrain_y)-3] * SCALE#TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(0, 51, 230), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
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

# Maybe can be eliminated
if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = customEnv()
    augment_vector = (1.0 + (np.random.rand(8)*2-1.0)*0.5)
    print("augment_vector", augment_vector)
    env.augment_env(augment_vector)
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break
