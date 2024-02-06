"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gym
"""

import logging
import math
from typing import TYPE_CHECKING, List, Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding
from gymnasium.envs.registration import register
import numpy as np
import cv2 # installed with gym anyways
from collections import deque
import pygame
from pygame import gfxdraw

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = True

REF_W = 12*2 # Width
REF_L = REF_W / 4 # Length
REF_H = REF_W*2 # Height
REF_U = 1.5 # ground height
REF_WALL_WIDTH = 1.0 # wall width
REF_WALL_LENGTH = REF_L
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10*1.75
PLAYER_SPEED_Y = 10*1.75
PLAYER_SPEED_Z = 10*1.35
MAX_BALL_SPEED = 15*1.5
TIMESTEP = 1/30.
NUDGE = 0.1
FRICTION = 1.0 # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8*2*1.5

MAXLIVES = 5 # game ends when one agent loses this many games
MAXEPS = 10 # maximum number of episodes

WINDOW_WIDTH = REF_W * 50#1200
WINDOW_HEIGHT = REF_H * 10#500
SCALE = 1.0

FACTOR = WINDOW_WIDTH / REF_W

# if set to true, renders using cv2 directly on numpy array
# (otherwise uses pyglet / opengl -> much smoother for human player)
PIXEL_MODE = False 
PIXEL_SCALE = 4 # first render at multiple of Pixel Obs resolution, then downscale. Looks better.

PIXEL_WIDTH = 84*2*1
PIXEL_HEIGHT = 84*1

def setNightColors():
  ### night time color:
  global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
  global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
  global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
  BALL_COLOR = (217, 79, 0)
  AGENT_LEFT_COLOR = (35, 93, 188)
  AGENT_RIGHT_COLOR = (255, 236, 0)
  PIXEL_AGENT_LEFT_COLOR = (255, 191, 0) # AMBER
  PIXEL_AGENT_RIGHT_COLOR = (255, 191, 0) # AMBER
  
  BACKGROUND_COLOR = (11, 16, 19)
  FENCE_COLOR = (102, 56, 35)
  COIN_COLOR = FENCE_COLOR
  GROUND_COLOR = (116, 114, 117)

def setDayColors():
  ### day time color:
  ### note: do not use day time colors for pixel-obs training.
  global BALL_COLOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR
  global PIXEL_AGENT_LEFT_COLOR, PIXEL_AGENT_RIGHT_COLOR
  global BACKGROUND_COLOR, FENCE_COLOR, COIN_COLOR, GROUND_COLOR
  global PIXEL_SCALE, PIXEL_WIDTH, PIXEL_HEIGHT
  PIXEL_SCALE = int(4*1.0)
  PIXEL_WIDTH = int(84*2*1.0)
  PIXEL_HEIGHT = int(84*1.0)
  BALL_COLOR = (255, 200, 20)
  AGENT_LEFT_COLOR = (240, 75, 0)
  AGENT_RIGHT_COLOR = (0, 150, 255)
  PIXEL_AGENT_LEFT_COLOR = (240, 75, 0)
  PIXEL_AGENT_RIGHT_COLOR = (0, 150, 255)
  
  BACKGROUND_COLOR = (255, 255, 255)
  FENCE_COLOR = (240, 210, 130)
  COIN_COLOR = FENCE_COLOR
  GROUND_COLOR = (128, 227, 153)

setNightColors()

# by default, don't load rendering (since we want to use it in headless cloud machines)
rendering = None
def checkRendering():
  global rendering
  if rendering is None:
    from gym.envs.classic_control import rendering as rendering

def setPixelObsMode():
  """
  used for experimental pixel-observation mode
  note: new dim's chosen to be PIXEL_SCALE (2x) as Pixel Obs dims (will be downsampled)

  also, both agent colors are identical, to potentially facilitate multiagent
  """
  global WINDOW_WIDTH, WINDOW_HEIGHT, FACTOR, AGENT_LEFT_COLOR, AGENT_RIGHT_COLOR, PIXEL_MODE
  PIXEL_MODE = True
  WINDOW_WIDTH = PIXEL_WIDTH * PIXEL_SCALE
  WINDOW_HEIGHT = PIXEL_HEIGHT * PIXEL_SCALE
  FACTOR = WINDOW_WIDTH / REF_W
  AGENT_LEFT_COLOR = PIXEL_AGENT_LEFT_COLOR
  AGENT_RIGHT_COLOR = PIXEL_AGENT_RIGHT_COLOR

def upsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH * PIXEL_SCALE, PIXEL_HEIGHT * PIXEL_SCALE), interpolation=cv2.INTER_NEAREST)
def downsize_image(img):
  return cv2.resize(img, (PIXEL_WIDTH, PIXEL_HEIGHT), interpolation=cv2.INTER_AREA)

# conversion from space to pixels (allows us to render to diff resolutions)
def toX(x):
  return (x+REF_W/2)*FACTOR
def toP(x):
  return (x)*FACTOR
def toY(y):
  return (y+REF_L/2)*FACTOR
def toZ(z):
  return z*FACTOR

class DelayScreen:
  """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """
  def __init__(self, life=INIT_DELAY_FRAMES):
    self.life = 0
    self.reset(life)
  def reset(self, life=INIT_DELAY_FRAMES):
    self.life = life
  def status(self):
    if (self.life == 0):
      return True
    self.life -= 1
    return False

def make_half_circle(surf, x, y, color, radius=10, res=20):
  """ helper function for pyglet renderer"""
  points = []
  for i in range(res+1):
    ang = math.pi-math.pi*i / res
    points.append((x + math.cos(ang)*radius, y + math.sin(ang)*radius))
  pygame.draw.polygon(surf, color=color, points=points)
  gfxdraw.aapolygon(surf, points, color)

def create_canvas(canvas, c):
  if PIXEL_MODE:
    result = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
      result[:, :, channel] *= c[channel]
    return result
  else:
    pygame.draw.polygon(
            canvas,
            color=BACKGROUND_COLOR,
            points=[
                (0, 0),
                (WINDOW_WIDTH, 0),
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                (0, WINDOW_HEIGHT),
            ],
        )
    return canvas

def rect(canvas, x, y, width, height, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    canvas = cv2.rectangle(canvas, (round(x), round(WINDOW_HEIGHT-y)),
      (round(x+width), round(WINDOW_HEIGHT-y+height)),
      color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas
  else:
    points = [(0,0), (0,-height), (width, -height), (width,0)]
    pygame.draw.polygon(canvas, color=color, points=points)
    gfxdraw.aapolygon(canvas, points, color)
    return canvas

def half_circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    return cv2.ellipse(canvas, (round(x), WINDOW_HEIGHT-round(y)),
      (round(r), round(r)), 0, 0, -180, color, thickness=-1, lineType=cv2.LINE_AA)
  else:
    canvas = make_half_circle(canvas, x, y, color, radius=r)
    return canvas

def circle(canvas, x, y, r, color):
  """ Processing style function to make it easy to port p5.js program to python """
  if PIXEL_MODE:
    return cv2.circle(canvas, (round(x), round(WINDOW_HEIGHT-y)), round(r),
      color, thickness=-1, lineType=cv2.LINE_AA)
  else:
    pygame.draw.circle(
      canvas,
      color=color,
      center=(x,y),
      radius=r,
    )
    return canvas

class Particle:
  """ used for the ball, and also for the round stub above the fence """
  def __init__(self, x, y, z, vx, vy, vz, r, c):
    self.x = x
    self.y = y
    self.z = z
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z
    self.vx = vx
    self.vy = vy
    self.vz = vz
    self.r = r
    self.c = c
  def display(self, canvas):
    return circle(canvas, toX(self.x), toZ(self.z), toP(self.r), color=self.c)
  def display2(self, canvas):
    zrange = (REF_W/4 - REF_U) / 3 # Taken from game initialization!!!
    # Change color depending on the z coord
    bcolor = self.c
    if self.z >= REF_U and self.z < (REF_U + zrange):
        bcolor = (255, 0, 0)
    elif self.z >= (REF_U + zrange) and self.z < (REF_U + 2.0 * zrange):
        bcolor = (236, 39, 0)
    else:
        bcolor = self.c
    # Use a different color for fenceStub
    if self.z == self.prev_z:
        bcolor = self.c
    canvas = circle(canvas, toX(self.x), toY(self.y), toP(self.r), color=bcolor)#self.c)
    return canvas
  def move(self):
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.z += self.vz * TIMESTEP
  def applyAcceleration(self, ax, ay, az):
    self.vx += ax * TIMESTEP
    self.vy += ay * TIMESTEP
    self.vz += az * TIMESTEP
  def checkEdges(self):
    if (self.x<=(self.r-REF_W/2)):
      self.vx *= -FRICTION
      self.x = self.r-REF_W/2+NUDGE*TIMESTEP

    if (self.x >= (REF_W/2-self.r)):
      self.vx *= -FRICTION;
      self.x = REF_W/2-self.r-NUDGE*TIMESTEP

    if (self.y<=(self.r-REF_L/2)):
      self.vy *= -FRICTION
      self.y = self.r-REF_L/2+NUDGE*TIMESTEP

    if (self.y >= (REF_L/2-self.r)):
      self.vy *= -FRICTION;
      self.y = REF_L/2-self.r-NUDGE*TIMESTEP

    if (self.z<=(self.r+REF_U)):
      self.vz *= -FRICTION
      self.z = self.r+REF_U+NUDGE*TIMESTEP
      if (self.x <= 0):
        return -1
      else:
        return 1
    if (self.z >= (REF_H-self.r)):
      self.vz *= -FRICTION
      self.z = REF_H-self.r-NUDGE*TIMESTEP
    # fence:
    if ((self.x <= (REF_WALL_WIDTH/2+self.r)) and (self.prev_x > (REF_WALL_WIDTH/2+self.r)) and (self.z <= (REF_WALL_HEIGHT))):
      self.vx *= -FRICTION
      self.x = REF_WALL_WIDTH/2+self.r+NUDGE*TIMESTEP

    if ((self.x >= (-REF_WALL_WIDTH/2-self.r)) and (self.prev_x < (-REF_WALL_WIDTH/2-self.r)) and (self.z <= (REF_WALL_HEIGHT))):
      self.vx *= -FRICTION
      self.x = -REF_WALL_WIDTH/2-self.r-NUDGE*TIMESTEP
    return 0;
  def getDist2(self, p): # returns distance squared from p
    dy = p.y - self.y
    dx = p.x - self.x
    dz = p.z - self.z
    return (dx*dx+dy*dy+dz*dz)
  def isColliding(self, p): # returns true if it is colliding w/ p
    r = self.r+p.r
    return (r*r > self.getDist2(p)) # if distance is less than total radius, then colliding.
  def bounce(self, p): # bounce two balls that have collided (this and that)
    abx = self.x-p.x
    aby = self.y-p.y
    abz = self.z-p.z
    abd = math.sqrt(abx*abx+aby*aby+abz*abz)
    abx /= abd # normalize
    aby /= abd
    abz /= abd
    nx = abx # reuse calculation
    ny = aby
    nz = abz
    abx *= NUDGE
    aby *= NUDGE
    abz *= NUDGE
    while(self.isColliding(p)):
      self.x += abx
      self.y += aby
      self.z += abz
    ux = self.vx - p.vx
    uy = self.vy - p.vy
    uz = self.vz - p.vz
    un = ux*nx + uy*ny + uz*nz
    unx = nx*(un*2.) # added factor of 2
    uny = ny*(un*2.) # added factor of 2
    unz = nz*(un*2.) # added factor of 2
    ux -= unx
    uy -= uny
    uz -= unz
    self.vx = ux + p.vx
    self.vy = uy + p.vy
    self.vz = uz + p.vz
  def limitSpeed(self, minSpeed, maxSpeed):
    mag2 = self.vx*self.vx+self.vy*self.vy+self.vz*self.vz;
    if (mag2 > (maxSpeed*maxSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vz /= mag
      self.vx *= maxSpeed
      self.vy *= maxSpeed
      self.vz *= maxSpeed

    if (mag2 < (minSpeed*minSpeed) ):
      mag = math.sqrt(mag2)
      self.vx /= mag
      self.vy /= mag
      self.vz /= mag
      self.vx *= minSpeed
      self.vy *= minSpeed
      self.vz *= minSpeed

class Wall:
  """ used for the fence, and also the ground """
  def __init__(self, x, y, z, w, l, h, c):
    self.x = x
    self.y = y
    self.z = z
    self.w = w
    self.l = l
    self.h = h
    self.c = c
  def display(self, canvas):
    return rect(canvas, toX(self.x-self.w/2), toZ(self.z+self.h/2), toP(self.w), toP(self.h), color=self.c)
  def display2(self, canvas):
    return rect(canvas, toX(self.x-self.w/2), toY(self.y+self.l/2), toP(self.w), toP(self.l), color=self.c)

class RelativeState:
  """
  keeps track of the obs.
  Note: the observation is from the perspective of the agent.
  an agent playing either side of the fence must see obs the same way
  """
  def __init__(self):
    # agent
    self.x = 0
    self.y = 0
    self.z = 0
    self.vx = 0
    self.vy = 0
    self.vz = 0
    # ball
    self.bx = 0
    self.by = 0
    self.bz = 0
    self.bvx = 0
    self.bvy = 0
    self.bvz = 0
    # opponent
    self.ox = 0
    self.oy = 0
    self.oz = 0
    self.ovx = 0
    self.ovy = 0
    self.ovz = 0
  def getObservation(self):
    result = [self.x, self.y, self.z, self.vx, self.vy, self.vz,
              self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
              self.ox, self.oy, self.oz, self.ovx, self.ovy, self.ovz]
    scaleFactor = 10.0  # scale inputs to be in the order of magnitude of 10 for neural network.
    result = np.array(result) / scaleFactor
    return result

class Agent:
  """ keeps track of the agent in the game. note this is not the policy network """
  def __init__(self, direction, x, y, z, c):
    self.dir = direction # -1 means left, 1 means right player for symmetry.
    self.x = x
    self.y = y
    self.z = z
    self.r = 1.5
    self.c = c
    self.vx = 0
    self.vy = 0
    self.vz = 0
    self.desired_vx = 0
    self.desired_vy = 0
    self.desired_vz = 0
    self.state = RelativeState()
    self.emotion = "happy"; # hehe...
    self.life = MAXLIVES
  def lives(self):
    return self.life
  def setAction(self, action):
    forward = False
    backward = False
    left = False
    right = False
    jump = False
    if action[0] > 0:
      forward = True
    if action[1] > 0:
      backward = True
    if action[2] > 0:
      left = True
    if action[3] > 0:
      right = True
    if action[4] > 0:
      jump = True
    self.desired_vx = 0
    self.desired_vy = 0
    self.desired_vz = 0
    if (forward and (not backward)):
      self.desired_vx = -PLAYER_SPEED_X
    if (backward and (not forward)):
      self.desired_vx = PLAYER_SPEED_X
    if (left and (not right)):
      self.desired_vy = -PLAYER_SPEED_Y
    if (right and (not left)):
      self.desired_vy = PLAYER_SPEED_Y
    if jump:
      self.desired_vz = PLAYER_SPEED_Z
  def move(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.z += self.vz * TIMESTEP
  def step(self):
    self.x += self.vx * TIMESTEP
    self.y += self.vy * TIMESTEP
    self.z += self.vz * TIMESTEP
  def update(self): # TO BE FIXED (SOMETHING WRONG/MISSING!!!)
    self.vz += GRAVITY * TIMESTEP

    if (self.z <= REF_U + NUDGE*TIMESTEP):
      self.vz = self.desired_vz

    self.vx = self.desired_vx*self.dir
    self.vy = self.desired_vy*self.dir

    self.move()

    if (self.z <= REF_U):
      self.z = REF_U;
      self.vz = 0;

    # stay in their own half:
    if (self.x*self.dir <= (REF_WALL_WIDTH/2+self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_WALL_WIDTH/2+self.r)

    if (self.x*self.dir >= (REF_W/2-self.r) ):
      self.vx = 0;
      self.x = self.dir*(REF_W/2-self.r)
    # and do not go too left/right!!!
    if (self.y*self.dir <= (self.r-REF_L/2) ):
      self.vy = 0;
      self.y = self.dir*(self.r-REF_L/2)

    if (self.y*self.dir >= (REF_L/2-self.r) ):
      self.vy = 0;
      self.y = self.dir*(REF_L/2-self.r)

  def updateState(self, ball, opponent):
    """ normalized to side, appears different for each agent's perspective"""
    # agent's self
    self.state.x = self.x*self.dir
    self.state.y = self.y*self.dir
    self.state.z = self.z
    self.state.vx = self.vx*self.dir
    self.state.vy = self.vy*self.dir
    self.state.vz = self.vz
    # ball
    self.state.bx = ball.x*self.dir
    self.state.by = ball.y*self.dir
    self.state.bz = ball.z
    self.state.bvx = ball.vx*self.dir
    self.state.bvy = ball.vy*self.dir
    self.state.bvz = ball.vz
    # opponent
    self.state.ox = opponent.x*(-self.dir)
    self.state.oy = opponent.y*(-self.dir)
    self.state.oz = opponent.z
    self.state.ovx = opponent.vx*(-self.dir)
    self.state.ovy = opponent.vy*(-self.dir)
    self.state.ovz = opponent.vz
  def getObservation(self):
    return self.state.getObservation()

  def display(self, canvas, bx, bz):
    x = self.x
    z = self.z
    r = self.r

    angle = math.pi * 60 / 180
    if self.dir == 1:
      angle = math.pi * 120 / 180
    eyeX = 0
    eyeY = 0

    canvas = half_circle(canvas, toX(x), toZ(z), toP(r), color=self.c)

    # track ball with eyes (replace with observed info later):
    c = math.cos(angle)
    s = math.sin(angle)
    ballX = bx-(x+(0.6)*r*c);
    ballZ = bz-(z+(0.6)*r*s);

    if (self.emotion == "sad"):
      ballX = -self.dir
      ballZ = -3

    dist = math.sqrt(ballX*ballX+ballZ*ballZ)
    eyeX = ballX/dist
    eyeZ = ballZ/dist

    canvas = circle(canvas, toX(x+(0.6)*r*c), toZ(z+(0.6)*r*s), toP(r)*0.3, color=(255, 255, 255))
    canvas = circle(canvas, toX(x+(0.6)*r*c+eyeX*0.15*r), toZ(z+(0.6)*r*s+eyeY*0.15*r), toP(r)*0.1, color=(0, 0, 0))

    # draw coins (lives) left
    """for i in range(1, eps):#self.life):
      canvas = circle(canvas, toX(self.dir*(REF_W/2+0.05-i)), WINDOW_HEIGHT-toZ(1.5), toP(0.5), color=COIN_COLOR) #toX(self.dir*(REF_W/2+0.5-i*2.)), WINDOW_HEIGHT-toZ(1.5), toP(0.5), color=COIN_COLOR)"""
    for i in range(1, self.life):
      canvas = circle(canvas, toX(self.dir*(REF_W/2+0.5-i*2.)), WINDOW_HEIGHT-toZ(1.5), toP(0.5), color=COIN_COLOR)

    return canvas

  def display2(self, canvas):
    x = self.x
    y = self.y
    z = self.z
    r = self.r

    # Agent colors
    """
    AGENT_LEFT_COLOR = (35, 93, 188)
    AGENT_RIGHT_COLOR = (255, 236, 0)
    """

    zrange = (REF_W/8 - REF_U) / 3 # Taken from game initialization!!!
    # Change color depending on the z coord
    acolor = self.c
    if self.z >= REF_U and self.z < (REF_U + zrange):
      acolor = self.c
    elif self.z >= (REF_U + zrange) and self.z < (REF_U + 2.0 * zrange):
      if self.dir == 1:
        acolor = (245, 245, 0)
      else:
        acolor = (35,103,178)
    else:
      if self.dir == 1:
        acolor = (236, 255, 0)
      else:
        acolor = (35,113,168)

    canvas = circle(canvas, toX(x), toY(y), toP(r), color=acolor)

    return canvas

class BaselinePolicy:
  """ Tiny RNN policy with only 120 parameters of otoro.net/slimevolley agent """
  def __init__(self):
    self.nGameInput = 12 # 12 states for agent
    self.nGameOutput = 5 # 5 buttons (forward, backward, left, right, jump)
    self.nRecurrentState = 4 # extra recurrent states for feedback.

    self.nOutput = self.nGameOutput+self.nRecurrentState
    self.nInput = self.nGameInput+self.nOutput
    
    # store current inputs and outputs
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)
    self.prevOutputState = np.zeros(self.nOutput)

    """See training details: https://blog.otoro.net/2015/03/28/neural-slime-volleyball/ """
    """
    self.weight = np.array(
      [7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034, 0.3935, 1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689,
       1.646, -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451, 0.3987, -2.9501, -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887,
       2.4586, 13.4191, 2.7395, -3.9708, 1.6548, -2.7554, -1.5345, -6.4708, 9.2426, -0.7392, 0.4452, 1.8828, -2.6277, -10.851, -3.2353,
       -4.4653, -3.1153, -1.3707, 7.318, 16.0902, 1.4686, 7.0391, 1.7765, -1.155, 2.6697, -8.8877, 1.1958, -3.2839, -5.4425, 1.6809,
       7.6812, -2.4732, 1.738, 0.3781, 0.8718, 2.5886, 1.6911, 1.2953, -9.0052, -4.6038, -6.7447, -2.5528, 0.4391, -4.9278, -3.6695,
       -4.8673, -1.6035, 1.5011, -5.6124, 4.9747, 1.8998, 3.0359, 6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459, 4.7134, 2.8952,
       -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596, 0.1918, 3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977, -0.9377])

    self.bias = np.array([2.2935,-2.0353,-1.7786,5.4567,-3.6368,3.4996,-0.0685])
    """
    prange = 10.0
    self.weight = np.random.uniform(-prange, prange, self.nInput * self.nOutput)
    self.bias = np.random.uniform(-prange, prange, self.nOutput)
    # unflatten weight, convert it into 7x15 matrix.
    self.weight = self.weight.reshape(self.nGameOutput+self.nRecurrentState,
      self.nGameInput+self.nGameOutput+self.nRecurrentState)
  def reset(self):
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)
    self.prevOutputState = np.zeros(self.nOutput)
  def _forward(self):
    self.prevOutputState = self.outputState
    self.outputState = np.tanh(np.dot(self.weight, self.inputState)+self.bias)
  def _setInputState(self, obs):
    # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
    [x, y, z, vx, vy, vz, ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz, op_x, op_y, op_z, op_vx, op_vy, op_vz] = obs
    self.inputState[0:self.nGameInput] = np.array([x, y, z, vx, vy, vz, ball_x, ball_y, ball_z, ball_vx, ball_vy, ball_vz])
    self.inputState[self.nGameInput:] = self.outputState
  def _getAction(self):
    forward = 0
    backward = 0
    left = 0
    right = 0
    jump = 0
    if (self.outputState[0] > 0.75):
      forward = 1
    if (self.outputState[1] > 0.75):
      backward = 1
    if (self.outputState[2] > 0.75):
      left = 1
    if (self.outputState[3] > 0.75):
      right = 1
    if (self.outputState[4] > 0.75):
      jump = 1
    return [forward, backward, left, right, jump]
  def predict(self, obs):
    """ take obs, update rnn state, return action """
    self._setInputState(obs)
    self._forward()
    return self._getAction()

class Game:
  """
  the main slime volley game.
  can be used in various settings, such as ai vs ai, ai vs human, human vs human
  """
  def __init__(self, np_random=np.random):
    self.ball = None
    self.ground = None
    self.fence = None
    self.fenceStub = None
    self.agent_left = None
    self.agent_right = None
    self.delayScreen = None
    self.np_random = np_random
    self.counter = 0
    self.prev_ball_vx = None
    self.prev_ball_vy = None
    self.prev_ball_vz = None
    self.reset()
  def reset(self):
    self.ground = Wall(0, 0, 0.75, REF_W, REF_L, REF_U, c=GROUND_COLOR)
    self.fence = Wall(0, 0, 0.75 + REF_WALL_HEIGHT/2, REF_WALL_WIDTH, REF_WALL_LENGTH, (REF_WALL_HEIGHT-1.5), c=FENCE_COLOR)
    self.fenceStub = []
    cy = -REF_L/2
    cnt = 0
    while cy <= REF_L/2:
        self.fenceStub.append(Particle(0, cy, REF_WALL_HEIGHT, 0, 0, 0, REF_WALL_WIDTH/2, c=FENCE_COLOR))
        cy += REF_WALL_WIDTH/8#4
        cnt += 1
    vx2 = self.np_random.uniform(low=1.25, high=10)
    ball_vx = self.np_random.choice([vx1, vx2])#self.np_random.uniform(low=-10, high=10) # -20 20
    ball_vy = self.np_random.uniform(low=-2, high=2)
    ball_vz = self.np_random.uniform(low=10, high=15) # 10 25
    # Save previous ball velocities
    self.prev_ball_vx = ball_vx
    self.prev_ball_vy = ball_vy
    self.prev_ball_vz = ball_vz
    self.ball = Particle(0, 0, REF_W/4, ball_vx, ball_vy, ball_vz, 0.5, c=BALL_COLOR);
    self.agent_left = Agent(-1, -REF_W/4, 0.0, 1.5, c=AGENT_LEFT_COLOR)
    self.agent_right = Agent(1, REF_W/4, 0.0, 1.5, c=AGENT_RIGHT_COLOR)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)
    self.delayScreen = DelayScreen()
    self.counter = 0
    self.done = False
  def newMatch(self):
    if self.counter % 2 != 0:
      vx1 = self.np_random.uniform(low=-10, high=-1.25)
      vx2 = self.np_random.uniform(low=1.25, high=10)
      ball_vx = self.np_random.choice([vx1, vx2])#ball_vx = self.np_random.uniform(low=-10, high=10)
      ball_vy = self.np_random.uniform(low=-2, high=2)
      ball_vz = self.np_random.uniform(low=10, high=15)
      # Save previous ball velocities
      self.prev_ball_vx = ball_vx
      self.prev_ball_vy = ball_vy
      self.prev_ball_vz = ball_vz
    else:
      # Use symmetric values of previous ball velocities
      ball_vx = -self.prev_ball_vx
      ball_vy = -self.prev_ball_vy
      ball_vz = self.prev_ball_vz
    self.ball = Particle(0, 0, REF_W/4, ball_vx, ball_vy, ball_vz, 0.5, c=BALL_COLOR)
    self.delayScreen.reset()
    self.counter += 1
    self.done = False
  def step(self):
    """ main game loop """

    self.betweenGameControl()
    self.agent_left.update()
    self.agent_right.update()

    if self.delayScreen.status():
      self.ball.applyAcceleration(0, 0, GRAVITY)
      self.ball.limitSpeed(0, MAX_BALL_SPEED)
      self.ball.move()

    if (self.ball.isColliding(self.agent_left)):
      self.ball.bounce(self.agent_left)
    if (self.ball.isColliding(self.agent_right)):
      self.ball.bounce(self.agent_right)
    for elem in self.fenceStub:
      if (self.ball.isColliding(elem)):#self.fenceStub)):
        self.ball.bounce(elem)#self.fenceStub)
    # negated, since we want reward to be from the persepctive of right agent being trained.
    result = -self.ball.checkEdges()

    if (result != 0):
      if self.counter < MAXEPS - 1: # Last valid episode is when counter = MAXEPS - 1
        self.newMatch() # not reset, but after a point is scored
      else:
        self.done = True
      if result < 0: # baseline agent won
        self.agent_left.emotion = "happy"
        self.agent_right.emotion = "sad"
        self.agent_right.life -= 1
      else:
        self.agent_left.emotion = "sad"
        self.agent_right.emotion = "happy"
        self.agent_left.life -= 1
      return result

    # update internal states (the last thing to do)
    self.agent_left.updateState(self.ball, self.agent_right)
    self.agent_right.updateState(self.ball, self.agent_left)

    return result
  def display(self, canvas):
    # background color
    # if PIXEL_MODE is True, canvas is an RGB array.
    # if PIXEL_MODE is False, canvas is viewer object
    canvas = create_canvas(canvas, c=BACKGROUND_COLOR)
    canvas = self.fence.display(canvas)
    canvas = self.fenceStub[3].display(canvas)
    canvas = self.agent_left.display(canvas, self.ball.x, self.ball.z)
    canvas = self.agent_right.display(canvas, self.ball.x, self.ball.z)
    canvas = self.ball.display(canvas)
    canvas = self.ground.display(canvas)
    return canvas
  def betweenGameControl(self):
    agent = [self.agent_left, self.agent_right]
    if (self.delayScreen.life > 0):
      pass
      '''
      for i in range(2):
        if (agent[i].emotion == "sad"):
          agent[i].setAction([0, 0, 0]) # nothing
      '''
    else:
      agent[0].emotion = "happy"
      agent[1].emotion = "happy"

  def display2(self, canvas):
    canvas = create_canvas(canvas, c=(255, 255, 255))
    canvas = self.fence.display2(canvas)
    for elem in self.fenceStub:
        canvas = elem.display2(canvas)
    canvas = self.agent_left.display2(canvas)
    canvas = self.agent_right.display2(canvas)
    canvas = self.ball.display2(canvas)
    
    return canvas

  def isDone(self):
    return self.done

class SlimeVolleyHardEnv(gym.Env):
  """
  Gym wrapper for Slime Volley game.

  By default, the agent you are training controls the right agent
  on the right. The agent on the left is controlled by the baseline
  RNN policy.

  Game ends when an agent loses 5 matches (or at t=3000 timesteps).

  Note: Optional mode for MARL experiments, like self-play which
  deviates from Gym env. Can be enabled via supplying optional action
  to override the default baseline agent's policy:

  obs1, reward, done, info = env.step(action1, action2)

  the next obs for the right agent is returned in the optional
  fourth item from the step() method.

  reward is in the perspective of the right agent so the reward
  for the left agent is the negative of this number.
  """
  metadata = {
    'render.modes': ['human', 'rgb_array', 'state'],
    'video.frames_per_second' : 50
  }

  # for compatibility with typical atari wrappers
  atari_action_meaning = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
  }
  atari_action_set = {
    0, # NOOP
    4, # LEFT
    7, # UPLEFT
    2, # UP
    6, # UPRIGHT
    3, # RIGHT
  }

  action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)

  from_pixels = False
  atari_mode = False
  survival_bonus = False # Depreciated: augment reward, easier to train
  multiagent = True # optional args anyways

  def __init__(self, render_mode: Optional[str] = None):
    """
    Reward modes:

    net score = right agent wins minus left agent wins

    0: returns net score (basic reward)
    1: returns 0.01 x number of timesteps (max 3000) (survival reward)
    2: sum of basic reward and survival reward

    0 is suitable for evaluation, while 1 and 2 may be good for training

    Setting multiagent to True puts in info (4th thing returned in stop)
    the otherObs, the observation for the other agent. See multiagent.py

    Setting self.from_pixels to True makes the observation with multiples
    of 84, since usual atari wrappers downsample to 84x84
    """

    self.t = 0
    self.t_limit = 3000

    #self.action_space = spaces.Box(0.0, 1.0, shape=(5,))
    if self.atari_mode:
      self.action_space = spaces.Discrete(10)
    else:
      self.action_space = spaces.MultiBinary(5)
    self.ac_len = 5

    if self.from_pixels:
      setPixelObsMode()
      self.observation_space = spaces.Box(low=0, high=255,
        shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
    else:
      high = np.array([np.finfo(np.float32).max] * 18)
      self.observation_space = spaces.Box(-high, high)
    self.ob_len = 18
    self.canvas = None
    self.previous_rgbarray = None

    self.game = Game()
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function

    self.policy = BaselinePolicy() # the “bad guy”

    # another avenue to override the built-in AI's action, going past many env wraps:
    self.otherAction = None
    
    self.render_mode = render_mode
    self.screen: Optional[pygame.Surface] = None
    self.screen2: Optional[pygame.Surface] = None
    self.clock = None
    self.clock2 = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.game = Game(np_random=self.np_random)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    return [seed]

  def getObs(self):
    if self.from_pixels:
      obs = self.render(mode='state')
      otherObs = cv2.flip(obs, 1) # horizontal flip
      self.canvas = obs
    else:
      obs = self.game.agent_right.getObservation()
      otherObs = self.game.agent_left.getObservation()
    return np.concatenate((obs, otherObs))

  def discreteToBox(self, n):
    # convert discrete action n into the actual triplet action
    if isinstance(n, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
      if len(n) == 5:#3:
        return n
    assert (int(n) == n) and (n >= 0) and (n < 10)#6)
    return self.action_table[n]

  # Modified original step to deal with an issue concerning the number of arguments of step() method!!!
  # This method sets the action for the second agent, so to overcome the issue and use different actions for the two players
  def setOtherAction(self, otherAction):
    self.otherAction = otherAction

  def step(self, action):#, otherAction=None):
  #def step(self, action, otherAction=None):
    """
    baseAction is only used if multiagent mode is True
    note: although the action space is multi-binary, float vectors
    are fine (refer to setAction() to see how they get interpreted)
    """
    done = False
    self.t += 1

    otherAction = None
    if self.otherAction is not None:
      otherAction = self.otherAction
      
    if otherAction is None: # override baseline policy
      obs = self.game.agent_left.getObservation()
      otherAction = self.policy.predict(obs)

    if self.atari_mode:
      action = self.discreteToBox(action)
      otherAction = self.discreteToBox(otherAction)

    self.game.agent_left.setAction(action[self.ac_len:])#otherAction)
    self.game.agent_right.setAction(action[0:self.ac_len]) # external agent is agent_right

    reward = self.game.step()

    obs = self.getObs()

    if self.t >= self.t_limit:
      done = True

    if self.game.isDone():#self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
      done = True

    otherObs = None
    if self.multiagent:
      if self.from_pixels:
        otherObs = cv2.flip(obs, 1) # horizontal flip
      else:
        otherObs = self.game.agent_left.getObservation()

    info = {
      'ale.lives': self.game.agent_right.lives(),
      'ale.otherLives': self.game.agent_left.lives(),
      'otherObs': otherObs,
      'state': self.game.agent_right.getObservation(),
      'otherState': self.game.agent_left.getObservation(),
    }

    if self.survival_bonus:
      return np.concatenate((obs, otherObs)), reward+0.01, done, False, info
    return np.concatenate((obs, otherObs)), reward, done, False, info

  def init_game_state(self):
    self.t = 0
    self.game.reset()

  def reset(self, seed=None):
    self.init_game_state()
    if self.render_mode == "human":
        self.render()
    return self.getObs(), {}

  def render(self, mode='human', close=False):

    # pygame renderer
    if self.screen is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    if self.clock is None:
      self.clock = pygame.time.Clock()
    self.surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.transform.scale(self.surf, (SCALE, SCALE))

    self.game.display(self.surf)
    self.surf = pygame.transform.flip(self.surf, False, True)

    if self.render_mode == "human":
      assert self.screen is not None
      self.screen.blit(self.surf, (0, 0))
      pygame.event.pump()
      self.clock.tick(self.metadata["render_fps"])
      pygame.display.flip()
    elif self.render_mode == "rgb_array":
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
      )[:, -WINDOW_WIDTH:]

  def render2(self, mode='human', close=False):
    
    # pygame renderer
    if self.screen2 is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.screen2 = pygame.display.set_mode((WINDOW_WIDTH, int(WINDOW_WIDTH / 4)))
    if self.clock2 is None:
      self.clock2 = pygame.time.Clock()
    self.surf2 = pygame.Surface((WINDOW_WIDTH, int(WINDOW_WIDTH / 4)))
    pygame.transform.scale(self.surf2, (SCALE, SCALE))

    self.game.display2(self.surf2)
    self.surf2 = pygame.transform.flip(self.surf2, False, True)

    if self.render_mode == "human":
      assert self.screen2 is not None
      self.screen2.blit(self.surf2, (0, 0))
      pygame.event.pump()
      self.clock2.tick(self.metadata["render_fps"])
      pygame.display.flip()
    elif self.render_mode == "rgb_array":
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(self.surf2)), axes=(1, 0, 2)
      )[:, -WINDOW_WIDTH:]

  def close(self):
    pass
      
class SlimeVolleyHardLargeFieldEnv(SlimeVolleyHardEnv):
  REF_W *= 2
  REF_L *= 2

#####################
# helper functions: #
#####################

def multiagent_rollout(env, policy_right, policy_left, render_mode=False):
  """
  play one agent vs the other in modified gym-style loop.
  important: returns the score from perspective of policy_right.
  """
  obs_right = env.reset()
  obs_left = obs_right # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  t = 0

  while not done:

    action_right = policy_right.predict(obs_right)
    action_left = policy_left.predict(obs_left)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs_right, reward, done, truncated, info = env.step(action_right, action_left)
    obs_left = info['otherObs']

    total_reward += reward
    t += 1

    if render_mode:
      env.render()

  return total_reward, t

def render_atari(obs):
  """
  Helper function that takes in a processed obs (84,84,4)
  Useful for visualizing what an Atari agent actually *sees*
  Outputs in Atari visual format (Top: resized to orig dimensions, buttom: 4 frames)
  """
  tempObs = []
  obs = np.copy(obs)
  for i in range(4):
    if i == 3:
      latest = np.copy(obs[:, :, i])
    if i > 0: # insert vertical lines
      obs[:, 0, i] = 141
    tempObs.append(obs[:, :, i])
  latest = np.expand_dims(latest, axis=2)
  latest = np.concatenate([latest*255.0] * 3, axis=2).astype(np.uint8)
  latest = cv2.resize(latest, (84 * 8, 84 * 4), interpolation=cv2.INTER_NEAREST)
  tempObs = np.concatenate(tempObs, axis=1)
  tempObs = np.expand_dims(tempObs, axis=2)
  tempObs = np.concatenate([tempObs*255.0] * 3, axis=2).astype(np.uint8)
  tempObs = cv2.resize(tempObs, (84 * 8, 84 * 2), interpolation=cv2.INTER_NEAREST)
  return np.concatenate([latest, tempObs], axis=0)

####################
# Reg envs for gym #
####################

register(
    id='SlimeVolleyHard-v0',
    entry_point='slimevolleyhard:SlimeVolleyHardEnv'
)

register(
    id='SlimeVolleyHardLargeField-v0',
    entry_point='slimevolleyhard:SlimeVolleyHardLargeFieldEnv'
)

if __name__=="__main__":
  """
  Example of how to use Gym env, in single or multiplayer setting

  Humans can override controls:

  left Agent:
  W - Jump
  A - Left
  D - Right

  right Agent:
  Up Arrow, Left Arrow, Right Arrow
  """

  if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

  manualAction = [0, 0, 0, 0, 0] # forward, backward, left, right, jump
  otherManualAction = [0, 0, 0, 0, 0]
  manualMode = False
  otherManualMode = False

  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 1
    if k == key.RIGHT: manualAction[1] = 1
    if k == key.DOWN:  manualAction[2] = 1
    if k == key.UP:    manualAction[3] = 1
    if k == key.SPACE: manualAction[4] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP or k == key.DOWN or k == key.SPACE): manualMode = True

    if k == key.D:     otherManualAction[0] = 1
    if k == key.A:     otherManualAction[1] = 1
    if k == key.W:     otherManualAction[2] = 1
    if k == key.S:     otherManualAction[3] = 1
    if k == key.H:     otherManualAction[4] = 1
    if (k == key.D or k == key.A or k == key.W or k == key.S or k == key.H): otherManualMode = True

  def key_release(k, mod):
    global manualMode, manualAction, otherManualMode, otherManualAction
    if k == key.LEFT:  manualAction[0] = 0
    if k == key.RIGHT: manualAction[1] = 0
    if k == key.DOWN:  manualAction[2] = 0
    if k == key.UP:    manualAction[3] = 0
    if k == key.SPACE: manualAction[4] = 0
    if k == key.D:     otherManualAction[0] = 0
    if k == key.A:     otherManualAction[1] = 0
    if k == key.W:     otherManualAction[2] = 0
    if k == key.S:     otherManualAction[3] = 0
    if k == key.H:     otherManualAction[4] = 0

  policy = BaselinePolicy() # defaults to use RNN Baseline for player

  env = SlimeVolleyHardEnv(render_mode="human")
  env.seed(np.random.randint(0, 10000))
  #env.seed(721)

  if RENDER_MODE:
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

  obs = env.reset()

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

  done = False

  while not done:

    if manualMode: # override with keyboard
      action = manualAction
    else:
      #action = [1, 0, 1, 0, 0]
      action = policy.predict(obs)

    if otherManualMode:
      otherAction = otherManualAction
      obs, reward, done, _, _ = env.step(action, otherAction)
    else:
      #otherAction = [0, 1, 0, 1, 0]
      obs, reward, done, _, _ = env.step(action)#, otherAction)

    if reward > 0 or reward < 0:
      print("reward", reward)
      manualMode = False
      otherManualMode = False

    total_reward += reward

    if RENDER_MODE:
      env.render()
      try:
        env.render2()
      except:
        print("No render2() method in slimevolleyhard!!!")
      sleep(0.01)

    # make the game go slower for human players to be fair to humans.
    if (manualMode or otherManualMode):
      if PIXEL_MODE:
        sleep(0.01)
      else:
        sleep(0.02)

  env.close()
  print("cumulative score", total_reward)
