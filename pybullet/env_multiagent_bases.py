import gymnasium as gym
from gymnasium import spaces
import gymnasium.utils
from gymnasium.utils import seeding
import numpy as np
import pybullet
import os

from typing import TYPE_CHECKING, List, Optional

from pybullet_utils import bullet_client

from pkg_resources import parse_version

try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except:
  pass

class MJCFMultiAgentBaseBulletEnv(gym.Env):
  """
	Base class for Bullet physics simulation loading MJCF (MuJoCo .xml) environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

  metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 2, robots=[]):
    self.scene = None
    self.physicsClientId = -1
    self.ownsPhysicsClient = 0
    self.camera = Camera(self)
    self.render_mode = render_mode
    self.isRender = False
    if render_mode is not None:
      self.isRender = True
    self.robots = robots
    self.seed()
    self._cam_dist = 3
    self._cam_yaw = 0
    self._cam_pitch = -30
    self._render_width = 320
    self._render_height = 240

    self.action_space = robots[0].action_space
    self.observation_space = robots[0].observation_space
    #self.reset()

  def configure(self, args):
    for robot in self.robots:
      robot.args = args

  def seed(self, seed=None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    for robot in self.robots:
      robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
    return [seed]

  def reset(self, seed=None):
    if (self.physicsClientId < 0):
      self.ownsPhysicsClient = True

      if self.isRender:
        self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
      else:
        self._p = bullet_client.BulletClient()
      self._p.resetSimulation()
      self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
      #optionally enable EGL for faster headless rendering
      try:
        if os.environ["PYBULLET_EGL"]:
          con_mode = self._p.getConnectionInfo()['connectionMethod']
          if con_mode==self._p.DIRECT:
            egl = pkgutil.get_loader('eglRenderer')
            if (egl):
              self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
              self._p.loadPlugin("eglRendererPlugin")
      except:
        pass
      self.physicsClientId = self._p._client
      self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    if self.scene is None:
      self.scene = self.create_single_player_scene(self._p)
    if not self.scene.multiplayer and self.ownsPhysicsClient:
      self.scene.episode_restart(self._p)

    for robot in self.robots:
      robot.scene = self.scene

    self.frame = 0
    self.done = 0
    self.reward = 0
    dump = 0
    s = []
    self.potential = []
    for robot in self.robots:
      robot_state = robot.reset(self._p)
      s.append(robot_state)
      self.potential.append(robot.calc_potential())
    s = np.concatenate(s, axis=0)
    return s, {}

  def camera_adjust(self):
    pass

  def render(self, mode='human', close=False):
  
    if self.render_mode is None:
      assert self.spec is not None
      gym.logger.warn(
        "You are calling render method without specifying any render mode. "
        "You can specify the render_mode at initialization, "
        f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
      )
      return
  
    if mode == "human":
      self.isRender = True
    if self.physicsClientId>=0:
      self.camera_adjust()

    if mode != "rgb_array":
      return np.array([])

    base_pos = [0, 0, 0]
    """
    if (hasattr(self, 'robot')):
      for robot in self.robots:
        if (hasattr(robot, 'body_real_xyz')):
          base_pos = robot[0].body_real_xyz # We use the first value
    """
    if (self.physicsClientId>=0):
      view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
      proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(self._render_width) /
                                                     self._render_height,
                                                     nearVal=0.1,
                                                     farVal=100.0)
      (_, _, px, _, _) = self._p.getCameraImage(width=self._render_width,
                                              height=self._render_height,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

      self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
    else:
      px = np.array([[[255,255,255,255]]*self._render_width]*self._render_height, dtype=np.uint8)
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def close(self):
    if (self.ownsPhysicsClient):
      if (self.physicsClientId >= 0):
        self._p.disconnect()
    self.physicsClientId = -1

  def HUD(self, state, a, done):
    pass

  # def step(self, *args, **kwargs):
  # 	if self.isRender:
  # 		base_pos=[0,0,0]
  # 		if (hasattr(self,'robot')):
  # 			if (hasattr(self.robot,'body_xyz')):
  # 				base_pos = self.robot.body_xyz
  # 				# Keep the previous orientation of the camera set by the user.
  # 				#[yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
  # 				self._p.resetDebugVisualizerCamera(3,0,0, base_pos)
  #
  #
  # 	return self.step(*args, **kwargs)
  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed


class Camera:

  def __init__(self, env):
    self.env = env
    try:
      self.nagents = self.env.nagents
    except:
      self.nagents = 1
    pass

  def move_and_look_at(self, i, j, k, x, y, z):
    lookat = [x, y, z]
    camInfo = self.env._p.getDebugVisualizerCamera()
    
    distance = 5.0#camInfo[10]
    pitch = -10.0 * self.nagents#camInfo[9]
    yaw = 0.0#camInfo[8]
    self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
