# custom version adapted for evolutionary strategies
from .scene_stadium import SinglePlayerStadiumScene
from .env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
from robot_locomotors2 import Hopper, Walker2D, HalfCheetah, Ant, Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder, HalfCheetahMulti, AntSwarm
from typing import TYPE_CHECKING, List, Optional
from .env_multiagent_bases import MJCFMultiAgentBaseBulletEnv
import math
import sys

class WalkerBaseBulletEnv(MJCFBaseBulletEnv):

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1, robot=None):
    # print("WalkerBase::__init__ start")

    self.camera_x = 0
    self.walk_target_x = 1e3  # kilometer away
    self.walk_target_y = 0
    self.stateId = -1
    MJCFBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=robot)


  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene

  def reset(self, seed=None):
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r, _ = MJCFBaseBulletEnv.reset(self, seed=seed)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
    self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)

    return r, {}

  def _isDone(self):
    return self._alive < 0

  def move_robot(self, init_x, init_y, init_z):
    "Used by multiplayer stadium to move sideways, to another running lane."
    self.cpp_robot.query_position()
    pose = self.cpp_robot.root_part.pose()
    pose.move_xyz(
        init_x, init_y, init_z
    )  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
    self.cpp_robot.set_pose(pose)

  electricity_cost = -2.0  # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
  stall_torque_cost = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
  foot_collision_cost = -1.0  # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
    ))  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    debugmode = 0
    if (debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [
        self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
    ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, a, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), False, {}


  def camera_adjust(self):
    x, y, z = self.robot.body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)


class HopperBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1):
    self.robot = Hopper()
    WalkerBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)
    print("PyBullet Hopper-v5: reward = progress")

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

 
    self.HUD(state, a, done)

    return state, progress, bool(done), False, {"progress" : progress}


class Walker2DBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1):
    self.robot = Walker2D()
    WalkerBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)
    print("PyBullet Walker2d-v5: reward = progress")
    self.oldz = 0

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    #self.robot.mystep += 1

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0
 
    self.HUD(state, a, done)

    return state, progress, bool(done), False, {"progress" : progress}


class HalfCheetahBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1):
    self.robot = HalfCheetah()
    WalkerBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)
    print("PyBullet Halfcheetah-v5: reward = progress + (njoint_at_limit * -0.1), terminate also when z < 0.3")

  def _isDone(self):
    return False

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
 
    self.HUD(state, a, done)

    if (self._alive < 0):
        return state, progress + joints_at_limit_cost, True, False, {"progress" : progress}
    else:
        return state, progress + joints_at_limit_cost, False, False, {"progress" : progress}        


class AntBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1):
    self.robot = Ant()
    WalkerBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)
    print("PyBullet Ant-v5: reward = progress + 0.01 + (torque_cost * -0.01) + (nJointLimit * -0.1)")

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    stall_cost = -0.01 * float(np.square(a).mean())
    joints_at_limit_cost = float(-0.1 * self.robot.joints_at_limit)
    #jspeedrew  = 1.0  * float(np.abs(self.robot.joint_speeds).mean()) 
 
    self.HUD(state, a, done)

    return state, progress + 0.01 + stall_cost + joints_at_limit_cost, bool(done), False, {"progress" : progress}


class HumanoidBulletEnv(WalkerBaseBulletEnv):

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1, robot=Humanoid()):
    self.robot = robot
    WalkerBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)
    print("PyBullet Humanoid-v5: reward = progress + 1.0 + (jexcess * -10.0) + (nJLimits * -0.1) + (angleoffset * -0.1): init_range [-0.03,0.03]")

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0
    
    joints_at_limit_cost = float(-0.1 * self.robot.joints_at_limit)
    
    jexcess = 0
    for i in range(17):
        if (abs(state[i*2+8]) > 1.0):
            jexcess += (abs(state[i*2+8]) - 1.0)
    joints_excess_cost = jexcess * -10.0
    
    angle_offset_cost = (self.robot.angle_to_target * self.robot.angle_to_target) * -0.1

    #feet_cost = 0
    #if (self.robot.feet_contact[0] == 0.0 and self.robot.feet_contact[1] == 0.0):
        #feet_cost -= 1.0
    #if (self.robot.feet_contact[0] > 0.0 and self.robot.feet_contact[1] > 0.0):
        #feet_cost -= 0.33
    #if (self.robot.mystep < 200):
        #progress = np.clip(progress, -1.0, 1.0)
    #self.robot.mystep += 1
 
    self.HUD(state, a, done)

    #print("progress %.2f excess %.2f limit %.2f angle %.2f" % (progress, joints_excess_cost, joints_at_limit_cost, angle_offset_cost))

    return state, progress + 1.0 + joints_excess_cost + joints_at_limit_cost + angle_offset_cost , bool(done), False, {"progress" : progress}


class HumanoidFlagrunBulletEnv(HumanoidBulletEnv):
  random_yaw = True

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1):
    self.robot = HumanoidFlagrun()
    HumanoidBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)

  def create_single_player_scene(self, bullet_client):
    s = HumanoidBulletEnv.create_single_player_scene(self, bullet_client)
    s.zero_at_running_strip_start_line = False
    return s

  def step(self, a):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(a)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0
    
    joints_at_limit_cost = float(-0.1 * self.robot.joints_at_limit)
    
    jexcess = 0
    for i in range(17):
        if (abs(state[i*2+8]) > 1.0):
            jexcess += (abs(state[i*2+8]) - 1.0)
    joints_excess_cost = jexcess * -10.0
    
    angle_offset_cost = (self.robot.angle_to_target * self.robot.angle_to_target) * -0.1 
 
    self.HUD(state, a, done)

    return state, progress + 0.75 + joints_excess_cost + joints_at_limit_cost + angle_offset_cost , bool(done), False, {"progress" : progress}


class HumanoidFlagrunHarderBulletEnv(HumanoidBulletEnv):
  random_lean = True  # can fall on start

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 1):
    self.robot = HumanoidFlagrunHarder()
    self.electricity_cost /= 4  # don't care that much about electricity, just stand up!
    HumanoidBulletEnv.__init__(self, render_mode=render_mode, nrobots=nrobots, robot=self.robot)

  def create_single_player_scene(self, bullet_client):
    s = HumanoidBulletEnv.create_single_player_scene(self, bullet_client)
    s.zero_at_running_strip_start_line = False
    return s
    
class HalfCheetahMultiBulletEnv(MJCFMultiAgentBaseBulletEnv):

  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  joints_at_limit_cost = -0.1  # discourage stuck joints

  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 2):
    self.nrobots = nrobots
    self.robots = []
    for r in range(self.nrobots):
      self.robots.append(HalfCheetahMulti(robotId=r))
    self.camera_x = 0
    self.stateId = -1
    MJCFMultiAgentBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=self.nrobots, robots=self.robots)
    print("PyBullet HalfcheetahMulti-v0: reward = progress + (njoint_at_limit * -0.1), terminate also when z < 0.3")
    
  def create_single_player_scene(self, bullet_client):
    # It is weird but using multi-player is not trivial
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene
    
  def camera_adjust(self):
    x, y, z = self.robots[0].body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)

  def _isDone(self):
    return False
    
  def reset(self, seed=None):
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r, _ = MJCFMultiAgentBaseBulletEnv.reset(self, seed=seed)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    for robot in self.robots:
      self.parts, self.jdict, self.ordered_joints, self.robot_body = robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
      self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)

    return r, {}

  def step(self, a):
    ac_len = int(len(a) / self.nrobots)
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      ac_idx = 0
      for robot in self.robots:
        robot.apply_action(a[ac_idx:(ac_idx+ac_len)])
        ac_idx += ac_len
      self.scene.global_step()

    state = []
    for robot in self.robots:
      state.append(robot.calc_state())  # also calculates self.joints_at_limit
    state = np.concatenate(state, axis=0)

    self._alive = 1
    for robot in self.robots:
        alive = float(
        robot.alive_bonus(
            state[0] + robot.initial_z,
            robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        if alive < 0:
           self._alive = -1
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    r = 0
    progress = 0.0
    for robot in self.robots:
      potential_old = self.potential[r]
      self.potential[r] = robot.calc_potential()
      progress += float(self.potential[r] - potential_old)
    progress /= float(self.nrobots)

    feet_collision_cost = 0.0
    joints_at_limit_cost = 0.0
    for robot in self.robots:
      for i, f in enumerate(
          robot.feet
      ):  # TODO: Maybe calculating feet contacts could be done within the robot code
        contact_ids = set((x[2], x[4]) for x in f.contact_list())
        #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
        if (self.ground_ids & contact_ids):
          robot.feet_contact[i] = 1.0
        else:
          robot.feet_contact[i] = 0.0

      joints_at_limit_cost += float(self.joints_at_limit_cost * robot.joints_at_limit)
    joints_at_limit_cost /= float(self.nrobots)
 
    self.HUD(state, a, done)

    if (self._alive < 0):
        return state, progress + joints_at_limit_cost, True, False, {"progress" : progress}
    else:
        return state, progress + joints_at_limit_cost, False, False, {"progress" : progress}    

class AntSwarmBulletEnv(MJCFMultiAgentBaseBulletEnv):

  foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
  
  def __init__(self, render_mode: Optional[str] = None, nrobots: Optional[int] = 2):
    self.nrobots = nrobots
    self.robots = []
    for r in range(self.nrobots):
      self.robots.append(AntSwarm(robotId=r))
    self.camera_x = 0
    self.stateId = -1
    MJCFMultiAgentBaseBulletEnv.__init__(self, render_mode=render_mode, nrobots=self.nrobots, robots=self.robots)
    print("PyBullet AntSwarm-v0: reward = progress + 0.01 + (torque_cost * -0.01) + (nJointLimit * -0.1)")

  def create_single_player_scene(self, bullet_client):
    # It is weird but using multi-player is not trivial
    self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                  gravity=9.8,
                                                  timestep=0.0165 / 4,
                                                  frame_skip=4)
    return self.stadium_scene
    
  def camera_adjust(self):
    x, y, z = self.robots[0].body_real_xyz

    self.camera_x = x
    self.camera.move_and_look_at(self.camera_x, y , 1.4, x, y, 1.0)

  def _isDone(self):
    return False
    
  def reset(self, seed=None):
    if (self.stateId >= 0):
      #print("restoreState self.stateId:",self.stateId)
      self._p.restoreState(self.stateId)

    r, _ = MJCFMultiAgentBaseBulletEnv.reset(self, seed=seed)
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    for robot in self.robots:
      self.parts, self.jdict, self.ordered_joints, self.robot_body = robot.addToScene(
        self._p, self.stadium_scene.ground_plane_mjcf)
      self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex],
                            self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
    self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("saving state self.stateId:",self.stateId)

    return r, {}
    
  def calcProgress(self, robot, i):
    potential_old = self.potential[i]
    self.potential[i] = robot.calc_potential()
    progress = float(self.potential[i] - potential_old)
    return progress
    
  def calcStallCost(self, a):
    stall_cost = -0.01 * float(np.square(a).mean())
    return stall_cost
    
  def jointAtLimitCost(self, robot):
    joints_at_limit_cost = float(-0.1 * robot.joints_at_limit)
    return joints_at_limit_cost
    
  def calcDist(self, robot, i):
    dist = 0.0
    epsilon = 0.01 # Can be tuned
    # Target distance
    target_dist = 1.5 # Can be tuned
    # Robot average distance from others
    avg_dist = 0.0
    for r in range(self.nrobots):
      if r != i:
        cdist = np.linalg.norm([self.robots[i].body_xyz[1] - self.robots[r].body_xyz[1], self.robots[i].body_xyz[0] - self.robots[r].body_xyz[0]])
        avg_dist += abs(target_dist - cdist)
    avg_dist /= float(self.nrobots - 1)
    # Default is exponential
    dist = np.exp(-100.0 * avg_dist)
    return dist
     
  def getFitness(self, robot, i, action):
    progress = self.calcProgress(robot, i)
    stall_cost = self.calcStallCost(action)
    joints_at_limit_cost = self.jointAtLimitCost(robot)
    dist_rew = self.calcDist(robot, i)
    rew = progress + dist_rew + stall_cost + joints_at_limit_cost
    return rew
    
  def step(self, a):
    ac_len = int(len(a) / self.nrobots)
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with >
      ac_idx = 0
      for robot in self.robots:
        robot.apply_action(a[ac_idx:(ac_idx+ac_len)])
        ac_idx += ac_len
      self.scene.global_step()

    state = []
    for robot in self.robots:
      state.append(robot.calc_state())  # also calculates self.joints_at_limit
    state = np.concatenate(state, axis=0)

    state_len = len(state)
    rob_state_len = int(state_len / self.nrobots)
    idx = 0
    self._alive = 1
    for robot in self.robots:
        alive = float(
        robot.alive_bonus(
            state[idx] + robot.initial_z,
            robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        if alive < 0:
            self._alive = -1
        idx += rob_state_len
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    reward = 0.0
    ac_idx = 0
    r = 0
    for robot in self.robots:
      for i, f in enumerate(
          robot.feet
      ):  # TODO: Maybe calculating feet contacts could be done within the robot code
        contact_ids = set((x[2], x[4]) for x in f.contact_list())
        #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
        if (self.ground_ids & contact_ids):
          robot.feet_contact[i] = 1.0
        else:
          robot.feet_contact[i] = 0.0
      # Calculate fitness
      reward += self.getFitness(robot, r, a[ac_idx:(ac_idx+ac_len)])
      ac_idx += ac_len
      r += 1
    reward /= float(self.nrobots)
    
    self.HUD(state, a, done)

    if (self._alive < 0):
        return state, reward, True, False, {}
    else:
        return state, reward, False, False, {}

