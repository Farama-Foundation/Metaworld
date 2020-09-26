import numpy as np
from gym.spaces import Box
from dm_control.mujoco import Physics

from metaworld.envs.asset_path_utils import full_visual_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from .heuristics import (
    GreaterThanYValue,
    MinimizeOcclusions  ,
    OnTopOf,
)
from .tools import (
    Basketball,
    BasketballHoop,
    BinA,
    BinB,
    BinLid,
    ButtonBox,
    CoffeeMachine,
    CoffeeMug,
    Dial,
    Door,
    Drawer,
    ElectricalPlug,
    ElectricalOutlet,
    FaucetBase,
    HammerBody,
    Lever,
    NailBox,
    Puck,
    ScrewEye,
    ScrewEyePeg,
    Shelf,
    SoccerGoal,
    Thermos,
    ToasterHandle,
    Window,
)
from .tools.tool import get_position_of, set_position_of
from .solver import Solver
from .voxelspace import VoxelSpace


class VisualSawyerSandboxEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.6, 0.02)
        obj_high = (0, 0.6, 0.02)
        goal_low = (-1., 0, 0.)
        goal_high = (1., 1., 1.)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.6, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.4), dtype=np.float32),
        }
        self.goal = np.array([-0.2, 0.8, 0.05], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._obj_names = list(self.model.body_names)

        print(self._obj_names)

        # self.physics = Physics.from_xml_path(self.model_name)
        # # print(self.physics.named.data.xpos)
        # # print(self.physics.named.data.cvel)
        # self.model = self.physics
        # self._set_obj_pos_by_name('Basketball', [0.8,0.4,0.03])

        self.liftThresh = liftThresh
        self.max_path_length = 200

        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_visual_path_for('sawyer_xyz/sawyer_sandbox_small.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        # self._set_obj_vel_by_name('Basketball', np.array([0,0,0,0.5,0,0]))
        # self._set_obj_pos_by_name('BasketballHoop', [0.6, 1.14, 0.03])
        print(self._get_obj_pos_by_name('BasketballHoop'))
        print(self._get_obj_vel_by_name('BasketballHoop'))
        print(self._get_obj_quat_by_name('BasketballHoop'))
        print("~~~~~~~~~~~~~~~~~~~~~~")

        reward, _, reachDist, pickRew, _, placingDist, _, success = self.compute_reward(action, ob)
        self.curr_path_length += 1
        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': placingDist,
            'success': float(success)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        '''
        Note: At a later point it may be worth it to replace this with
        self._get_obj_pos_dict
        '''
        return self.data.site_xpos[self.model.site_name2id('RoundNut-8')]

    def _get_obj_pos_by_name(self, obj_name):
        return self.sim.data.body_xpos[self.model.body_name2id(obj_name)]

    def _set_obj_pos_by_name(self, obj_name, new_pos):
        self.sim.model.body_pos[self.model.body_name2id(obj_name)] = new_pos
        return

    def _get_obj_pos_dict(self):
        return {
            name: self._get_obj_pos_by_name(name) for name in self._obj_names
        }

    def _get_obj_quat_by_name(self, obj_name):
        return self.sim.data.body_xquat[self.model.body_name2id(obj_name)]

    def _set_obj_quat_by_name(self, obj_name, new_pos):
        self.sim.model.body_quat[self.model.body_name2id(obj_name)] = new_pos
        return

    def _get_obj_vel_by_name(self, obj_name):
        return self.sim.data.cvel[self.model.body_name2id(obj_name)]

    def _set_obj_vel_by_name(self, obj_name, new_vel):
        '''
        mjmodel does not have access to set cvel values since they are calculated:
        self.sim.model.cvel[self.model.body_name2id(obj_name)] = new_vel
        '''
        raise NotImplementedError


    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.site_xpos[self.model.site_name2id('RoundNut-8')][2]
        self.heightTarget = self.objHeight + self.liftThresh

        basketball = Basketball()
        hoop = BasketballHoop()
        bin_a = BinA()
        bin_b = BinB()
        bin_lid = BinLid()
        button = ButtonBox()
        coffee_machine = CoffeeMachine()
        coffee_mug = CoffeeMug()
        dial = Dial()
        door = Door()
        drawer = Drawer()
        plug = ElectricalPlug()
        outlet = ElectricalOutlet()
        faucet = FaucetBase()
        hammer = HammerBody()
        lever = Lever()
        nail = NailBox()
        puck = Puck()
        screw_eye = ScrewEye()
        screw_eye_peg = ScrewEyePeg()
        shelf = Shelf()
        soccer_goal = SoccerGoal()
        thermos = Thermos()
        toaster = ToasterHandle()
        window = Window()

        world = VoxelSpace((0.8, 2.75, 0.5), 100)
        solver = Solver(world)

        # Place large artifacts along the back of the table
        solver.apply(GreaterThanYValue(0.76), [
            door, nail, hoop, window, soccer_goal, coffee_machine, drawer
        ])
        # Place certain artifacts on top of one another to save space
        solver.apply(OnTopOf(coffee_machine),   [toaster])
        solver.apply(OnTopOf(drawer),           [shelf])
        solver.apply(OnTopOf(nail),             [outlet])
        solver.apply(OnTopOf(door),             [button])
        # Place the rest of the tools/artifacts such that they don't occlude
        # one another
        solver.apply(MinimizeOcclusions(), [
            thermos, lever, faucet, basketball, hammer, bin_a, bin_b, bin_lid,
            coffee_mug, dial, plug, puck, screw_eye, screw_eye_peg
        ])

        for tool in solver.tools:
            tool.specified_pos[1] += 0.2
            set_position_of(tool, self.sim, self.model)

        self._set_obj_pos_by_name('BasketballHoop', [0.6, 2.14, 0.03])
        self._set_obj_quat_by_name('BasketballHoop', [0.70, 0.70, 0., 0.])

        # tool_pos = get_position_of(hammer, self.sim)
        # for site, corner in zip(
        #         ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        #         hammer.get_bbox_corners()
        # ):
        #     self.sim.model.site_pos[
        #         self.model.site_name2id(f'BBox{site}')
        #     ] = tool_pos + np.array(corner)

        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget
        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        self.placeCompleted = False

    def compute_reward(self, actions, obs):

        graspPos = obs[3:6]
        objPos = self.get_body_com('ScrewEye')

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._target_pos

        reachDist = np.linalg.norm(graspPos - fingerCOM)

        placingDist = np.linalg.norm(objPos[:2] - placingGoal[:2])
        placingDistFinal = np.abs(objPos[-1] - self.objHeight)

        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.04:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance) and reachDist < 0.03:
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)

        def placeCompletionCriteria():
            if abs(objPos[0] - placingGoal[0]) < 0.03 and \
                abs(objPos[1] - placingGoal[1]) < 0.03:
                return True
            else:
                return False

        if placeCompletionCriteria():
            self.placeCompleted = True
        else:
            self.placeCompleted = False

        def orig_pickReward():
            hScale = 100
            if self.placeCompleted or (self.pickCompleted and not(objDropped())):
                return hScale*heightTarget
            elif (reachDist < 0.04) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeRewardMove():
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            if self.placeCompleted:
                c4 = 2000; c5 = 0.003; c6 = 0.0003
                placeRew += 2000*(heightTarget - placingDistFinal) + c4*(np.exp(-(placingDistFinal**2)/c5) + np.exp(-(placingDistFinal**2)/c6))
            placeRew = max(placeRew,0)
            cond = self.placeCompleted or (self.pickCompleted and (reachDist < 0.04) and not(objDropped()))
            if cond:
                return [placeRew, placingDist, placingDistFinal]
            else:
                return [0, placingDist, placingDistFinal]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew , placingDist, placingDistFinal = placeRewardMove()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        success = (abs(objPos[0] - placingGoal[0]) < 0.03 and abs(objPos[1] - placingGoal[1]) < 0.03 and placingDistFinal <= 0.04)
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist, placingDistFinal, success]
