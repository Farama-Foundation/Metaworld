import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_visual_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set
from .visual_sawyer_sandbox_env import VisualSawyerSandboxEnv
from .heuristics import (
    AlongBackWall,
    Ceiling,
    GreaterThanXValue,
    InFrontOf,
    LessThanXValue,
    LessThanYValue,
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
from .tools.tool import (
    get_position_of,
    set_position_of,
    get_joint_pos_of,
    set_joint_pos_of,
    get_joint_vel_of,
    set_joint_vel_of,
)
from .solver import Solver
from .voxelspace import VoxelSpace


class VisualSawyerButtonPressEnv(VisualSawyerSandboxEnv):

    def __init__(self):
        obj_low = (-1, -1, -1)
        obj_high = (1, 1, 1)
        goal_low = (-1., -1., -1.)
        goal_high = (1., 1., 1.)

        super().__init__()

        self.init_config = {
            'hand_init_pos': self.hand_init_pos,
        }
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        self.curr_path_length += 1
        info = {'success': float(False)}
        return ob, 0, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        '''
        Note: At a later point it may be worth it to replace this with
        self._get_obj_pos_dict
        '''
        return np.zeros(3)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()

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

        world = VoxelSpace((1.75, 0.7, 0.5), 100)
        solver = Solver(world)

        # Place large artifacts along the back of the table
        solver.apply(AlongBackWall(0.95 * world.size[1]), [
            door, nail, hoop, window, coffee_machine, drawer
        ], tries=2)

        # Place certain artifacts on top of one another to save space
        solver.apply(OnTopOf(coffee_machine),   [toaster])
        solver.apply(OnTopOf(drawer),           [shelf])
        solver.apply(OnTopOf(nail),             [outlet])
        solver.apply(OnTopOf(door),             [button])

        # Put the faucet under the basketball hoop
        faucet.specified_pos = hoop.specified_pos + np.array([.0, -.1, .0])
        faucet.specified_pos[2] = faucet.resting_pos_z * world.resolution
        solver.did_manual_set(faucet)

        # Place certain artifacts in front of one another to simplify config
        solver.apply(InFrontOf(window),         [soccer_goal])
        solver.apply(InFrontOf(nail),           [bin_a])
        solver.apply(InFrontOf(bin_a),          [bin_b])

        # The ceiling means that taller objects get placed along the edges
        # of the table. We place them first (note that `shuffle=False` so list
        # order matters) so that the shorter objects don't take up space along
        # the edges until tall objects have had a chance to fill that area.

        def ceiling(i, j):
            # A bowl-shaped ceiling, centered at the Sawyer
            i -= world.mat.shape[0] // 2
            return (0.02 * i * i) + (0.005 * j * j) + 20

        solver.apply(Ceiling(ceiling), [
            thermos, lever, screw_eye_peg, dial
        ], tries=20, shuffle=False)

        # At this point we only have a few objects left to place. They're all
        # tools (not artifacts) which means they can move around freely. As
        # such, they can ignore the immense bounding boxes of the the door and
        # drawer (NOTE: in the future, it may make sense to have separate
        # `bounding` and `clearance` boxes so that freejointed tools can
        # automatically ignore clearance boxes). For now, to ignore the
        # existing bounding boxes, we manually reset their voxels:
        world.fill_tool(door, value=False)
        world.fill_tool(drawer, value=False)

        edge_buffer = (world.size[0] - 1.0) / 2.0
        created_overlaps = solver.apply([
            # Must have this Y constraint to avoid placing inside the door
            # and drawer whose voxels got erased
            LessThanYValue(0.6 * world.size[1]),
            GreaterThanXValue(edge_buffer),
            LessThanXValue(world.size[0] - edge_buffer)
        ], [
            hammer, bin_lid, screw_eye, coffee_mug, puck, basketball, plug
        ], tries=100, shuffle=False)

        if created_overlaps:
            print('Reconfiguring to avoid overlaps')
            self.reset_model()

        for tool in solver.tools:
            tool.specified_pos[0] -= world.size[0] / 2.0
            tool.specified_pos[1] += 0.3

        self._world = world
        self._solver = solver

        # self._make_everything_match_solver()
        self._anneal_free_joints(steps=50)

        return self._get_obs()

