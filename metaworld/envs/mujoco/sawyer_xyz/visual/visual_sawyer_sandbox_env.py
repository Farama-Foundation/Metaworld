import functools
import random

import numpy as np

from metaworld.envs.asset_path_utils import full_visual_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
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
from .tasks.library import TOOLSETS


class VisualSawyerSandboxEnv(SawyerXYZEnv):

    def __init__(self):
        super().__init__(
            self.model_name,
            hand_low=(-0.5, 0.4, .05),
            hand_high=(0.5, 1.0, 0.5),
        )
        self.hand_init_pos = np.array((0, 0.6, 0.4), dtype=np.float32)
        self.max_path_length = 500

        self._all_tool_names = list(self._body_names)
        self._world = None
        self._solver = None

        # toolset_required should contain the names of tools that are necessary
        # to solve the current task
        self._toolset_required = set()
        # toolset_extra should contain the named of additional tools that will
        # be placed on the table to make task identification non-trivial
        self._toolset_extra = set()

    @property
    def model_name(self):
        return full_visual_path_for('sawyer_xyz/sawyer_sandbox_empty_table.xml')

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return None

    def randomize_extra_toolset(self, n_tasks, seed=None):
        extra_toolsets = {**TOOLSETS}
        del extra_toolsets[type(self).__name__]

        if seed is not None:
            random.seed(seed)

        # This will select n - 1 additional tasks from which to pull
        # objects. We enforce no guarantee that those additional tasks will
        # have unique objects. For example, if 4 additional tasks are selected
        # and end up being [window open/close and door open/close], there will
        # only be 2 additional objects in the scene despite there being 4 tasks
        selected = random.choices(list(extra_toolsets.values()), k=n_tasks - 1)
        self._toolset_extra = functools.reduce(lambda a,b: a.union(b), selected)

    def _reset_required_tools(self, world, solver):
        """
        Allows implementations to customize how the required tools are placed,
        as those placements may drastically impact task solvability, and
        automatic placement may be undesirable
        """
        pass

    def reset_model(self, solve_required_tools=False):
        self._reset_hand()

        world = VoxelSpace((1.75, 0.7, 0.5), 100)
        solver = Solver(world)

        toolset = self._toolset_extra
        if solve_required_tools:
            toolset = toolset.union(self._toolset_required)
        else:
            self._reset_required_tools(world, solver)

        def construct(cls):
            return cls(enabled=cls.__name__ in toolset)

        basketball = construct(Basketball)
        hoop = construct(BasketballHoop)
        bin_a = construct(BinA)
        bin_b = construct(BinB)
        bin_lid = construct(BinLid)
        button = construct(ButtonBox)
        coffee_machine = construct(CoffeeMachine)
        coffee_mug = construct(CoffeeMug)
        dial = construct(Dial)
        door = construct(Door)
        drawer = construct(Drawer)
        plug = construct(ElectricalPlug)
        outlet = construct(ElectricalOutlet)
        faucet = construct(FaucetBase)
        hammer = construct(HammerBody)
        lever = construct(Lever)
        nail = construct(NailBox)
        puck = construct(Puck)
        screw_eye = construct(ScrewEye)
        screw_eye_peg = construct(ScrewEyePeg)
        shelf = construct(Shelf)
        soccer_goal = construct(SoccerGoal)
        thermos = construct(Thermos)
        toaster = construct(ToasterHandle)
        window = construct(Window)

        # Place large artifacts along the back of the table
        back_lineup = [door, nail, hoop, window, coffee_machine, drawer]
        # Place certain artifacts on top of one another to save space
        stacked = (
            (coffee_machine,    toaster),
            (drawer,            shelf),
            (nail,              outlet),
            (door,              button),
        )
        # If the lowermost item in the stack is disabled, place the upper item
        # along the back of the table instead
        back_lineup += [upper for lower, upper in stacked if not lower.enabled]

        solver.apply(AlongBackWall(0.95 * world.size[1]), back_lineup, tries=2)
        for lower, upper in stacked:
            if lower.enabled:
                solver.apply(OnTopOf(lower), [upper])

        # Put the faucet under the basketball hoop
        if faucet.enabled and hoop.enabled:
            faucet.specified_pos = hoop.specified_pos + np.array([.0, -.1, .0])
            faucet.specified_pos[2] = faucet.resting_pos_z * world.resolution
            solver.did_manual_set(faucet)

        # The ceiling means that taller objects get placed along the edges
        # of the table. We place them first (note that `shuffle=False` so list
        # order matters) so that the shorter objects don't take up space along
        # the edges until tall objects have had a chance to fill that area.
        def ceiling(i, j):
            # A bowl-shaped ceiling, centered at the Sawyer
            i -= world.mat.shape[0] // 2
            return (0.02 * i * i) + (0.005 * j * j) + 20

        # Place certain objects under a ceiling
        under_ceiling = [thermos, lever, screw_eye_peg, dial]
        # Place certain artifacts in front of one another to simplify config
        layered = (
            (window, soccer_goal),
            (nail, bin_a),
            (bin_a, bin_b),
        )
        # If rearmost item is disabled, place other item under ceiling instead
        under_ceiling += [fore for aft, fore in layered if not aft.enabled]
        # If faucet wasn't placed earlier, put it under the ceiling as well
        if faucet.enabled and not hoop.enabled:
            under_ceiling.append(faucet)

        solver.apply(Ceiling(ceiling), under_ceiling, tries=20, shuffle=False)
        for aft, fore in layered:
            if aft.enabled:
                solver.apply(InFrontOf(aft), [fore])

        # At this point we only have a few objects left to place. They're all
        # tools (not artifacts) which means they can move around freely. As
        # such, they can ignore the immense bounding boxes of the the door and
        # drawer (NOTE: in the future, it may make sense to have separate
        # `bounding` and `clearance` boxes so that freejointed tools can
        # automatically ignore clearance boxes). For now, to ignore the
        # existing bounding boxes, we manually reset their voxels:
        if door.enabled:
            world.fill_tool(door, value=False)
        if drawer.enabled:
            world.fill_tool(drawer, value=False)

        edge_buffer = (world.size[0] - 1.0) / 2.0
        free_of_overlaps = solver.apply([
            # Must have this Y constraint to avoid placing inside the door
            # and drawer whose voxels got erased
            LessThanYValue(0.6 * world.size[1]),
            GreaterThanXValue(edge_buffer),
            LessThanXValue(world.size[0] - edge_buffer)
        ], [
            hammer, bin_lid, screw_eye, coffee_mug, puck, basketball, plug
        ], tries=100, shuffle=False)

        if not free_of_overlaps:
            print('Reconfiguring to avoid overlaps')
            self.reset_model()

        for tool in solver.tools:
            tool.specified_pos[0] -= world.size[0] / 2.0
            tool.specified_pos[1] += 0.3

        self._world = world
        self._solver = solver

        self._make_everything_match_solver()
        self._anneal_free_joints(steps=50)

        return self._get_obs()

    def show_bbox_for(self, tool):
        tool_pos = get_position_of(tool, self.sim)
        for site, corner in zip(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                tool.get_bbox_corners()
        ):
            self.sim.model.site_pos[
                self.model.site_name2id(f'BBox{site}')
            ] = tool_pos + np.array(corner)

    def _make_everything_match_solver(self):
        for tool in self._solver.tools:
            if not tool.enabled:
                continue
            if tool.name not in self._all_tool_names:
                print(f'Skipping {tool.name} placement. You sure it\'s in XML?')
                continue
            if tool.name + 'Joint' in self.model.joint_names:
                qpos_old = get_joint_pos_of(tool, self.sim)
                qpos_new = qpos_old.copy()

                print(tool.name)
                print(self.model.body_dofadr[self.model.body_name2id(tool.name)])

                qpos_new[:3] = tool.specified_pos
                qpos_new[3:] = np.round(qpos_old[3:], decimals=1)

                set_joint_pos_of(tool, self.sim, qpos_new)
                set_joint_vel_of(tool, self.sim, np.zeros(6))
            else:
                set_position_of(tool, self.sim, self.model)

    def _anneal_free_joints(self, steps=10):
        for step in range(steps):
            self.sim.step()

            for tool in self._solver.tools:
                if tool.name + 'Joint' in self.model.joint_names:
                    qvel_old = get_joint_vel_of(tool, self.sim)

                    qvel_new = qvel_old.copy()
                    qvel_new[[0, 1, 3, 4, 5]] /= 2 ** step
                    qvel_new[2] /= 1.05 ** step

                    set_joint_vel_of(tool, self.sim, qvel_new)

    @property
    def _body_names(self):
        return self.model.body_names

    @property
    def _joint_names(self):
        return self.model.joint_names

    @property
    def _joint_pos(self):
        return self.model.jnt_pos

    @property
    def _joint_type(self):
        return self.model.jnt_type
