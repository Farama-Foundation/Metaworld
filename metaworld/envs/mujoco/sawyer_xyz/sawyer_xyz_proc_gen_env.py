import functools
import random
from typing import Union

import numpy as np

from metaworld.envs.utils_asset_path import full_visual_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

from .tools import (
    get_position_of,
    set_position_of,
    get_joint_pos_of,
    set_joint_pos_of,
    get_joint_vel_of,
    set_joint_vel_of,
)
from .solver import Solver
from .voxelspace import VoxelSpace
from .proc_gen import standard as run_standard_proc_gen
from .tasks import TOOLSETS, Task
from .sawyer_xyz_state import SawyerXYZState


class VisualSawyerSandboxEnv(SawyerXYZEnv):

    def __init__(self, task: Task):
        super().__init__(
            self.model_name,
            hand_low=(-0.5, 0.4, .05),
            hand_high=(0.5, 1.0, 0.5),
        )
        self.hand_init_pos = np.array((0, 0.6, 0.4), dtype=np.float32)
        self.max_path_length = 500

        # MARK: Task requirements ----------------------------------------------
        self._task = task
        self._state: Union[SawyerXYZState, None] = None
        self._random_reset_space = self._task.random_reset_space

        # MARK: Procedural generation requirements -----------------------------
        self._all_tool_names = list(self._body_names)
        self._world = None
        self._solver = None

        # toolset_required should contain the names of tools that are necessary
        # to solve the current task
        self._toolset_required = TOOLSETS[type(self._task).__name__]
        # toolset_extra should contain the named of additional tools that will
        # be placed on the table to make task identification non-trivial
        self._toolset_extra = set()

    @property
    def model_name(self):
        return full_visual_path_for('sawyer_xyz/sawyer_sandbox_empty_table.xml')

    @property
    def _target_site_config(self):
        # TODO this method should probably be fleshed out.
        # Could have a mapping (like TOOLSETS in library.py) that goes from
        # task name to associated sites. Then just return those names
        return []

    def randomize_extra_toolset(self, n_tasks, seed=None):
        extra_toolsets = {**TOOLSETS}
        del extra_toolsets[type(self._task).__name__]

        if seed is not None:
            random.seed(seed)

        # This will select n - 1 additional tasks from which to pull
        # objects. We enforce no guarantee that those additional tasks will
        # have unique objects. For example, if 4 additional tasks are selected
        # and end up being [window open/close and door open/close], there will
        # only be 2 additional objects in the scene despite there being 4 tasks
        selected = random.choices(list(extra_toolsets.values()), k=n_tasks - 1)
        self._toolset_extra = functools.reduce(
            lambda a, b: set(a).union(b), selected
        )

    def _reset_required_tools(self, world, solver):
        """
        Allows implementations to customize how the required tools are placed,
        as those placements may drastically impact task solvability, making
        automatic placement undesirable
        """
        random_reset_vec = self._get_state_rand_vec() if self.random_init else \
            (self._task.random_reset_space.low +
             self._task.random_reset_space.high) / 2.0

        self._task.reset_required_tools(
            world,
            solver,
            random_reset_vec
        )

    def reset_model(self, solve_required_tools=False):
        self._reset_hand()
        self._state = SawyerXYZState()  # Resets timestep counter

        # Define a new world into which solver can place tools
        world = VoxelSpace((1.75, 0.6, 0.5), 100)
        solver = Solver(world)

        # Figure out scope of the problem -- do we need to place both extra
        # *and* required tools, or just the extra ones?
        toolset = self._toolset_extra
        if solve_required_tools:
            toolset = toolset.union(self._toolset_required)
        else:
            self._reset_required_tools(world, solver)

        # Run the standard (and currently only) set of procedural generation
        # steps/rules. Could add more in the future
        free_of_overlaps = run_standard_proc_gen(solver, world, toolset)
        if not free_of_overlaps:
            print('Reconfiguring to avoid overlaps')
            self.reset_model()

        # Transform back to Mujoco-space (X=0 is now center, Y shifted up a bit)
        for tool in solver.tools:
            tool.specified_pos[0] -= world.size[0] / 2.0
            tool.specified_pos[1] += 0.3
        # Store our hard work for future reference
        self._world = world
        self._solver = solver
        # Tell Mujoco to play nice and match the positions we've generated
        self._make_everything_match_solver()
        self._anneal_free_joints(steps=50)

    def show_bbox_for(self, tool):
        """
        Places sites at the 6 corners of `tool`'s bounding box. Useful when
        debugging procedural generation.
        """
        tool_pos = get_position_of(tool, self.sim)
        for site, corner in zip(
                ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                tool.get_bbox_corners()
        ):
            self.sim.model.site_pos[
                self.model.site_name2id(f'BBox{site}')
            ] = tool_pos + np.array(corner)

    def _make_everything_match_solver(self):
        """
        Looks at all tools associated with `self._solver`, including those that
        were placed manually. For each enabled tool which is actually present
        in the XML, this will ensure Mujoco position matches locally-defined
        position. If tool name includes 'Joint', joint position will be set
        rather than absolute object position.
        """
        for tool in self._solver.tools:
            if not tool.enabled:
                continue
            if tool.name not in self._all_tool_names:
                print(f'Skipping {tool.name} placement. You sure it\'s in XML?')
                continue
            if tool.name + 'Joint' in self.model.joint_names:
                qpos_old = get_joint_pos_of(tool, self.sim)
                qpos_new = qpos_old.copy()

                print(f'{tool.name} is associated with Mujoco DoF {self.model.body_dofadr[self.model.body_name2id(tool.name)]}. Refer to these messages if something goes unstable')

                qpos_new[:3] = tool.specified_pos
                qpos_new[3:] = np.round(qpos_old[3:], decimals=1)

                set_joint_pos_of(tool, self.sim, qpos_new)
                set_joint_vel_of(tool, self.sim, np.zeros(6))
            else:
                set_position_of(tool, self.sim, self.model)

    def _anneal_free_joints(self, steps=10):
        """
        Exponentially reduces joint velocities over some number of `steps`.
        Should be called immediately after `self._make_everything_match_solver`
        """
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

    def evaluate_state(self, obs, action):
        # TODO this function doesn't (and shouldn't) use the obs arg passed to
        # it from superclass. Remove this arg (and associated superclass logic)

        self._state.populate(
            action,
            self._task.get_pos_objects(self.sim),
            self._task.get_quat_objects(self.sim),
            self.sim
        )

        return self._task.compute_reward(self._state)
