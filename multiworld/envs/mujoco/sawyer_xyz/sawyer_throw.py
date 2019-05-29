from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_6dof import SawyerReach6DOFEnv
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import (
   SawyerPickAndPlaceEnv,
   SawyerPickAndPlaceEnvYZ,
)
import numpy as np

class SawyerThrowEnv(SawyerPickAndPlaceEnv):
    def __init__(
        self,
        **kwargs

    ):
        SawyerPickAndPlaceEnv.__init__(
            self,

            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
            action_scale=0.02,
            hide_goal_markers=False,
            num_goals_presampled=5,
            p_obj_in_hand=1
    )

    def reset_model(self):
        self._reset_hand()
        if self.reset_free:
            self._set_obj_xyz(self.last_obj_pos)
            self.set_goal(self.sample_goal())
            self._set_goal_marker(self._state_goal)
            return self._get_obs()

        if self.random_init:
            goal = np.random.uniform(
                self.hand_and_obj_space.low[3:],
                self.hand_and_obj_space.high[3:],
                size=(1, self.hand_and_obj_space.low.size - 3),
            )
            goal[:, 2] = self.obj_init_z
            self._set_obj_xyz(goal)
        else:
            obj_idx = np.random.choice(len(self.obj_init_positions))
            self._set_obj_xyz(self.obj_init_positions[obj_idx])

        if self.oracle_reset_prob > np.random.random():
            self.set_to_goal(self.sample_goal())

        goal = self.sample_goal()
        # print(goal)
        self.set_goal(goal)
        self._set_goal_marker(self._state_goal)

        self.set_to_goal(
           {'state_desired_goal': self.generate_uncorrected_env_goals(1)['state_desired_goal'][0]}
        )
        # Close gripper for 20 timesteps
        # action = np.array([0, 0, 0, 1])
        # for _ in range(20):
        #    obs, _, _, _ = self.step(action)
           # env.render()

        return self._get_obs()

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        obj_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        obj_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        obj_distances = np.linalg.norm(obj_goals - obj_pos, axis=1)
        hand_and_obj_distances = hand_distances + obj_distances
        touch_distances = np.linalg.norm(hand_pos - obj_pos, axis=1)
        touch_and_obj_distances = touch_distances + obj_distances

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'obj_distance':
            r = -obj_distances
        elif self.reward_type == 'obj_success':
            r = -(obj_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_obj_distance':
            r = -hand_and_obj_distances
        elif self.reward_type == 'touch_and_obj_distance':
            r = -touch_and_obj_distances
        elif self.reward_type == 'hand_and_obj_success':
            r = -(
                hand_and_obj_distances < self.indicator_threshold
            ).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r
