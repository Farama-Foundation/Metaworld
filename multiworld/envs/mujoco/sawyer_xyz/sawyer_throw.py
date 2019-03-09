from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_6dof import SawyerReachXYZ6DOFEnv
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

        self.set_goal(self.sample_goal())
        self._set_goal_marker(self._state_goal)

        env.set_to_goal(
           {'state_desired_goal': env.generate_uncorrected_env_goals(1)['state_desired_goal'][0]}
        )
        # Close gripper for 20 timesteps
        action = np.array([0, 0, 1])
        for _ in range(20):
           obs, _, _, _ = env.step(action)
           env.render()

        return self._get_obs()




if __name__ == '__main__':

    env = SawyerPickAndPlaceEnvYZ(
       hand_low=(-0.1, 0.55, 0.05),
       hand_high=(0.0, 0.65, 0.2),
       action_scale=0.02,
       hide_goal_markers=False,
       num_goals_presampled=5,
       p_obj_in_hand=1,
    )

    while True:

    #    obs = env.reset()
    #    """
    #    Sample a goal (object will be in hand as p_obj_in_hand=1) and try to set
    #    the env state to the goal. I think there's a small chance this can fail
    #    and the object falls out.
    #    """
    #    env.set_to_goal(
    #        {'state_desired_goal': env.generate_uncorrected_env_goals(1)['state_desired_goal'][0]}
    #    )
    #    # Close gripper for 20 timesteps
       action = np.array([0, 0, 1])
       for _ in range(20):
           obs, _, _, _ = env.step(action)
           env.render()