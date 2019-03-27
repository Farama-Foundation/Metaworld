from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

class SawyerDoorCloseEnv(SawyerDoor6DOFEnv):
    def __init__(
        self,
        goal_low=None,
        goal_high=None,
        action_reward_scale=0,
        indicator_threshold=(.02, .03),
        reset_free=False,
        fixed_hand_z=0.12,
        hand_low=(-0.25, 0.3, .12),
        # hand_high=(0.25, 0.6, .12),
        hand_high=(0.25, 0.8, .12),
        target_pos_scale=1,
        target_angle_scale=1,
        min_angle=-1.5708,
        max_angle=0,
        xml_path='sawyer_xyz/sawyer_door_pull.xml',
        frame_skip=5,
        **sawyer_xyz_kwargs
    ):
        SawyerDoor6DOFEnv.__init__(
        self,
        goal_low=None,
        goal_high=None,
        action_reward_scale=0,
        reward_type='angle_difference',
        indicator_threshold=(.02, .03),
        fix_goal=True,
        fixed_goal=(0, .45, .12, 0),
        reset_free=False,
        fixed_hand_z=0.12,
        hand_low=(-0.25, 0.3, .12),
        # hand_high=(0.25, 0.6, .12),
        hand_high=(0.25, 0.8, .12),
        target_pos_scale=1,
        target_angle_scale=1,
        min_angle=-1.5708,
        max_angle=0,
        xml_path='sawyer_xyz/sawyer_door_pull.xml',
        frame_skip=5,
        **sawyer_xyz_kwargs)

    def reset_model(self):
        if not self.reset_free:
            self._reset_hand()
            self._set_door_pos(-1.5708)
        goal = self.sample_goal()
        self.set_goal(goal)
        self.reset_mocap_welds()
        return self._get_obs()