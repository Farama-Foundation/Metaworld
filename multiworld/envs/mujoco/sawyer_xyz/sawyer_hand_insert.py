from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_6dof import SawyerReachXYZ6DOFEnv
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

class SawyerHandInsertEnv(SawyerReachXYZ6DOFEnv):
    def __init__(
        self,
        reward_type='hand_distance',
        norm_order=1,
        indicator_threshold=0.06,
        hide_goal_markers=False,
        frame_skip=5,

        **kwargs

    ):
        SawyerReachXYZ6DOFEnv.__init__(
            self,
            reward_type=reward_type,
            norm_order=norm_order,
            indicator_threshold=indicator_threshold,
            fix_goal=True,
            fixed_goal=(0., 0.4, -0.2),
            hide_goal_markers=hide_goal_markers,
            frame_skip=frame_skip,
            **kwargs)

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_table_with_hole.xml')