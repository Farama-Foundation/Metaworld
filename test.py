from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import corrected_image_env_goals, setup_image_presampled_goals

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera_slanted_angle
import time
import numpy as np

env = SawyerPickAndPlaceEnvYZ(
    hide_arm=False,
    hide_goal_markers=True,
    oracle_reset_prob=0.0,
)
image_env = ImageEnv(
    env,
    transpose=True,
    normalize=True,
    init_camera=sawyer_pick_and_place_camera_slanted_angle
)
setup_image_presampled_goals(image_env, 10)

image_goals = image_env._presampled_goals['desired_goal']
import cv2
for image_goal in image_goals:
    cv2.imshow('env', image_goal.reshape(3, 84, 84).transpose())
    import time; time.sleep(2)
    cv2.waitKey(1)

import pdb; pdb.set_trace()

env.reset()
# env.render()
for i in range(100000):
    print(i)
    env.reset()
#    env.put_obj_in_hand()
#    delta = env.action_space.sample()#np.array([0.0, -1.0, -1.0])
#    for _ in range(10):
#        delta[2] = 1
#        env.step(delta)
    """hand_pos = env.sample_hand_pos(1)[0]
    env._move_hand(hand_pos, -1)
    env.put_obj_in_hand()
    obs = env._get_obs()
    env.render()
    env.reset()"""
#    env.set_to_goal(env.sample_goal())
    delta = np.array([-.0, -0.2, -1.0])
    for _ in range(100):
        env.step(delta)
        env.render()
    delta = np.array([0.0, .6, 1.0])
    for _ in range(250):
        env.step(delta)
        env.render()

    env.reset()
