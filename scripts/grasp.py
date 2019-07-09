from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import (
   SawyerPickAndPlaceEnv,
   SawyerPickAndPlaceEnvYZ,
)
# from metaworld.core.image_env import ImageEnv
# from metaworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
import numpy as np

env = SawyerPickAndPlaceEnvYZ(
   hand_low=(-0.1, 0.55, 0.05),
   hand_high=(0.0, 0.65, 0.2),
   action_scale=0.02,
   hide_goal_markers=False,
   num_goals_presampled=5,
   p_obj_in_hand=1,
)

while True:
   obs = env.reset()
   """
   Sample a goal (object will be in hand as p_obj_in_hand=1) and try to set
   the env state to the goal. I think there's a small chance this can fail
   and the object falls out.
   """
   env.set_to_goal(
       {'state_desired_goal': env.generate_uncorrected_env_goals(1)['state_desired_goal'][0]}
   )
   # Close gripper for 20 timesteps
   action = np.array([0, 0, 1])
   for _ in range(20):
       obs, _, _, _ = env.step(action)
       env.render()