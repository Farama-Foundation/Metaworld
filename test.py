from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place \
        import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
import time

env = SawyerPickAndPlaceEnvYZ(hide_arm=False, hide_goal_markers=True)
env.reset()
#env.put_obj_in_hand()
env.render()
time.sleep(1)
while True:
    for _ in range(500):
        delta = env.action_space.sample()
        delta[2] = 1

        env.step(delta)
        time.sleep(.1)
        env.render()
import pdb; pdb.set_trace()
