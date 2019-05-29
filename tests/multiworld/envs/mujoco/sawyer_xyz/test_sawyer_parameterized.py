import pytest
import numpy as np

from tests.helpers import step_env
from tests.helpers import close_env

from multiworld.envs.mujoco.sawyer_xyz import SawyerNutAssembly6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerBinPicking6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerBoxClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerBoxOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerButtonPress6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerButtonPressTopdown6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDialTurn6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDoorEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDoor6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDoorCloseEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDoorHookEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDrawerClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerDrawerOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerHandInsertEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerLaptopClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerLeverPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import MultiSawyerEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPegInsertionSide6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPickAndPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPickAndPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPickAndPlaceWsg6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPushAndReachXYZDoublePuckEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerTwoObjectEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerTwoObject6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerPushAndReachXYEasyEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerReachXYZEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerReach6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerRope6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerShelfPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerStickPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerStickPush6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerSweepEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerSweepIntoGoalEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerThrowEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerWindowClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz import SawyerWindowOpen6DOFEnv

@pytest.mark.parametrize("env_class", [
                                SawyerNutAssembly6DOFEnv,
                                SawyerBinPicking6DOFEnv,
                                SawyerBoxClose6DOFEnv,
                                SawyerBoxOpen6DOFEnv,
                                SawyerButtonPress6DOFEnv,
                                SawyerButtonPressTopdown6DOFEnv,
                                SawyerDialTurn6DOFEnv,
                                SawyerDoor6DOFEnv,
                                SawyerDoorCloseEnv,
                                SawyerDoorEnv,
                                SawyerDoorHookEnv,
                                SawyerDrawerClose6DOFEnv,
                                SawyerDrawerOpen6DOFEnv,
                                SawyerHammer6DOFEnv,
                                SawyerHandInsertEnv,
                                SawyerLaptopClose6DOFEnv,
                                SawyerLeverPull6DOFEnv,
                                SawyerPegInsertionSide6DOFEnv,
                                SawyerPickAndPlaceEnv,
                                SawyerPickAndPlace6DOFEnv,
                                SawyerPickAndPlaceWsg6DOFEnv,
                                SawyerPushAndReachXYEnv,
                                SawyerTwoObjectEnv,
                                SawyerTwoObject6DOFEnv,
                                SawyerPushAndReachXYEasyEnv,
                                SawyerReachXYZEnv,
                                SawyerReach6DOFEnv,
                                SawyerReachPushPickPlace6DOFEnv,
                                SawyerRope6DOFEnv,
                                SawyerShelfPlace6DOFEnv,
                                SawyerStack6DOFEnv,
                                SawyerStickPull6DOFEnv,
                                SawyerStickPush6DOFEnv,
                                SawyerSweepEnv,
                                SawyerSweepIntoGoalEnv,
                                SawyerThrowEnv,
                                SawyerWindowClose6DOFEnv,
                                SawyerWindowOpen6DOFEnv,
                                ])
def test_sawyer(env_class):
    env = env_class()
    step_env(env)
    close_env(env)

def test_sawyer_multiple_objects():
    size = 0.1
    low = np.array([-size, 0.4 - size, 0])
    high = np.array([size, 0.4 + size, 0.1])
    env = MultiSawyerEnv(
        do_render=False,
        finger_sensors=False,
        num_objects=1,
        object_meshes=None,
        # randomize_initial_pos=True,
        fix_z=True,
        fix_gripper=True,
        fix_rotation=True,
        cylinder_radius = 0.03,
        maxlen = 0.03,
        workspace_low = low,
        workspace_high = high,
        hand_low = low,
        hand_high = high,
        init_hand_xyz=(0, 0.4-size, 0.089),
    )
    for i in range(100):
        a = np.random.uniform(-1, 1, 5)
        o, r, _, _ = env.step(a)
        if i % 100 == 0:
            o = env.reset()
        env.render()
    close_env(env)


def test_sawyer_push_and_reach_two_pucks():
    env = SawyerPushAndReachXYZDoublePuckEnv()
    env.set_goal({'state_desired_goal': np.array([1, 1, 1, 1, 1, 1, 1])})
    for i in range(100):
        env.render()
        env.step(env.action_space.sample())
    close_env(env)
