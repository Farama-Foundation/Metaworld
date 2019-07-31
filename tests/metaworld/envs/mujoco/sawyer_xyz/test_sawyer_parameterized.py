import pytest
import numpy as np

from tests.helpers import step_env
from tests.helpers import close_env

from metaworld.envs.mujoco.sawyer_xyz import SawyerNutAssembly6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBinPicking6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBoxClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBoxOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerButtonPress6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerButtonPressTopdown6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerDialTurn6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerDoor6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerDoorClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerDoorHookEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerDrawerClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerDrawerOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerHammer6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerHandInsert6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerLaptopClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerLeverPull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import MultiSawyerEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerPegInsertionSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerTwoObjectEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerTwoObject6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerPushAndReachXYEasyEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlace6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerRope6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerShelfPlace6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerStack6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerStickPull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerStickPush6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerSweep6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerSweepIntoGoal6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerWindowClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerWindowOpen6DOFEnv


from metaworld.envs.mujoco.sawyer_xyz.env_lists import HARD_MODE_LIST


# @pytest.mark.parametrize("env_class", [
#                                 SawyerNutAssembly6DOFEnv,
#                                 SawyerBinPicking6DOFEnv,
#                                 SawyerBoxClose6DOFEnv,
#                                 # This is failing due to some recent changes
#                                 # TODO: consult kevin for box height
#                                 # SawyerBoxOpen6DOFEnv,
#                                 SawyerButtonPress6DOFEnv,
#                                 SawyerButtonPressTopdown6DOFEnv,
#                                 SawyerDialTurn6DOFEnv,
#                                 SawyerDoor6DOFEnv,
#                                 SawyerDoorClose6DOFEnv,
#                                 SawyerDoorHookEnv,
#                                 SawyerDrawerClose6DOFEnv,
#                                 SawyerDrawerOpen6DOFEnv,
#                                 SawyerHammer6DOFEnv,
#                                 SawyerHandInsert6DOFEnv,
#                                 SawyerLaptopClose6DOFEnv,
#                                 SawyerLeverPull6DOFEnv,
#                                 SawyerPegInsertionSide6DOFEnv,
#                                 SawyerTwoObjectEnv,
#                                 SawyerTwoObject6DOFEnv,
#                                 SawyerReachPushPickPlace6DOFEnv,
#                                 # This is failing due to mjcf file error
#                                 # TODO: fix this and add it back to the test
#                                 # SawyerRope6DOFEnv,
#                                 SawyerShelfPlace6DOFEnv,
#                                 SawyerStack6DOFEnv,
#                                 SawyerStickPull6DOFEnv,
#                                 SawyerStickPush6DOFEnv,
#                                 SawyerSweep6DOFEnv,
#                                 SawyerSweepIntoGoal6DOFEnv,
#                                 SawyerWindowClose6DOFEnv,
#                                 SawyerWindowOpen6DOFEnv,
#                                 ])

@pytest.mark.parametrize('env_cls', HARD_MODE_LIST)
def test_sawyer(env_cls):
    env = env_cls()
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
        # env.render()
    close_env(env)

