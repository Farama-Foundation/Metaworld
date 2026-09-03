import numpy as np

from metaworld.envs import (
    SawyerHandInsertEnvV3,
    SawyerPegInsertionSideEnvV3,
    SawyerStickPushEnvV3,
)


class _HandInsertProbe(SawyerHandInsertEnvV3):
    @property
    def touching_main_object(self):
        return False


class _StickPushProbe(SawyerStickPushEnvV3):
    @property
    def touching_main_object(self):
        return False


def _hand_insert_probe():
    env = _HandInsertProbe.__new__(_HandInsertProbe)
    env._set_task_called = True
    env._target_pos = np.array([0.0, 0.0, 0.0])
    env.obj_init_pos = np.array([0.2, 0.0, 0.05])
    env.maxReachDist = 0.2
    env.reward_function_version = "v1"
    env._get_site_pos = lambda _name: np.zeros(3)
    return env


def test_hand_insert_v1_reports_achieved_object_distance():
    env = _hand_insert_probe()
    obs = np.zeros(39)
    obs[3] = 1.0
    obs[4:7] = [0.01, 0.0, 0.0]

    _, info = env.evaluate_state(obs, np.zeros(4))

    assert info["obj_to_target"] == 0.01
    assert info["success"] == 1.0


def test_peg_insert_v1_uses_peg_head_for_success_distance():
    env = SawyerPegInsertionSideEnvV3.__new__(SawyerPegInsertionSideEnvV3)
    env._set_task_called = True
    env._target_pos = np.array([0.0, 0.0, 0.0])
    env.obj_init_pos = np.array([0.2, 0.0, 0.02])
    env.heightTarget = 0.13
    env.objHeight = 0.02
    env.maxPlacingDist = 1.0
    env.init_tcp = np.array([0.0, 0.0, 0.2])
    env.reward_function_version = "v1"
    env._get_site_pos = lambda name: {
        "pegHead": np.array([0.01, 0.0, 0.0]),
        "rightEndEffector": np.zeros(3),
        "leftEndEffector": np.zeros(3),
    }[name]
    obs = np.zeros(39)
    obs[4:7] = env.obj_init_pos

    _, info = env.evaluate_state(obs, np.zeros(4))

    # The body center is 20 cm from the goal, while the peg head is inside
    # the 7 cm success radius.
    assert info["obj_to_target"] == 0.01
    assert info["success"] == 1.0


def test_stick_push_v1_reports_success_after_releasing_stick():
    env = _StickPushProbe.__new__(_StickPushProbe)
    env._set_task_called = True
    env._target_pos = np.array([0.4, 0.6, 0.02])
    env.obj_init_pos = np.array([0.2, 0.6, 0.02])
    env.stick_init_pos = np.array([-0.1, 0.6, 0.02])
    env.heightTarget = 0.06
    env.stickHeight = 0.02
    env.maxPlaceDist = 1.0
    env.maxPushDist = 0.2
    env.reward_function_version = "v1"
    env._get_site_pos = lambda _name: np.zeros(3)
    obs = np.zeros(39)
    obs[4:7] = [-0.1, 0.6, 0.06]
    obs[7:11] = [1.0, 0.0, 0.0, 0.0]
    # The second object block is the pushed container.  The probe deliberately
    # leaves the gripper out of contact to model a released terminal state.
    obs[11:14] = env._target_pos

    _, info = env.evaluate_state(obs, np.zeros(4))

    assert info["obj_to_target"] == 0.0
    assert info["success"] == 1.0
    assert info["grasp_success"] == 0.0
