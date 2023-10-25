import re
from collections import OrderedDict

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import *
from metaworld.envs.mujoco.jaco.v2 import *
from metaworld.envs.mujoco.fetch.v2 import *

SAWYER_ENVIRONMENTS = OrderedDict(
    (
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("basketball-v2", SawyerBasketballEnvV2),
        ("bin-picking-v2", SawyerBinPickingEnvV2),
        ("box-close-v2", SawyerBoxCloseEnvV2),
        ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("button-press-v2", SawyerButtonPressEnvV2),
        ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
        ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
        ("coffee-pull-v2", SawyerCoffeePullEnvV2),
        ("coffee-push-v2", SawyerCoffeePushEnvV2),
        ("dial-turn-v2", SawyerDialTurnEnvV2),
        ("disassemble-v2", SawyerNutDisassembleEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
        ("door-lock-v2", SawyerDoorLockEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("door-unlock-v2", SawyerDoorUnlockEnvV2),
        ("hand-insert-v2", SawyerHandInsertEnvV2),
        ("drawer-close-v2", SawyerDrawerCloseEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("faucet-open-v2", SawyerFaucetOpenEnvV2),
        ("faucet-close-v2", SawyerFaucetCloseEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
        ("handle-press-v2", SawyerHandlePressEnvV2),
        ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
        ("handle-pull-v2", SawyerHandlePullEnvV2),
        ("lever-pull-v2", SawyerLeverPullEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
        ("push-back-v2", SawyerPushBackEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("plate-slide-v2", SawyerPlateSlideEnvV2),
        ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
        ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
        ("soccer-v2", SawyerSoccerEnvV2),
        ("stick-push-v2", SawyerStickPushEnvV2),
        ("stick-pull-v2", SawyerStickPullEnvV2),
        ("push-wall-v2", SawyerPushWallEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("reach-wall-v2", SawyerReachWallEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("reach-goal-as-obj-v2", SawyerReachGoalAsObjEnvV2),
        ("shelf-place-v2", SawyerShelfPlaceEnvV2),
        ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
        ("sweep-v2", SawyerSweepEnvV2),
        ("window-open-v2", SawyerWindowOpenEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
    )
)

JACO_ENVIRONMENTS = OrderedDict(
    (
        ("assembly-v2", JacoNutAssemblyEnvV2),
        ("basketball-v2", JacoBasketballEnvV2),
        ("bin-picking-v2", JacoBinPickingEnvV2),
        ("box-close-v2", JacoBoxCloseEnvV2),
        ("button-press-topdown-v2", JacoButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", JacoButtonPressTopdownWallEnvV2),
        ("button-press-v2", JacoButtonPressEnvV2),
        ("button-press-wall-v2", JacoButtonPressWallEnvV2),
        ("coffee-button-v2", JacoCoffeeButtonEnvV2),
        ("coffee-pull-v2", JacoCoffeePullEnvV2),
        ("coffee-push-v2", JacoCoffeePushEnvV2),
        ("dial-turn-v2", JacoDialTurnEnvV2),
        ("disassemble-v2", JacoNutDisassembleEnvV2),
        ("door-close-v2", JacoDoorCloseEnvV2),
        ("door-lock-v2", JacoDoorLockEnvV2),
        ("door-open-v2", JacoDoorEnvV2),
        ("door-unlock-v2", JacoDoorUnlockEnvV2),
        ("hand-insert-v2", JacoHandInsertEnvV2),
        ("drawer-close-v2", JacoDrawerCloseEnvV2),
        ("drawer-open-v2", JacoDrawerOpenEnvV2),
        ("faucet-open-v2", JacoFaucetOpenEnvV2),
        ("faucet-close-v2", JacoFaucetCloseEnvV2),
        ("hammer-v2", JacoHammerEnvV2),
        ("handle-press-side-v2", JacoHandlePressSideEnvV2),
        ("handle-press-v2", JacoHandlePressEnvV2),
        ("handle-pull-side-v2", JacoHandlePullSideEnvV2),
        ("handle-pull-v2", JacoHandlePullEnvV2),
        ("lever-pull-v2", JacoLeverPullEnvV2),
        ("peg-insert-side-v2", JacoPegInsertionSideEnvV2),
        ("pick-place-wall-v2", JacoPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", JacoPickOutOfHoleEnvV2),
        ("push-back-v2", JacoPushBackEnvV2),
        ("push-v2", JacoPushEnvV2),
        ("pick-place-v2", JacoPickPlaceEnvV2),
        ("plate-slide-v2", JacoPlateSlideEnvV2),
        ("plate-slide-side-v2", JacoPlateSlideSideEnvV2),
        ("plate-slide-back-v2", JacoPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", JacoPlateSlideBackSideEnvV2),
        ("peg-insert-side-v2", JacoPegInsertionSideEnvV2),
        ("peg-unplug-side-v2", JacoPegUnplugSideEnvV2),
        ("soccer-v2", JacoSoccerEnvV2),
        ("stick-push-v2", JacoStickPushEnvV2),
        ("stick-pull-v2", JacoStickPullEnvV2),
        ("push-wall-v2", JacoPushWallEnvV2),
        ("push-v2", JacoPushEnvV2),
        ("reach-wall-v2", JacoReachWallEnvV2),
        ("reach-v2", JacoReachEnvV2),
        ("reach-goal-as-obj-v2", JacoReachGoalAsObjEnvV2),
        ("shelf-place-v2", JacoShelfPlaceEnvV2),
        ("sweep-into-v2", JacoSweepIntoGoalEnvV2),
        ("sweep-v2", JacoSweepEnvV2),
        ("window-open-v2", JacoWindowOpenEnvV2),
        ("window-close-v2", JacoWindowCloseEnvV2),
    )
)

FETCH_ENVIRONMENTS = OrderedDict(
    (
        ("assembly-v2", FetchNutAssemblyEnvV2),
        ("basketball-v2", FetchBasketballEnvV2),
        ("bin-picking-v2", FetchBinPickingEnvV2),
        ("box-close-v2", FetchBoxCloseEnvV2),
        ("button-press-topdown-v2", FetchButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", FetchButtonPressTopdownWallEnvV2),
        ("button-press-v2", FetchButtonPressEnvV2),
        ("button-press-wall-v2", FetchButtonPressWallEnvV2),
        ("coffee-button-v2", FetchCoffeeButtonEnvV2),
        ("coffee-pull-v2", FetchCoffeePullEnvV2),
        ("coffee-push-v2", FetchCoffeePushEnvV2),
        ("dial-turn-v2", FetchDialTurnEnvV2),
        ("disassemble-v2", FetchNutDisassembleEnvV2),
        ("door-close-v2", FetchDoorCloseEnvV2),
        ("door-lock-v2", FetchDoorLockEnvV2),
        ("door-open-v2", FetchDoorEnvV2),
        ("door-unlock-v2", FetchDoorUnlockEnvV2),
        ("hand-insert-v2", FetchHandInsertEnvV2),
        ("drawer-close-v2", FetchDrawerCloseEnvV2),
        ("drawer-open-v2", FetchDrawerOpenEnvV2),
        ("faucet-open-v2", FetchFaucetOpenEnvV2),
        ("faucet-close-v2", FetchFaucetCloseEnvV2),
        ("hammer-v2", FetchHammerEnvV2),
        ("handle-press-side-v2", FetchHandlePressSideEnvV2),
        ("handle-press-v2", FetchHandlePressEnvV2),
        ("handle-pull-side-v2", FetchHandlePullSideEnvV2),
        ("handle-pull-v2", FetchHandlePullEnvV2),
        ("lever-pull-v2", FetchLeverPullEnvV2),
        ("peg-insert-side-v2", FetchPegInsertionSideEnvV2),
        ("pick-place-wall-v2", FetchPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", FetchPickOutOfHoleEnvV2),
        ("push-back-v2", FetchPushBackEnvV2),
        ("push-v2", FetchPushEnvV2),
        ("pick-place-v2", FetchPickPlaceEnvV2),
        ("plate-slide-v2", FetchPlateSlideEnvV2),
        ("plate-slide-side-v2", FetchPlateSlideSideEnvV2),
        ("plate-slide-back-v2", FetchPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", FetchPlateSlideBackSideEnvV2),
        ("peg-insert-side-v2", FetchPegInsertionSideEnvV2),
        ("peg-unplug-side-v2", FetchPegUnplugSideEnvV2),
        ("soccer-v2", FetchSoccerEnvV2),
        ("stick-push-v2", FetchStickPushEnvV2),
        ("stick-pull-v2", FetchStickPullEnvV2),
        ("push-wall-v2", FetchPushWallEnvV2),
        ("push-v2", FetchPushEnvV2),
        ("reach-wall-v2", FetchReachWallEnvV2),
        ("reach-v2", FetchReachEnvV2),
        ("reach-goal-as-obj-v2", FetchReachGoalAsObjEnvV2),
        ("shelf-place-v2", FetchShelfPlaceEnvV2),
        ("sweep-into-v2", FetchSweepIntoGoalEnvV2),
        ("sweep-v2", FetchSweepEnvV2),
        ("window-open-v2", FetchWindowOpenEnvV2),
        ("window-close-v2", FetchWindowCloseEnvV2),
    )
)

_NUM_METAWORLD_ENVS = len(SAWYER_ENVIRONMENTS)
# V2 DICTS

MT10_V2 = OrderedDict(
    (
        ("reach-v2", SawyerReachEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("drawer-close-v2", SawyerDrawerCloseEnvV2),
        ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("window-open-v2", SawyerWindowOpenEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
    ),
)


MT10_V2_ARGS_KWARGS = {
    key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT10_V2.items()
}

ML10_V2 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("reach-v2", SawyerReachEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    ("door-open-v2", SawyerDoorEnvV2),
                    ("drawer-close-v2", SawyerDrawerCloseEnvV2),
                    ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("window-open-v2", SawyerWindowOpenEnvV2),
                    ("sweep-v2", SawyerSweepEnvV2),
                    ("basketball-v2", SawyerBasketballEnvV2),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("drawer-open-v2", SawyerDrawerOpenEnvV2),
                    ("door-close-v2", SawyerDoorCloseEnvV2),
                    ("shelf-place-v2", SawyerShelfPlaceEnvV2),
                    ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
                    (
                        "lever-pull-v2",
                        SawyerLeverPullEnvV2,
                    ),
                )
            ),
        ),
    )
)


ml10_train_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML10_V2["train"].items()
}

ml10_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML10_V2["test"].items()
}

ML10_ARGS_KWARGS = dict(
    train=ml10_train_args_kwargs,
    test=ml10_test_args_kwargs,
)

ML1_V2 = OrderedDict((("train", SAWYER_ENVIRONMENTS), ("test", SAWYER_ENVIRONMENTS)))

ML1_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML1_V2["train"].items()
}
MT50_V2 = OrderedDict(
    (
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("basketball-v2", SawyerBasketballEnvV2),
        ("bin-picking-v2", SawyerBinPickingEnvV2),
        ("box-close-v2", SawyerBoxCloseEnvV2),
        ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
        ("button-press-v2", SawyerButtonPressEnvV2),
        ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
        ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
        ("coffee-pull-v2", SawyerCoffeePullEnvV2),
        ("coffee-push-v2", SawyerCoffeePushEnvV2),
        ("dial-turn-v2", SawyerDialTurnEnvV2),
        ("disassemble-v2", SawyerNutDisassembleEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
        ("door-lock-v2", SawyerDoorLockEnvV2),
        ("door-open-v2", SawyerDoorEnvV2),
        ("door-unlock-v2", SawyerDoorUnlockEnvV2),
        ("hand-insert-v2", SawyerHandInsertEnvV2),
        ("drawer-close-v2", SawyerDrawerCloseEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("faucet-open-v2", SawyerFaucetOpenEnvV2),
        ("faucet-close-v2", SawyerFaucetCloseEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
        ("handle-press-v2", SawyerHandlePressEnvV2),
        ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
        ("handle-pull-v2", SawyerHandlePullEnvV2),
        ("lever-pull-v2", SawyerLeverPullEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("push-back-v2", SawyerPushBackEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("plate-slide-v2", SawyerPlateSlideEnvV2),
        ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
        ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
        ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
        ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
        ("soccer-v2", SawyerSoccerEnvV2),
        ("stick-push-v2", SawyerStickPushEnvV2),
        ("stick-pull-v2", SawyerStickPullEnvV2),
        ("push-wall-v2", SawyerPushWallEnvV2),
        ("push-v2", SawyerPushEnvV2),
        ("reach-wall-v2", SawyerReachWallEnvV2),
        ("reach-v2", SawyerReachEnvV2),
        ("shelf-place-v2", SawyerShelfPlaceEnvV2),
        ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
        ("sweep-v2", SawyerSweepEnvV2),
        ("window-open-v2", SawyerWindowOpenEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
    )
)

MT50_V2_ARGS_KWARGS = {
    key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT50_V2.items()
}

ML45_V2 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("assembly-v2", SawyerNutAssemblyEnvV2),
                    ("basketball-v2", SawyerBasketballEnvV2),
                    ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
                    ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
                    ("button-press-v2", SawyerButtonPressEnvV2),
                    ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
                    ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
                    ("coffee-pull-v2", SawyerCoffeePullEnvV2),
                    ("coffee-push-v2", SawyerCoffeePushEnvV2),
                    ("dial-turn-v2", SawyerDialTurnEnvV2),
                    ("disassemble-v2", SawyerNutDisassembleEnvV2),
                    ("door-close-v2", SawyerDoorCloseEnvV2),
                    ("door-open-v2", SawyerDoorEnvV2),
                    ("drawer-close-v2", SawyerDrawerCloseEnvV2),
                    ("drawer-open-v2", SawyerDrawerOpenEnvV2),
                    ("faucet-open-v2", SawyerFaucetOpenEnvV2),
                    ("faucet-close-v2", SawyerFaucetCloseEnvV2),
                    ("hammer-v2", SawyerHammerEnvV2),
                    ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
                    ("handle-press-v2", SawyerHandlePressEnvV2),
                    ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
                    ("handle-pull-v2", SawyerHandlePullEnvV2),
                    ("lever-pull-v2", SawyerLeverPullEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
                    ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
                    ("reach-v2", SawyerReachEnvV2),
                    ("push-back-v2", SawyerPushBackEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    ("plate-slide-v2", SawyerPlateSlideEnvV2),
                    ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
                    ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
                    ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
                    ("soccer-v2", SawyerSoccerEnvV2),
                    ("stick-push-v2", SawyerStickPushEnvV2),
                    ("stick-pull-v2", SawyerStickPullEnvV2),
                    ("push-wall-v2", SawyerPushWallEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("reach-wall-v2", SawyerReachWallEnvV2),
                    ("reach-v2", SawyerReachEnvV2),
                    ("shelf-place-v2", SawyerShelfPlaceEnvV2),
                    ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
                    ("sweep-v2", SawyerSweepEnvV2),
                    ("window-open-v2", SawyerWindowOpenEnvV2),
                    ("window-close-v2", SawyerWindowCloseEnvV2),
                )
            ),
        ),
        (
            "test",
            OrderedDict(
                (
                    ("bin-picking-v2", SawyerBinPickingEnvV2),
                    ("box-close-v2", SawyerBoxCloseEnvV2),
                    ("hand-insert-v2", SawyerHandInsertEnvV2),
                    ("door-lock-v2", SawyerDoorLockEnvV2),
                    ("door-unlock-v2", SawyerDoorUnlockEnvV2),
                )
            ),
        ),
    )
)

ml45_train_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML45_V2["train"].items()
}

ml45_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML45_V2["test"].items()
}

ML45_ARGS_KWARGS = dict(
    train=ml45_train_args_kwargs,
    test=ml45_test_args_kwargs,
)


def create_hidden_goal_envs(arm_name, all_env_dict):
    hidden_goal_envs = {}
    for env_name, env_cls in all_env_dict.items():
        d = {}

        def initialize(env, seed=None, render_mode=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.render_mode = render_mode
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed=seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        env_name = f"{arm_name}-{env_name}"
        hg_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = f"{env_name}-goal-hidden"
        hg_env_name = f"{hg_env_name}GoalHidden"
        HiddenGoalEnvCls = type(hg_env_name, (env_cls,), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def create_observable_goal_envs(arm_name, all_env_dict):
    observable_goal_envs = {}
    for env_name, env_cls in all_env_dict.items():
        d = {}

        def initialize(env, seed=None, render_mode=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()

            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.render_mode = render_mode
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        env_name = f"{arm_name}-{env_name}"
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")
        og_env_key = f"{env_name}-goal-observable"
        og_env_name = f"{og_env_name}GoalObservable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


SAWYER_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs("sawyer", SAWYER_ENVIRONMENTS)
SAWYER_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs(
    "sawyer", SAWYER_ENVIRONMENTS
)

JACO_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs("jaco", JACO_ENVIRONMENTS)
JACO_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs(
    "jaco", JACO_ENVIRONMENTS
)

FETCH_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs("fetch", FETCH_ENVIRONMENTS)
FETCH_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs(
    "fetch", FETCH_ENVIRONMENTS
)


ALL_ENVIRONMENTS_GOAL_HIDDEN = OrderedDict(
    list(SAWYER_ENVIRONMENTS_GOAL_HIDDEN.items())
    + list(JACO_ENVIRONMENTS_GOAL_HIDDEN.items())
    + list(FETCH_ENVIRONMENTS_GOAL_HIDDEN.items())
)
ALL_ENVIRONMENTS_GOAL_OBSERVABLE = OrderedDict(
    list(SAWYER_ENVIRONMENTS_GOAL_OBSERVABLE.items())
    + list(JACO_ENVIRONMENTS_GOAL_OBSERVABLE.items())
    + list(FETCH_ENVIRONMENTS_GOAL_OBSERVABLE.items())
)

ALL_ENVIRONMENTS = OrderedDict(
    list(ALL_ENVIRONMENTS_GOAL_HIDDEN.items())
    + list(ALL_ENVIRONMENTS_GOAL_OBSERVABLE.items())
)
