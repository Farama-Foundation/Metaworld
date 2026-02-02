"""Dictionaries mapping environment name strings to environment classes,
and organising them into various collections and splits for the benchmarks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Literal
from typing import OrderedDict as Typing_OrderedDict
from typing import Union

from typing_extensions import TypeAlias

import gymnasium as gym
from metaworld import envs
from metaworld.sawyer_xyz_env import SawyerXYZEnv

# Utils

EnvDict: TypeAlias = "Typing_OrderedDict[str, type[SawyerXYZEnv]]"
TrainTestEnvDict: TypeAlias = "Typing_OrderedDict[Literal['train', 'test'], EnvDict]"
EnvArgsKwargsDict: TypeAlias = (
    "Dict[str, Dict[Literal['args', 'kwargs'], Union[List, Dict]]]"
)

ALL_V3_ENVIRONMENTS: OrderedDict[str, type[SawyerXYZEnv]] = OrderedDict([
    (getattr(env_class, 'ENV_NAME'), env_class)
    for env_class in [
        envs.SawyerNutAssemblyEnvV3,
        envs.SawyerBasketballEnvV3,
        envs.SawyerBinPickingEnvV3,
        envs.SawyerBoxCloseEnvV3,
        envs.SawyerButtonPressTopdownEnvV3,
        envs.SawyerButtonPressTopdownWallEnvV3,
        envs.SawyerButtonPressEnvV3,
        envs.SawyerButtonPressWallEnvV3,
        envs.SawyerCoffeeButtonEnvV3,
        envs.SawyerCoffeePullEnvV3,
        envs.SawyerCoffeePushEnvV3,
        envs.SawyerDialTurnEnvV3,
        envs.SawyerNutDisassembleEnvV3,
        envs.SawyerDoorCloseEnvV3,
        envs.SawyerDoorLockEnvV3,
        envs.SawyerDoorEnvV3,
        envs.SawyerDoorUnlockEnvV3,
        envs.SawyerHandInsertEnvV3,
        envs.SawyerDrawerCloseEnvV3,
        envs.SawyerDrawerOpenEnvV3,
        envs.SawyerFaucetOpenEnvV3,
        envs.SawyerFaucetCloseEnvV3,
        envs.SawyerHammerEnvV3,
        envs.SawyerHandlePressSideEnvV3,
        envs.SawyerHandlePressEnvV3,
        envs.SawyerHandlePullSideEnvV3,
        envs.SawyerHandlePullEnvV3,
        envs.SawyerLeverPullEnvV3,
        envs.SawyerPegInsertionSideEnvV3,
        envs.SawyerPickPlaceWallEnvV3,
        envs.SawyerPickOutOfHoleEnvV3,
        envs.SawyerReachEnvV3,
        envs.SawyerPushBackEnvV3,
        envs.SawyerPushEnvV3,
        envs.SawyerPickPlaceEnvV3,
        envs.SawyerPlateSlideEnvV3,
        envs.SawyerPlateSlideSideEnvV3,
        envs.SawyerPlateSlideBackEnvV3,
        envs.SawyerPlateSlideBackSideEnvV3,
        envs.SawyerPegUnplugSideEnvV3,
        envs.SawyerSoccerEnvV3,
        envs.SawyerStickPushEnvV3,
        envs.SawyerStickPullEnvV3,
        envs.SawyerPushWallEnvV3,
        envs.SawyerReachWallEnvV3,
        envs.SawyerShelfPlaceEnvV3,
        envs.SawyerSweepIntoGoalEnvV3,
        envs.SawyerSweepEnvV3,
        envs.SawyerWindowOpenEnvV3,
        envs.SawyerWindowCloseEnvV3,
    ]
])
"""Mapping from environment name to environment class for all V3 environments."""

ENV_NAMES = list(ALL_V3_ENVIRONMENTS.keys())
"""List of all V3 environment names."""

# MT Dicts

MT10_V3_ENV_NAMES = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
]

MT25_V3_ENV_NAMES = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
    "coffee-pull-v3",
    "pick-out-of-hole-v3",
    "disassemble-v3",
    "pick-place-wall-v3",
    "basketball-v3",
    "stick-pull-v3",
    "button-press-wall-v3",
    "faucet-open-v3",
    "door-lock-v3",
    "lever-pull-v3",
    "sweep-into-v3",
    "faucet-close-v3",
    "coffee-button-v3",
    "button-press-topdown-wall-v3",
    "dial-turn-v3",
]

MT50_V3_ENV_NAMES = ENV_NAMES

MT_BENCHMARKS_TRAIN_ENV_NAMES: dict[str, list[str]] = {
    "MT10": MT10_V3_ENV_NAMES,
    "MT25": MT25_V3_ENV_NAMES,
    "MT50": MT50_V3_ENV_NAMES,
}

# ML Dicts

ML10_V3 = {
    'train': [
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
        "window-open-v3",
        "sweep-v3",
        "basketball-v3",
    ],
    'test': [
        "drawer-open-v3",
        "door-close-v3",
        "shelf-place-v3",
        "sweep-into-v3",
        "lever-pull-v3",
    ],
}


ML25_V3 = {
    'train': [
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
        "window-open-v3",
        "window-close-v3",
        "coffee-pull-v3",
        "pick-out-of-hole-v3",
        "disassemble-v3",
        "pick-place-wall-v3",
        "basketball-v3",
        "stick-pull-v3",
        "button-press-wall-v3",
        "faucet-open-v3",
        "door-lock-v3",
        "lever-pull-v3",
        "sweep-into-v3",
        "faucet-close-v3",
        "coffee-button-v3",
        "button-press-topdown-wall-v3",
        "dial-turn-v3",
    ],
    'test': [
        "basketball-v3",
        "door-close-v3",
        "shelf-place-v3",
        "sweep-v3",
        "button-press-v3",
    ],
}

ML45_V3 = {
    'train': [
        "assembly-v3",
        "basketball-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "dial-turn-v3",
        "disassemble-v3",
        "door-close-v3",
        "door-open-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-open-v3",
        "faucet-close-v3",
        "hammer-v3",
        "handle-press-side-v3",
        "handle-press-v3",
        "handle-pull-side-v3",
        "handle-pull-v3",
        "lever-pull-v3",
        "pick-place-wall-v3",
        "pick-out-of-hole-v3",
        "push-back-v3",
        "pick-place-v3",
        "plate-slide-v3",
        "plate-slide-side-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "peg-insert-side-v3",
        "peg-unplug-side-v3",
        "soccer-v3",
        "stick-push-v3",
        "stick-pull-v3",
        "push-wall-v3",
        "push-v3",
        "reach-wall-v3",
        "reach-v3",
        "shelf-place-v3",
        "sweep-into-v3",
        "sweep-v3",
        "window-open-v3",
        "window-close-v3",
    ],
    'test': [
        "bin-picking-v3",
        "box-close-v3",
        "hand-insert-v3",
        "door-lock-v3",
        "door-unlock-v3",
    ],
}

ML_BENCHMARKS = {
    "ML10": ML10_V3,
    "ML25": ML25_V3,
    "ML45": ML45_V3,
}


def envs_get_env_names(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    """
    Get the environment names for each environment in a vectorized env.

    Args:
        envs: A vectorized gym environment containing Meta-World environments.
    Returns:
        A list of environment names corresponding to each environment in the vectorized env.
    """
    return [
        env_name
        for env_name in envs.get_attr("ENV_NAME")
    ]
