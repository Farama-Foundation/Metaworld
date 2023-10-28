import re
from collections import OrderedDict

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import *
from metaworld.envs.mujoco.jaco.v2 import *
from metaworld.envs.mujoco.fetch.v2 import *
from metaworld.envs.mujoco.ur5e.v2 import *

SAWYER_ENVIRONMENTS = OrderedDict(
    (
        ("assembly", SawyerNutAssemblyEnvV2),
        ("basketball", SawyerBasketballEnvV2),
        ("bin-picking", SawyerBinPickingEnvV2),
        ("box-close", SawyerBoxCloseEnvV2),
        ("button-press-topdown", SawyerButtonPressTopdownEnvV2),
        ("button-press-topdown-wall", SawyerButtonPressTopdownWallEnvV2),
        ("button-press", SawyerButtonPressEnvV2),
        ("button-press-wall", SawyerButtonPressWallEnvV2),
        ("coffee-button", SawyerCoffeeButtonEnvV2),
        ("coffee-pull", SawyerCoffeePullEnvV2),
        ("coffee-push", SawyerCoffeePushEnvV2),
        ("dial-turn", SawyerDialTurnEnvV2),
        ("disassemble", SawyerNutDisassembleEnvV2),
        ("door-close", SawyerDoorCloseEnvV2),
        ("door-lock", SawyerDoorLockEnvV2),
        ("door-open", SawyerDoorEnvV2),
        ("door-unlock", SawyerDoorUnlockEnvV2),
        ("hand-insert", SawyerHandInsertEnvV2),
        ("drawer-close", SawyerDrawerCloseEnvV2),
        ("drawer-open", SawyerDrawerOpenEnvV2),
        ("faucet-open", SawyerFaucetOpenEnvV2),
        ("faucet-close", SawyerFaucetCloseEnvV2),
        ("hammer", SawyerHammerEnvV2),
        ("handle-press-side", SawyerHandlePressSideEnvV2),
        ("handle-press", SawyerHandlePressEnvV2),
        ("handle-pull-side", SawyerHandlePullSideEnvV2),
        ("handle-pull", SawyerHandlePullEnvV2),
        ("lever-pull", SawyerLeverPullEnvV2),
        ("peg-insert-side", SawyerPegInsertionSideEnvV2),
        ("pick-place-wall", SawyerPickPlaceWallEnvV2),
        ("pick-out-of-hole", SawyerPickOutOfHoleEnvV2),
        ("push-back", SawyerPushBackEnvV2),
        ("push", SawyerPushEnvV2),
        ("pick-place", SawyerPickPlaceEnvV2),
        ("plate-slide", SawyerPlateSlideEnvV2),
        ("plate-slide-side", SawyerPlateSlideSideEnvV2),
        ("plate-slide-back", SawyerPlateSlideBackEnvV2),
        ("plate-slide-back-side", SawyerPlateSlideBackSideEnvV2),
        ("peg-insert-side", SawyerPegInsertionSideEnvV2),
        ("peg-unplug-side", SawyerPegUnplugSideEnvV2),
        ("soccer", SawyerSoccerEnvV2),
        ("stick-push", SawyerStickPushEnvV2),
        ("stick-pull", SawyerStickPullEnvV2),
        ("push-wall", SawyerPushWallEnvV2),
        ("push", SawyerPushEnvV2),
        ("reach-wall", SawyerReachWallEnvV2),
        ("reach", SawyerReachEnvV2),
        ("reach-goal-as-obj", SawyerReachGoalAsObjEnvV2),
        ("reach-top-approach", SawyerReachTopApproachEnvV2),
        ("shelf-place", SawyerShelfPlaceEnvV2),
        ("sweep-into", SawyerSweepIntoGoalEnvV2),
        ("sweep", SawyerSweepEnvV2),
        ("grip", SawyerGripEnvV2),
        ("window-open", SawyerWindowOpenEnvV2),
        ("window-close", SawyerWindowCloseEnvV2),
    )
)

JACO_ENVIRONMENTS = OrderedDict(
    (
        ("assembly", JacoNutAssemblyEnvV2),
        ("basketball", JacoBasketballEnvV2),
        ("bin-picking", JacoBinPickingEnvV2),
        ("box-close", JacoBoxCloseEnvV2),
        ("button-press-topdown", JacoButtonPressTopdownEnvV2),
        ("button-press-topdown-wall", JacoButtonPressTopdownWallEnvV2),
        ("button-press", JacoButtonPressEnvV2),
        ("button-press-wall", JacoButtonPressWallEnvV2),
        ("coffee-button", JacoCoffeeButtonEnvV2),
        ("coffee-pull", JacoCoffeePullEnvV2),
        ("coffee-push", JacoCoffeePushEnvV2),
        ("dial-turn", JacoDialTurnEnvV2),
        ("disassemble", JacoNutDisassembleEnvV2),
        ("door-close", JacoDoorCloseEnvV2),
        ("door-lock", JacoDoorLockEnvV2),
        ("door-open", JacoDoorEnvV2),
        ("door-unlock", JacoDoorUnlockEnvV2),
        ("hand-insert", JacoHandInsertEnvV2),
        ("drawer-close", JacoDrawerCloseEnvV2),
        ("drawer-open", JacoDrawerOpenEnvV2),
        ("faucet-open", JacoFaucetOpenEnvV2),
        ("faucet-close", JacoFaucetCloseEnvV2),
        ("hammer", JacoHammerEnvV2),
        ("handle-press-side", JacoHandlePressSideEnvV2),
        ("handle-press", JacoHandlePressEnvV2),
        ("handle-pull-side", JacoHandlePullSideEnvV2),
        ("handle-pull", JacoHandlePullEnvV2),
        ("lever-pull", JacoLeverPullEnvV2),
        ("peg-insert-side", JacoPegInsertionSideEnvV2),
        ("pick-place-wall", JacoPickPlaceWallEnvV2),
        ("pick-out-of-hole", JacoPickOutOfHoleEnvV2),
        ("push-back", JacoPushBackEnvV2),
        ("push", JacoPushEnvV2),
        ("pick-place", JacoPickPlaceEnvV2),
        ("plate-slide", JacoPlateSlideEnvV2),
        ("plate-slide-side", JacoPlateSlideSideEnvV2),
        ("plate-slide-back", JacoPlateSlideBackEnvV2),
        ("plate-slide-back-side", JacoPlateSlideBackSideEnvV2),
        ("peg-insert-side", JacoPegInsertionSideEnvV2),
        ("peg-unplug-side", JacoPegUnplugSideEnvV2),
        ("soccer", JacoSoccerEnvV2),
        ("stick-push", JacoStickPushEnvV2),
        ("stick-pull", JacoStickPullEnvV2),
        ("push-wall", JacoPushWallEnvV2),
        ("push", JacoPushEnvV2),
        ("reach-wall", JacoReachWallEnvV2),
        ("reach", JacoReachEnvV2),
        ("reach-goal-as-obj", JacoReachGoalAsObjEnvV2),
        ("shelf-place", JacoShelfPlaceEnvV2),
        ("sweep-into", JacoSweepIntoGoalEnvV2),
        ("sweep", JacoSweepEnvV2),
        ("window-open", JacoWindowOpenEnvV2),
        ("window-close", JacoWindowCloseEnvV2),
    )
)

FETCH_ENVIRONMENTS = OrderedDict(
    (
        ("assembly", FetchNutAssemblyEnvV2),
        ("basketball", FetchBasketballEnvV2),
        ("bin-picking", FetchBinPickingEnvV2),
        ("box-close", FetchBoxCloseEnvV2),
        ("button-press-topdown", FetchButtonPressTopdownEnvV2),
        ("button-press-topdown-wall", FetchButtonPressTopdownWallEnvV2),
        ("button-press", FetchButtonPressEnvV2),
        ("button-press-wall", FetchButtonPressWallEnvV2),
        ("coffee-button", FetchCoffeeButtonEnvV2),
        ("coffee-pull", FetchCoffeePullEnvV2),
        ("coffee-push", FetchCoffeePushEnvV2),
        ("dial-turn", FetchDialTurnEnvV2),
        ("disassemble", FetchNutDisassembleEnvV2),
        ("door-close", FetchDoorCloseEnvV2),
        ("door-lock", FetchDoorLockEnvV2),
        ("door-open", FetchDoorEnvV2),
        ("door-unlock", FetchDoorUnlockEnvV2),
        ("hand-insert", FetchHandInsertEnvV2),
        ("drawer-close", FetchDrawerCloseEnvV2),
        ("drawer-open", FetchDrawerOpenEnvV2),
        ("faucet-open", FetchFaucetOpenEnvV2),
        ("faucet-close", FetchFaucetCloseEnvV2),
        ("hammer", FetchHammerEnvV2),
        ("handle-press-side", FetchHandlePressSideEnvV2),
        ("handle-press", FetchHandlePressEnvV2),
        ("handle-pull-side", FetchHandlePullSideEnvV2),
        ("handle-pull", FetchHandlePullEnvV2),
        ("lever-pull", FetchLeverPullEnvV2),
        ("peg-insert-side", FetchPegInsertionSideEnvV2),
        ("pick-place-wall", FetchPickPlaceWallEnvV2),
        ("pick-out-of-hole", FetchPickOutOfHoleEnvV2),
        ("push-back", FetchPushBackEnvV2),
        ("push", FetchPushEnvV2),
        ("pick-place", FetchPickPlaceEnvV2),
        ("plate-slide", FetchPlateSlideEnvV2),
        ("plate-slide-side", FetchPlateSlideSideEnvV2),
        ("plate-slide-back", FetchPlateSlideBackEnvV2),
        ("plate-slide-back-side", FetchPlateSlideBackSideEnvV2),
        ("peg-insert-side", FetchPegInsertionSideEnvV2),
        ("peg-unplug-side", FetchPegUnplugSideEnvV2),
        ("soccer", FetchSoccerEnvV2),
        ("stick-push", FetchStickPushEnvV2),
        ("stick-pull", FetchStickPullEnvV2),
        ("push-wall", FetchPushWallEnvV2),
        ("push", FetchPushEnvV2),
        ("reach-wall", FetchReachWallEnvV2),
        ("reach", FetchReachEnvV2),
        ("reach-goal-as-obj", FetchReachGoalAsObjEnvV2),
        ("shelf-place", FetchShelfPlaceEnvV2),
        ("sweep-into", FetchSweepIntoGoalEnvV2),
        ("sweep", FetchSweepEnvV2),
        ("window-open", FetchWindowOpenEnvV2),
        ("window-close", FetchWindowCloseEnvV2),
    )
)

UR5E_ENVIRONMENTS = OrderedDict(
    (
        ("assembly", UR5eNutAssemblyEnvV2),
        ("basketball", UR5eBasketballEnvV2),
        ("bin-picking", UR5eBinPickingEnvV2),
        ("box-close", UR5eBoxCloseEnvV2),
        ("button-press-topdown", UR5eButtonPressTopdownEnvV2),
        ("button-press-topdown-wall", UR5eButtonPressTopdownWallEnvV2),
        ("button-press", UR5eButtonPressEnvV2),
        ("button-press-wall", UR5eButtonPressWallEnvV2),
        ("coffee-button", UR5eCoffeeButtonEnvV2),
        ("coffee-pull", UR5eCoffeePullEnvV2),
        ("coffee-push", UR5eCoffeePushEnvV2),
        ("dial-turn", UR5eDialTurnEnvV2),
        ("disassemble", UR5eNutDisassembleEnvV2),
        ("door-close", UR5eDoorCloseEnvV2),
        ("door-lock", UR5eDoorLockEnvV2),
        ("door-open", UR5eDoorEnvV2),
        ("door-unlock", UR5eDoorUnlockEnvV2),
        ("hand-insert", UR5eHandInsertEnvV2),
        ("drawer-close", UR5eDrawerCloseEnvV2),
        ("drawer-open", UR5eDrawerOpenEnvV2),
        ("faucet-open", UR5eFaucetOpenEnvV2),
        ("faucet-close", UR5eFaucetCloseEnvV2),
        ("hammer", UR5eHammerEnvV2),
        ("handle-press-side", UR5eHandlePressSideEnvV2),
        ("handle-press", UR5eHandlePressEnvV2),
        ("handle-pull-side", UR5eHandlePullSideEnvV2),
        ("handle-pull", UR5eHandlePullEnvV2),
        ("lever-pull", UR5eLeverPullEnvV2),
        ("peg-insert-side", UR5ePegInsertionSideEnvV2),
        ("pick-place-wall", UR5ePickPlaceWallEnvV2),
        ("pick-out-of-hole", UR5ePickOutOfHoleEnvV2),
        ("push-back", UR5ePushBackEnvV2),
        ("push", UR5ePushEnvV2),
        ("pick-place", UR5ePickPlaceEnvV2),
        ("plate-slide", UR5ePlateSlideEnvV2),
        ("plate-slide-side", UR5ePlateSlideSideEnvV2),
        ("plate-slide-back", UR5ePlateSlideBackEnvV2),
        ("plate-slide-back-side", UR5ePlateSlideBackSideEnvV2),
        ("peg-insert-side", UR5ePegInsertionSideEnvV2),
        ("peg-unplug-side", UR5ePegUnplugSideEnvV2),
        ("soccer", UR5eSoccerEnvV2),
        ("stick-push", UR5eStickPushEnvV2),
        ("stick-pull", UR5eStickPullEnvV2),
        ("push-wall", UR5ePushWallEnvV2),
        ("push", UR5ePushEnvV2),
        ("reach-wall", UR5eReachWallEnvV2),
        ("reach", UR5eReachEnvV2),
        ("reach-goal-as-obj", UR5eReachGoalAsObjEnvV2),
        ("shelf-place", UR5eShelfPlaceEnvV2),
        ("sweep-into", UR5eSweepIntoGoalEnvV2),
        ("sweep", UR5eSweepEnvV2),
        ("window-open", UR5eWindowOpenEnvV2),
        ("window-close", UR5eWindowCloseEnvV2),
    )
)

_NUM_METAWORLD_ENVS = len(SAWYER_ENVIRONMENTS)
# V2 DICTS

# MT10_V2 = OrderedDict(
#     (
#         ("reach-v2", SawyerReachEnvV2),
#         ("push-v2", SawyerPushEnvV2),
#         ("pick-place-v2", SawyerPickPlaceEnvV2),
#         ("door-open-v2", SawyerDoorEnvV2),
#         ("drawer-open-v2", SawyerDrawerOpenEnvV2),
#         ("drawer-close-v2", SawyerDrawerCloseEnvV2),
#         ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
#         ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
#         ("window-open-v2", SawyerWindowOpenEnvV2),
#         ("window-close-v2", SawyerWindowCloseEnvV2),
#     ),
# )


# MT10_V2_ARGS_KWARGS = {
#     key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
#     for key, _ in MT10_V2.items()
# }

# ML10_V2 = OrderedDict(
#     (
#         (
#             "train",
#             OrderedDict(
#                 (
#                     ("reach-v2", SawyerReachEnvV2),
#                     ("push-v2", SawyerPushEnvV2),
#                     ("pick-place-v2", SawyerPickPlaceEnvV2),
#                     ("door-open-v2", SawyerDoorEnvV2),
#                     ("drawer-close-v2", SawyerDrawerCloseEnvV2),
#                     ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
#                     ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
#                     ("window-open-v2", SawyerWindowOpenEnvV2),
#                     ("sweep-v2", SawyerSweepEnvV2),
#                     ("basketball-v2", SawyerBasketballEnvV2),
#                 )
#             ),
#         ),
#         (
#             "test",
#             OrderedDict(
#                 (
#                     ("drawer-open-v2", SawyerDrawerOpenEnvV2),
#                     ("door-close-v2", SawyerDoorCloseEnvV2),
#                     ("shelf-place-v2", SawyerShelfPlaceEnvV2),
#                     ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
#                     (
#                         "lever-pull-v2",
#                         SawyerLeverPullEnvV2,
#                     ),
#                 )
#             ),
#         ),
#     )
# )


# ml10_train_args_kwargs = {
#     key: dict(
#         args=[],
#         kwargs={
#             "task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key),
#         },
#     )
#     for key, _ in ML10_V2["train"].items()
# }

# ml10_test_args_kwargs = {
#     key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
#     for key, _ in ML10_V2["test"].items()
# }

# ML10_ARGS_KWARGS = dict(
#     train=ml10_train_args_kwargs,
#     test=ml10_test_args_kwargs,
# )

# ML1_V2 = OrderedDict((("train", SAWYER_ENVIRONMENTS), ("test", SAWYER_ENVIRONMENTS)))

# ML1_args_kwargs = {
#     key: dict(
#         args=[],
#         kwargs={
#             "task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key),
#         },
#     )
#     for key, _ in ML1_V2["train"].items()
# }
# MT50_V2 = OrderedDict(
#     (
#         ("assembly-v2", SawyerNutAssemblyEnvV2),
#         ("basketball-v2", SawyerBasketballEnvV2),
#         ("bin-picking-v2", SawyerBinPickingEnvV2),
#         ("box-close-v2", SawyerBoxCloseEnvV2),
#         ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
#         ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
#         ("button-press-v2", SawyerButtonPressEnvV2),
#         ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
#         ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
#         ("coffee-pull-v2", SawyerCoffeePullEnvV2),
#         ("coffee-push-v2", SawyerCoffeePushEnvV2),
#         ("dial-turn-v2", SawyerDialTurnEnvV2),
#         ("disassemble-v2", SawyerNutDisassembleEnvV2),
#         ("door-close-v2", SawyerDoorCloseEnvV2),
#         ("door-lock-v2", SawyerDoorLockEnvV2),
#         ("door-open-v2", SawyerDoorEnvV2),
#         ("door-unlock-v2", SawyerDoorUnlockEnvV2),
#         ("hand-insert-v2", SawyerHandInsertEnvV2),
#         ("drawer-close-v2", SawyerDrawerCloseEnvV2),
#         ("drawer-open-v2", SawyerDrawerOpenEnvV2),
#         ("faucet-open-v2", SawyerFaucetOpenEnvV2),
#         ("faucet-close-v2", SawyerFaucetCloseEnvV2),
#         ("hammer-v2", SawyerHammerEnvV2),
#         ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
#         ("handle-press-v2", SawyerHandlePressEnvV2),
#         ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
#         ("handle-pull-v2", SawyerHandlePullEnvV2),
#         ("lever-pull-v2", SawyerLeverPullEnvV2),
#         ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
#         ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
#         ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
#         ("reach-v2", SawyerReachEnvV2),
#         ("push-back-v2", SawyerPushBackEnvV2),
#         ("push-v2", SawyerPushEnvV2),
#         ("pick-place-v2", SawyerPickPlaceEnvV2),
#         ("plate-slide-v2", SawyerPlateSlideEnvV2),
#         ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
#         ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
#         ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
#         ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
#         ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
#         ("soccer-v2", SawyerSoccerEnvV2),
#         ("stick-push-v2", SawyerStickPushEnvV2),
#         ("stick-pull-v2", SawyerStickPullEnvV2),
#         ("push-wall-v2", SawyerPushWallEnvV2),
#         ("push-v2", SawyerPushEnvV2),
#         ("reach-wall-v2", SawyerReachWallEnvV2),
#         ("reach-v2", SawyerReachEnvV2),
#         ("shelf-place-v2", SawyerShelfPlaceEnvV2),
#         ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
#         ("sweep-v2", SawyerSweepEnvV2),
#         ("window-open-v2", SawyerWindowOpenEnvV2),
#         ("window-close-v2", SawyerWindowCloseEnvV2),
#     )
# )

# MT50_V2_ARGS_KWARGS = {
#     key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
#     for key, _ in MT50_V2.items()
# }

# ML45_V2 = OrderedDict(
#     (
#         (
#             "train",
#             OrderedDict(
#                 (
#                     ("assembly-v2", SawyerNutAssemblyEnvV2),
#                     ("basketball-v2", SawyerBasketballEnvV2),
#                     ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
#                     ("button-press-topdown-wall-v2", SawyerButtonPressTopdownWallEnvV2),
#                     ("button-press-v2", SawyerButtonPressEnvV2),
#                     ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
#                     ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
#                     ("coffee-pull-v2", SawyerCoffeePullEnvV2),
#                     ("coffee-push-v2", SawyerCoffeePushEnvV2),
#                     ("dial-turn-v2", SawyerDialTurnEnvV2),
#                     ("disassemble-v2", SawyerNutDisassembleEnvV2),
#                     ("door-close-v2", SawyerDoorCloseEnvV2),
#                     ("door-open-v2", SawyerDoorEnvV2),
#                     ("drawer-close-v2", SawyerDrawerCloseEnvV2),
#                     ("drawer-open-v2", SawyerDrawerOpenEnvV2),
#                     ("faucet-open-v2", SawyerFaucetOpenEnvV2),
#                     ("faucet-close-v2", SawyerFaucetCloseEnvV2),
#                     ("hammer-v2", SawyerHammerEnvV2),
#                     ("handle-press-side-v2", SawyerHandlePressSideEnvV2),
#                     ("handle-press-v2", SawyerHandlePressEnvV2),
#                     ("handle-pull-side-v2", SawyerHandlePullSideEnvV2),
#                     ("handle-pull-v2", SawyerHandlePullEnvV2),
#                     ("lever-pull-v2", SawyerLeverPullEnvV2),
#                     ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
#                     ("pick-place-wall-v2", SawyerPickPlaceWallEnvV2),
#                     ("pick-out-of-hole-v2", SawyerPickOutOfHoleEnvV2),
#                     ("reach-v2", SawyerReachEnvV2),
#                     ("push-back-v2", SawyerPushBackEnvV2),
#                     ("push-v2", SawyerPushEnvV2),
#                     ("pick-place-v2", SawyerPickPlaceEnvV2),
#                     ("plate-slide-v2", SawyerPlateSlideEnvV2),
#                     ("plate-slide-side-v2", SawyerPlateSlideSideEnvV2),
#                     ("plate-slide-back-v2", SawyerPlateSlideBackEnvV2),
#                     ("plate-slide-back-side-v2", SawyerPlateSlideBackSideEnvV2),
#                     ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
#                     ("peg-unplug-side-v2", SawyerPegUnplugSideEnvV2),
#                     ("soccer-v2", SawyerSoccerEnvV2),
#                     ("stick-push-v2", SawyerStickPushEnvV2),
#                     ("stick-pull-v2", SawyerStickPullEnvV2),
#                     ("push-wall-v2", SawyerPushWallEnvV2),
#                     ("push-v2", SawyerPushEnvV2),
#                     ("reach-wall-v2", SawyerReachWallEnvV2),
#                     ("reach-v2", SawyerReachEnvV2),
#                     ("shelf-place-v2", SawyerShelfPlaceEnvV2),
#                     ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
#                     ("sweep-v2", SawyerSweepEnvV2),
#                     ("window-open-v2", SawyerWindowOpenEnvV2),
#                     ("window-close-v2", SawyerWindowCloseEnvV2),
#                 )
#             ),
#         ),
#         (
#             "test",
#             OrderedDict(
#                 (
#                     ("bin-picking-v2", SawyerBinPickingEnvV2),
#                     ("box-close-v2", SawyerBoxCloseEnvV2),
#                     ("hand-insert-v2", SawyerHandInsertEnvV2),
#                     ("door-lock-v2", SawyerDoorLockEnvV2),
#                     ("door-unlock-v2", SawyerDoorUnlockEnvV2),
#                 )
#             ),
#         ),
#     )
# )

# ml45_train_args_kwargs = {
#     key: dict(
#         args=[],
#         kwargs={
#             "task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key),
#         },
#     )
#     for key, _ in ML45_V2["train"].items()
# }

# ml45_test_args_kwargs = {
#     key: dict(args=[], kwargs={"task_id": list(SAWYER_ENVIRONMENTS.keys()).index(key)})
#     for key, _ in ML45_V2["test"].items()
# }

# ML45_ARGS_KWARGS = dict(
#     train=ml45_train_args_kwargs,
#     test=ml45_test_args_kwargs,
# )


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
        hg_env_key = f"{env_name}-hidden"
        hg_env_name = f"{hg_env_name}Hidden"
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
        og_env_key = f"{env_name}-observable"
        og_env_name = f"{og_env_name}Observable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


def create_observable_random_goal_envs(arm_name, all_env_dict):
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
            # env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        env_name = f"{arm_name}-{env_name}"
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")
        og_env_key = f"{env_name}-random"
        og_env_name = f"{og_env_name}Random"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


ARMS = ["sawyer", "jaco", "fetch", "ur5e"]
ARM_ENVS = [
    SAWYER_ENVIRONMENTS,
    JACO_ENVIRONMENTS,
    FETCH_ENVIRONMENTS,
    UR5E_ENVIRONMENTS,
]
ENV_CONSTRUCTORS = [
    create_hidden_goal_envs,
    create_observable_goal_envs,
    create_observable_random_goal_envs,
]

ALL_ENVIRONMENTS = OrderedDict()
for env_constructor in ENV_CONSTRUCTORS:
    for arm, arm_envs in zip(ARMS, ARM_ENVS):
        arm_envs_goal = env_constructor(arm, arm_envs)
        ALL_ENVIRONMENTS.update(arm_envs_goal)
