import re
from collections import OrderedDict

import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerBasketballEnvV2, SawyerBinPickingEnvV2, SawyerBoxCloseEnvV2,
    SawyerButtonPressEnvV2, SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnvV2, SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2, SawyerCoffeePullEnvV2, SawyerCoffeePushEnvV2,
    SawyerDialTurnEnvV2, SawyerDoorCloseEnvV2, SawyerDoorEnvV2,
    SawyerDoorLockEnvV2, SawyerDoorUnlockEnvV2, SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnvV2, SawyerFaucetCloseEnvV2, SawyerFaucetOpenEnvV2,
    SawyerHammerEnvV2, SawyerHandInsertEnvV2, SawyerHandlePressEnvV2,
    SawyerHandlePressSideEnvV2, SawyerHandlePullEnvV2,
    SawyerHandlePullSideEnvV2, SawyerLeverPullEnvV2, SawyerNutAssemblyEnvV2,
    SawyerNutDisassembleEnvV2, SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnvV2, SawyerPickOutOfHoleEnvV2, SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2, SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideBackSideEnvV2, SawyerPlateSlideEnvV2,
    SawyerPlateSlideSideEnvV2, SawyerPushBackEnvV2, SawyerPushEnvV2,
    SawyerPushWallEnvV2, SawyerReachEnvV2, SawyerReachWallEnvV2,
    SawyerShelfPlaceEnvV2, SawyerSoccerEnvV2, SawyerStickPullEnvV2,
    SawyerStickPushEnvV2, SawyerSweepEnvV2, SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnvV2, SawyerWindowOpenEnvV2)

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    TrainAssemblyv3, TestAssemblyv3, TrainBasketballv3, TestBasketballv3, TrainBinPickingv3, TestBinPickingv3,
    TrainBoxClosev3, TestBoxClosev3, TrainButtonPressv3, TestButtonPressv3, TrainButtonPressWallv3, TestButtonPressWallv3,
    TrainButtonPressTopdownv3, TestButtonPressTopdownv3, TrainButtonPressTopdownWallv3, TestButtonPressTopdownWallv3,
    TrainCoffeePullv3, TestCoffeePullv3, TrainCoffeePushv3, TestCoffeePushv3, TrainCoffeeButtonv3, TestCoffeeButtonv3,
    TrainDialTurnv3, TestDialTurnv3, TrainDisassemblev3, TestDisassemblev3, TrainFaucetOpenv3, TestFaucetOpenv3,
    TrainFaucetClosev3, TestFaucetClosev3, TrainDrawerClosev3, TestDrawerClosev3, TrainDoorClosev3, TestDoorClosev3,
    TrainDoorLockv3, TestDoorLockv3, TrainDoorOpenv3, TestDoorOpenv3, TrainDoorUnlockv3, TestDoorUnlockv3,
    TrainDrawerOpenv3, TestDrawerOpenv3, TrainHammerv3, TestHammerv3, TrainHandlePullv3, TestHandlePullv3,
    TrainHandlePressv3, TestHandlePressv3, TrainHandlePullSidev3, TestHandlePullSidev3, TrainHandlePressSidev3,
    TestHandlePressSidev3, TrainHandInsertv3, TestHandInsertv3, TrainLeverPullv3, TestLeverPullv3, TrainPushv3,
    TestPushv3, TrainPushBackv3, TestPushBackv3, TrainPushWallv3, TestPushWallv3, TrainPickPlaceWallv3,
    TestPickPlaceWallv3, TrainPickPlacev3, TestPickPlacev3, TrainPlateSlidev3, TestPlateSlidev3, TrainPlateSlideBackv3,
    TestPlateSlideBackv3, TrainPlateSlideSidev3, TestPlateSlideSidev3, TrainPlateSlideBackSidev3,
    TestPlateSlideBackSidev3, TrainPegUnplugSidev3, TestPegUnplugSidev3, TrainPegInsertionSidev3,
    TestPegInsertionSidev3, TrainPickOutOfHolev3, TestPickOutOfHolev3, TrainReachv3, TestReachv3, TrainReachWallv3,
    TestReachWallv3, TrainSweepv3, TestSweepv3, TrainSoccerv3, TestSoccerv3, TrainSweepIntoGoalv3,
    TestSweepIntoGoalv3, TrainShelfPlacev3, TestShelfPlacev3, TrainStickPullv3, TestStickPullv3, TrainStickPushv3,
    TestStickPushv3, TrainWindowOpenv3, TestWindowOpenv3, TrainWindowClosev3, TestWindowClosev3)


ALL_V2_ENVIRONMENTS = OrderedDict(
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



_NUM_METAWORLD_ENVS = len(ALL_V2_ENVIRONMENTS)
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
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT10_V2.items()
}

ML10_V2 = OrderedDict(
    (('train',
      OrderedDict(
          (('reach-v2', SawyerReachEnvV2),
           ('push-v2', SawyerPushEnvV2),
           ('pick-place-v2', SawyerPickPlaceEnvV2),
           ('door-open-v2', SawyerDoorEnvV2),
           ('drawer-close-v2', SawyerDrawerCloseEnvV2),
           ('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2),
           ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
           ('window-open-v2', SawyerWindowOpenEnvV2),
           ('sweep-v2', SawyerSweepEnvV2),
           ('basketball-v2', SawyerBasketballEnvV2)))),
     ('test',
      OrderedDict(
          (('drawer-open-v2', SawyerDrawerOpenEnvV2),
           ('door-close-v2', SawyerDoorCloseEnvV2),
           ('shelf-place-v2', SawyerShelfPlaceEnvV2),
           ('sweep-into-v2', SawyerSweepIntoGoalEnvV2),
           ('lever-pull-v2', SawyerLeverPullEnvV2,
           ))))))


ml10_train_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML10_V2["train"].items()
}

ml10_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML10_V2["test"].items()
}

ML10_ARGS_KWARGS = dict(
    train=ml10_train_args_kwargs,
    test=ml10_test_args_kwargs,
)

ML1_V2 = OrderedDict((("train", ALL_V2_ENVIRONMENTS), ("test", ALL_V2_ENVIRONMENTS)))

ML1_args_kwargs = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
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
ML1_TRAIN_TEST_ENVS = OrderedDict(
    (('assembly-v2',
      OrderedDict((('train', TrainAssemblyv3),
                   ('test', TestAssemblyv3),
                   ))),
     ('basketball-v2',
      OrderedDict((('train', TrainBasketballv3),
                   ('test', TestBasketballv3),
                   ))),
     ('bin-picking-v2',
      OrderedDict((('train', TrainAssemblyv3),
                   ('test', TestAssemblyv3),
                   ))),
     ('box-close-v2',
      OrderedDict((('train', TrainBoxClosev3),
                   ('test', TestBoxClosev3),
                   ))),
     ('button-press-topdown-v2',
      OrderedDict((('train', TrainButtonPressTopdownv3),
                   ('test', TestButtonPressTopdownv3),
                   ))),
     ('button-press-topdown-wall-v2',
      OrderedDict((('train', TrainButtonPressTopdownWallv3),
                   ('test', TestButtonPressTopdownWallv3),
                   ))),
     ('button-press-v2',
      OrderedDict((('train', TrainButtonPressv3),
                   ('test', TestButtonPressv3),
                   ))),
     ('button-press-wall-v2',
      OrderedDict((('train', TrainButtonPressWallv3),
                   ('test', TestButtonPressWallv3),
                   ))),
     ('coffee-button-v2',
      OrderedDict((('train', TrainCoffeeButtonv3),
                   ('test', TestCoffeeButtonv3),
                   ))),
     ('coffee-pull-v2',
      OrderedDict((('train', TrainCoffeePullv3),
                   ('test', TestCoffeePullv3),
                   ))),
     ('coffee-push-v2',
      OrderedDict((('train', TrainCoffeePushv3),
                   ('test', TestCoffeePushv3),
                   ))),
     ('dial-turn-v2',
      OrderedDict((('train', TrainDialTurnv3),
                   ('test', TestDialTurnv3),
                   ))),
     ('disassemble-v2',
      OrderedDict((('train', TrainDisassemblev3),
                   ('test', TestDisassemblev3),
                   ))),
     ('door-close-v2',
      OrderedDict((('train', TrainDoorClosev3),
                   ('test', TestDoorClosev3),
                   ))),
     ('door-lock-v2',
      OrderedDict((('train', TrainDoorLockv3),
                   ('test', TestDoorLockv3),
                   ))),
     ('door-unlock-v2',
      OrderedDict((('train', TrainDoorUnlockv3),
                   ('test', TestDoorUnlockv3),
                   ))),
     ('door-open-v2',
      OrderedDict((('train', TrainDoorOpenv3),
                   ('test', TestDoorOpenv3),
                   ))),
     ('drawer-close-v2',
      OrderedDict((('train', TrainDrawerClosev3),
                   ('test', TestDrawerClosev3),
                   ))),
     ('drawer-open-v2',
      OrderedDict((('train', TrainDrawerOpenv3),
                   ('test', TestDrawerOpenv3),
                   ))),
     ('faucet-open-v2',
      OrderedDict((('train', TrainFaucetOpenv3),
                   ('test', TestFaucetOpenv3),
                   ))),
     ('faucet-close-v2',
      OrderedDict((('train', TrainFaucetClosev3),
                   ('test', TestFaucetClosev3),
                   ))),
     ('hammer-v2',
      OrderedDict((('train', TrainHammerv3),
                   ('test', TestHammerv3),
                   ))),
     ('hand-insert-v2',
      OrderedDict((('train', TrainHandInsertv3),
                   ('test', TestHandInsertv3),
                   ))),
     ('handle-press-side-v2',
      OrderedDict((('train', TrainButtonPressTopdownWallv3),
                   ('test', TestButtonPressTopdownWallv3),
                   ))),
     ('handle-press-v2',
      OrderedDict((('train', TrainHandlePressv3),
                   ('test', TestHandlePressv3),
                   ))),
     ('handle-pull-side-v2',
      OrderedDict((('train', TrainHandlePullSidev3),
                   ('test', TestHandlePullSidev3),
                   ))),
     ('handle-pull-v2',
      OrderedDict((('train', TrainHandlePullv3),
                   ('test', TestHandlePullv3),
                   ))),
     ('lever-pull-v2',
      OrderedDict((('train', TrainLeverPullv3),
                   ('test', TestLeverPullv3),
                   ))),
     ('peg-insert-side-v2',
      OrderedDict((('train', TrainPegInsertionSidev3),
                   ('test', TestPegInsertionSidev3),
                   ))),
     ('peg-unplug-side-v2',
      OrderedDict((('train', TrainPegUnplugSidev3),
                   ('test', TestPegUnplugSidev3),
                   ))),
     ('pick-place-wall-v2',
      OrderedDict((('train', TrainPickPlaceWallv3),
                   ('test', TestPickPlaceWallv3),
                   ))),
     ('pick-place-v2',
      OrderedDict((('train', TrainPickPlacev3),
                   ('test', TestPickPlacev3),
                   ))),
     ('pick-out-of-hole-v2',
      OrderedDict((('train', TrainPickOutOfHolev3),
                   ('test', TestPickOutOfHolev3),
                   ))),
     ('reach-v2',
      OrderedDict((('train', TrainReachv3),
                   ('test', TestReachv3),
                   ))),
     ('push-back-v2',
      OrderedDict((('train', TrainPushBackv3),
                   ('test', TestPushBackv3),
                   ))),
     ('push-v2',
      OrderedDict((('train', TrainPushv3),
                   ('test', TestPushv3),
                   ))),
     ('plate-slide-v2',
      OrderedDict((('train', TrainPlateSlidev3),
                   ('test', TestPlateSlidev3),
                   ))),
     ('plate-slide-side-v2',
      OrderedDict((('train', TrainPlateSlideSidev3),
                   ('test', TestPlateSlideSidev3),
                   ))),
     ('plate-slide-back-v2',
      OrderedDict((('train', TrainPlateSlideBackv3),
                   ('test', TestPlateSlideBackv3),
                   ))),
     ('plate-slide-back-side-v2',
      OrderedDict((('train', TrainPlateSlideBackSidev3),
                   ('test', TestPlateSlideBackSidev3),
                   ))),
     ('soccer-v2',
      OrderedDict((('train', TrainSoccerv3),
                   ('test', TestSoccerv3),
                   ))),
     ('stick-push-v2',
      OrderedDict((('train', TrainStickPushv3),
                   ('test', TestStickPushv3),
                   ))),
     ('stick-pull-v2',
      OrderedDict((('train', TrainStickPullv3),
                   ('test', TestStickPullv3),
                   ))),
     ('push-wall-v2',
      OrderedDict((('train', TrainPushWallv3),
                   ('test', TestPushWallv3),
                   ))),
     ('push-v2',
      OrderedDict((('train', TrainPushv3),
                   ('test', TestPushv3),
                   ))),
     ('reach-wall-v2',
      OrderedDict((('train', TrainReachWallv3),
                   ('test', TestReachWallv3),
                   ))),
     ('reach-v2',
      OrderedDict((('train', TrainReachv3),
                   ('test', TestReachv3),
                   ))),
     ('shelf-place-v2',
      OrderedDict((('train', TrainShelfPlacev3),
                   ('test', TestShelfPlacev3),
                   ))),
     ('sweep-into-v2',
      OrderedDict((('train', TrainSweepIntoGoalv3),
                   ('test', TestSweepIntoGoalv3),
                   ))),
     ('sweep-v2',
      OrderedDict((('train', TrainSweepv3),
                   ('test', TestSweepv3),
                   ))),
     ('window-open-v2',
      OrderedDict((('train', TrainWindowOpenv3),
                   ('test', TestWindowOpenv3),
                   ))),
     ('window-close-v2',
      OrderedDict((('train', TrainWindowClosev3),
                   ('test', TestWindowClosev3),
                   ))),

     ))



MT50_V2_ARGS_KWARGS = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
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
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ML45_V2["train"].items()
}

ml45_test_args_kwargs = {
    key: dict(args=[], kwargs={"task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML45_V2["test"].items()
}

ML45_ARGS_KWARGS = dict(
    train=ml45_train_args_kwargs,
    test=ml45_test_args_kwargs,
)


def create_hidden_goal_envs():
    hidden_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        hg_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = f"{env_name}-goal-hidden"
        hg_env_name = f"{hg_env_name}GoalHidden"
        HiddenGoalEnvCls = type(hg_env_name, (env_cls,), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()

            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")

        og_env_key = f"{env_name}-goal-observable"
        og_env_name = f"{og_env_name}GoalObservable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


ALL_V2_ENVIRONMENTS_GOAL_HIDDEN = create_hidden_goal_envs()
ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()
