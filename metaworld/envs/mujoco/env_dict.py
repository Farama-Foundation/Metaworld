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
    TrainAssemblyv2, TestAssemblyv2, TrainBasketballv2, TestBasketballv2, TrainBinPickingv2, TestBinPickingv2,
    TrainBoxClosev2, TestBoxClosev2, TrainButtonPressv2, TestButtonPressv2, TrainButtonPressWallv2, TestButtonPressWallv2,
    TrainButtonPressTopdownv2, TestButtonPressTopdownv2, TrainButtonPressTopdownWallv2, TestButtonPressTopdownWallv2,
    TrainCoffeePullv2, TestCoffeePullv2, TrainCoffeePushv2, TestCoffeePushv2, TrainCoffeeButtonv2, TestCoffeeButtonv2,
    TrainDialTurnv2, TestDialTurnv2, TrainDisassemblev2, TestDisassemblev2, TrainFaucetOpenv2, TestFaucetOpenv2,
    TrainFaucetClosev2, TestFaucetClosev2, TrainDrawerClosev2, TestDrawerClosev2, TrainDoorClosev2, TestDoorClosev2,
    TrainDoorLockv2, TestDoorLockv2, TrainDoorOpenv2, TestDoorOpenv2, TrainDoorUnlockv2, TestDoorUnlockv2,
    TrainDrawerOpenv2, TestDrawerOpenv2, TrainHammerv2, TestHammerv2, TrainHandlePullv2, TestHandlePullv2,
    TrainHandlePressv2, TestHandlePressv2, TrainHandlePullSidev2, TestHandlePullSidev2, TrainHandlePressSidev2,
    TestHandlePressSidev2, TrainHandInsertv2, TestHandInsertv2, TrainLeverPullv2, TestLeverPullv2, TrainPushv2,
    TestPushv2, TrainPushBackv2, TestPushBackv2, TrainPushWallv2, TestPushWallv2, TrainPickPlaceWallv2,
    TestPickPlaceWallv2, TrainPickPlacev2, TestPickPlacev2, TrainPlateSlidev2, TestPlateSlidev2, TrainPlateSlideBackv2,
    TestPlateSlideBackv2, TrainPlateSlideSidev2, TestPlateSlideSidev2, TrainPlateSlideBackSidev2,
    TestPlateSlideBackSidev2, TrainPegUnplugSidev2, TestPegUnplugSidev2, TrainPegInsertionSidev2,
    TestPegInsertionSidev2, TrainPickOutOfHolev2, TestPickOutOfHolev2, TrainReachv2, TestReachv2, TrainReachWallv2,
    TestReachWallv2, TrainSweepv2, TestSweepv2, TrainSoccerv2, TestSoccerv2, TrainSweepIntoGoalv2,
    TestSweepIntoGoalv2, TrainShelfPlacev2, TestShelfPlacev2, TrainStickPullv2, TestStickPullv2, TrainStickPushv2,
    TestStickPushv2, TrainWindowOpenv2, TestWindowOpenv2, TrainWindowClosev2, TestWindowClosev2)


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
      OrderedDict((('train', TrainAssemblyv2),
                   ('test', TestAssemblyv2),
                   ))),
     ('basketball-v2',
      OrderedDict((('train', TrainBasketballv2),
                   ('test', TestBasketballv2),
                   ))),
     ('bin-picking-v2',
      OrderedDict((('train', TrainAssemblyv2),
                   ('test', TestAssemblyv2),
                   ))),
     ('box-close-v2',
      OrderedDict((('train', TrainBoxClosev2),
                   ('test', TestBoxClosev2),
                   ))),
     ('button-press-topdown-v2',
      OrderedDict((('train', TrainButtonPressTopdownv2),
                   ('test', TestButtonPressTopdownv2),
                   ))),
     ('button-press-topdown-wall-v2',
      OrderedDict((('train', TrainButtonPressTopdownWallv2),
                   ('test', TestButtonPressTopdownWallv2),
                   ))),
     ('button-press-v2',
      OrderedDict((('train', TrainButtonPressv2),
                   ('test', TestButtonPressv2),
                   ))),
     ('button-press-wall-v2',
      OrderedDict((('train', TrainButtonPressWallv2),
                   ('test', TestButtonPressWallv2),
                   ))),
     ('coffee-button-v2',
      OrderedDict((('train', TrainCoffeeButtonv2),
                   ('test', TestCoffeeButtonv2),
                   ))),
     ('coffee-pull-v2',
      OrderedDict((('train', TrainCoffeePullv2),
                   ('test', TestCoffeePullv2),
                   ))),
     ('coffee-push-v2',
      OrderedDict((('train', TrainCoffeePushv2),
                   ('test', TestCoffeePushv2),
                   ))),
     ('dial-turn-v2',
      OrderedDict((('train', TrainDialTurnv2),
                   ('test', TestDialTurnv2),
                   ))),
     ('disassemble-v2',
      OrderedDict((('train', TrainDisassemblev2),
                   ('test', TestDisassemblev2),
                   ))),
     ('door-close-v2',
      OrderedDict((('train', TrainDoorClosev2),
                   ('test', TestDoorClosev2),
                   ))),
     ('door-lock-v2',
      OrderedDict((('train', TrainDoorLockv2),
                   ('test', TestDoorLockv2),
                   ))),
     ('door-unlock-v2',
      OrderedDict((('train', TrainDoorUnlockv2),
                   ('test', TestDoorUnlockv2),
                   ))),
     ('door-open-v2',
      OrderedDict((('train', TrainDoorOpenv2),
                   ('test', TestDoorOpenv2),
                   ))),
     ('drawer-close-v2',
      OrderedDict((('train', TrainDrawerClosev2),
                   ('test', TestDrawerClosev2),
                   ))),
     ('drawer-open-v2',
      OrderedDict((('train', TrainDrawerOpenv2),
                   ('test', TestDrawerOpenv2),
                   ))),
     ('faucet-open-v2',
      OrderedDict((('train', TrainFaucetOpenv2),
                   ('test', TestFaucetOpenv2),
                   ))),
     ('faucet-close-v2',
      OrderedDict((('train', TrainFaucetClosev2),
                   ('test', TestFaucetClosev2),
                   ))),
     ('hammer-v2',
      OrderedDict((('train', TrainHammerv2),
                   ('test', TestHammerv2),
                   ))),
     ('hand-insert-v2',
      OrderedDict((('train', TrainHandInsertv2),
                   ('test', TestHandInsertv2),
                   ))),
     ('handle-press-side-v2',
      OrderedDict((('train', TrainButtonPressTopdownWallv2),
                   ('test', TestButtonPressTopdownWallv2),
                   ))),
     ('handle-press-v2',
      OrderedDict((('train', TrainHandlePressv2),
                   ('test', TestHandlePressv2),
                   ))),
     ('handle-pull-side-v2',
      OrderedDict((('train', TrainHandlePullSidev2),
                   ('test', TestHandlePullSidev2),
                   ))),
     ('handle-pull-v2',
      OrderedDict((('train', TrainHandlePullv2),
                   ('test', TestHandlePullv2),
                   ))),
     ('lever-pull-v2',
      OrderedDict((('train', TrainLeverPullv2),
                   ('test', TestLeverPullv2),
                   ))),
     ('peg-insert-side-v2',
      OrderedDict((('train', TrainPegInsertionSidev2),
                   ('test', TestPegInsertionSidev2),
                   ))),
     ('peg-unplug-side-v2',
      OrderedDict((('train', TrainPegUnplugSidev2),
                   ('test', TestPegUnplugSidev2),
                   ))),
     ('pick-place-wall-v2',
      OrderedDict((('train', TrainPickPlaceWallv2),
                   ('test', TestPickPlaceWallv2),
                   ))),
     ('pick-place-v2',
      OrderedDict((('train', TrainPickPlacev2),
                   ('test', TestPickPlacev2),
                   ))),
     ('pick-out-of-hole-v2',
      OrderedDict((('train', TrainPickOutOfHolev2),
                   ('test', TestPickOutOfHolev2),
                   ))),
     ('reach-v2',
      OrderedDict((('train', TrainReachv2),
                   ('test', TestReachv2),
                   ))),
     ('push-back-v2',
      OrderedDict((('train', TrainPushBackv2),
                   ('test', TestPushBackv2),
                   ))),
     ('push-v2',
      OrderedDict((('train', TrainPushv2),
                   ('test', TestPushv2),
                   ))),
     ('plate-slide-v2',
      OrderedDict((('train', TrainPlateSlidev2),
                   ('test', TestPlateSlidev2),
                   ))),
     ('plate-slide-side-v2',
      OrderedDict((('train', TrainPlateSlideSidev2),
                   ('test', TestPlateSlideSidev2),
                   ))),
     ('plate-slide-back-v2',
      OrderedDict((('train', TrainPlateSlideBackv2),
                   ('test', TestPlateSlideBackv2),
                   ))),
     ('plate-slide-back-side-v2',
      OrderedDict((('train', TrainPlateSlideBackSidev2),
                   ('test', TestPlateSlideBackSidev2),
                   ))),
     ('soccer-v2',
      OrderedDict((('train', TrainSoccerv2),
                   ('test', TestSoccerv2),
                   ))),
     ('stick-push-v2',
      OrderedDict((('train', TrainStickPushv2),
                   ('test', TestStickPushv2),
                   ))),
     ('stick-pull-v2',
      OrderedDict((('train', TrainStickPullv2),
                   ('test', TestStickPullv2),
                   ))),
     ('push-wall-v2',
      OrderedDict((('train', TrainPushWallv2),
                   ('test', TestPushWallv2),
                   ))),
     ('push-v2',
      OrderedDict((('train', TrainPushv2),
                   ('test', TestPushv2),
                   ))),
     ('reach-wall-v2',
      OrderedDict((('train', TrainReachWallv2),
                   ('test', TestReachWallv2),
                   ))),
     ('reach-v2',
      OrderedDict((('train', TrainReachv2),
                   ('test', TestReachv2),
                   ))),
     ('shelf-place-v2',
      OrderedDict((('train', TrainShelfPlacev2),
                   ('test', TestShelfPlacev2),
                   ))),
     ('sweep-into-v2',
      OrderedDict((('train', TrainSweepIntoGoalv2),
                   ('test', TestSweepIntoGoalv2),
                   ))),
     ('sweep-v2',
      OrderedDict((('train', TrainSweepv2),
                   ('test', TestSweepv2),
                   ))),
     ('window-open-v2',
      OrderedDict((('train', TrainWindowOpenv2),
                   ('test', TestWindowOpenv2),
                   ))),
     ('window-close-v2',
      OrderedDict((('train', TrainWindowClosev2),
                   ('test', TestWindowClosev2),
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
