from metaworld.policies.sawyer_assembly_v3_policy import SawyerAssemblyV3Policy
from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.policies.sawyer_bin_picking_v3_policy import SawyerBinPickingV3Policy
from metaworld.policies.sawyer_box_close_v3_policy import SawyerBoxCloseV3Policy
from metaworld.policies.sawyer_button_press_topdown_v3_policy import (
    SawyerButtonPressTopdownV3Policy,
)
from metaworld.policies.sawyer_button_press_topdown_wall_v3_policy import (
    SawyerButtonPressTopdownWallV3Policy,
)
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy
from metaworld.policies.sawyer_button_press_wall_v3_policy import (
    SawyerButtonPressWallV3Policy,
)
from metaworld.policies.sawyer_coffee_button_v3_policy import SawyerCoffeeButtonV3Policy
from metaworld.policies.sawyer_coffee_pull_v3_policy import SawyerCoffeePullV3Policy
from metaworld.policies.sawyer_coffee_push_v3_policy import SawyerCoffeePushV3Policy
from metaworld.policies.sawyer_dial_turn_v3_policy import SawyerDialTurnV3Policy
from metaworld.policies.sawyer_disassemble_v3_policy import SawyerDisassembleV3Policy
from metaworld.policies.sawyer_door_close_v3_policy import SawyerDoorCloseV3Policy
from metaworld.policies.sawyer_door_lock_v3_policy import SawyerDoorLockV3Policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy
from metaworld.policies.sawyer_door_unlock_v3_policy import SawyerDoorUnlockV3Policy
from metaworld.policies.sawyer_drawer_close_v3_policy import SawyerDrawerCloseV3Policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
from metaworld.policies.sawyer_faucet_close_v3_policy import SawyerFaucetCloseV3Policy
from metaworld.policies.sawyer_faucet_open_v3_policy import SawyerFaucetOpenV3Policy
from metaworld.policies.sawyer_hammer_v3_policy import SawyerHammerV3Policy
from metaworld.policies.sawyer_hand_insert_v3_policy import SawyerHandInsertV3Policy
from metaworld.policies.sawyer_handle_press_side_v3_policy import (
    SawyerHandlePressSideV3Policy,
)
from metaworld.policies.sawyer_handle_press_v3_policy import SawyerHandlePressV3Policy
from metaworld.policies.sawyer_handle_pull_side_v3_policy import (
    SawyerHandlePullSideV3Policy,
)
from metaworld.policies.sawyer_handle_pull_v3_policy import SawyerHandlePullV3Policy
from metaworld.policies.sawyer_lever_pull_v3_policy import SawyerLeverPullV3Policy
from metaworld.policies.sawyer_peg_insertion_side_v3_policy import (
    SawyerPegInsertionSideV3Policy,
)
from metaworld.policies.sawyer_peg_unplug_side_v3_policy import (
    SawyerPegUnplugSideV3Policy,
)
from metaworld.policies.sawyer_pick_out_of_hole_v3_policy import (
    SawyerPickOutOfHoleV3Policy,
)
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
from metaworld.policies.sawyer_pick_place_wall_v3_policy import (
    SawyerPickPlaceWallV3Policy,
)
from metaworld.policies.sawyer_plate_slide_back_side_v3_policy import (
    SawyerPlateSlideBackSideV3Policy,
)
from metaworld.policies.sawyer_plate_slide_back_v3_policy import (
    SawyerPlateSlideBackV3Policy,
)
from metaworld.policies.sawyer_plate_slide_side_v3_policy import (
    SawyerPlateSlideSideV3Policy,
)
from metaworld.policies.sawyer_plate_slide_v3_policy import SawyerPlateSlideV3Policy
from metaworld.policies.sawyer_push_back_v3_policy import SawyerPushBackV3Policy
from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy
from metaworld.policies.sawyer_push_wall_v3_policy import SawyerPushWallV3Policy
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
from metaworld.policies.sawyer_reach_wall_v3_policy import SawyerReachWallV3Policy
from metaworld.policies.sawyer_shelf_place_v3_policy import SawyerShelfPlaceV3Policy
from metaworld.policies.sawyer_soccer_v3_policy import SawyerSoccerV3Policy
from metaworld.policies.sawyer_stick_pull_v3_policy import SawyerStickPullV3Policy
from metaworld.policies.sawyer_stick_push_v3_policy import SawyerStickPushV3Policy
from metaworld.policies.sawyer_sweep_into_v3_policy import SawyerSweepIntoV3Policy
from metaworld.policies.sawyer_sweep_v3_policy import SawyerSweepV3Policy
from metaworld.policies.sawyer_window_close_v3_policy import SawyerWindowCloseV3Policy
from metaworld.policies.sawyer_window_open_v3_policy import SawyerWindowOpenV3Policy

ENV_POLICY_MAP = dict(
    {
        "assembly-V3": SawyerAssemblyV3Policy,
        "basketball-V3": SawyerBasketballV3Policy,
        "bin-picking-V3": SawyerBinPickingV3Policy,
        "box-close-V3": SawyerBoxCloseV3Policy,
        "button-press-topdown-V3": SawyerButtonPressTopdownV3Policy,
        "button-press-topdown-wall-V3": SawyerButtonPressTopdownWallV3Policy,
        "button-press-V3": SawyerButtonPressV3Policy,
        "button-press-wall-V3": SawyerButtonPressWallV3Policy,
        "coffee-button-V3": SawyerCoffeeButtonV3Policy,
        "coffee-pull-V3": SawyerCoffeePullV3Policy,
        "coffee-push-V3": SawyerCoffeePushV3Policy,
        "dial-turn-V3": SawyerDialTurnV3Policy,
        "disassemble-V3": SawyerDisassembleV3Policy,
        "door-close-V3": SawyerDoorCloseV3Policy,
        "door-lock-V3": SawyerDoorLockV3Policy,
        "door-open-V3": SawyerDoorOpenV3Policy,
        "door-unlock-V3": SawyerDoorUnlockV3Policy,
        "drawer-close-V3": SawyerDrawerCloseV3Policy,
        "drawer-open-V3": SawyerDrawerOpenV3Policy,
        "faucet-close-V3": SawyerFaucetCloseV3Policy,
        "faucet-open-V3": SawyerFaucetOpenV3Policy,
        "hammer-V3": SawyerHammerV3Policy,
        "hand-insert-V3": SawyerHandInsertV3Policy,
        "handle-press-side-V3": SawyerHandlePressSideV3Policy,
        "handle-press-V3": SawyerHandlePressV3Policy,
        "handle-pull-V3": SawyerHandlePullV3Policy,
        "handle-pull-side-V3": SawyerHandlePullSideV3Policy,
        "peg-insert-side-V3": SawyerPegInsertionSideV3Policy,
        "lever-pull-V3": SawyerLeverPullV3Policy,
        "peg-unplug-side-V3": SawyerPegUnplugSideV3Policy,
        "pick-out-of-hole-V3": SawyerPickOutOfHoleV3Policy,
        "pick-place-V3": SawyerPickPlaceV3Policy,
        "pick-place-wall-V3": SawyerPickPlaceWallV3Policy,
        "plate-slide-back-side-V3": SawyerPlateSlideBackSideV3Policy,
        "plate-slide-back-V3": SawyerPlateSlideBackV3Policy,
        "plate-slide-side-V3": SawyerPlateSlideSideV3Policy,
        "plate-slide-V3": SawyerPlateSlideV3Policy,
        "reach-V3": SawyerReachV3Policy,
        "reach-wall-V3": SawyerReachWallV3Policy,
        "push-back-V3": SawyerPushBackV3Policy,
        "push-V3": SawyerPushV3Policy,
        "push-wall-V3": SawyerPushWallV3Policy,
        "shelf-place-V3": SawyerShelfPlaceV3Policy,
        "soccer-V3": SawyerSoccerV3Policy,
        "stick-pull-V3": SawyerStickPullV3Policy,
        "stick-push-V3": SawyerStickPushV3Policy,
        "sweep-into-V3": SawyerSweepIntoV3Policy,
        "sweep-V3": SawyerSweepV3Policy,
        "window-close-V3": SawyerWindowCloseV3Policy,
        "window-open-V3": SawyerWindowOpenV3Policy,
    }
)

__all__ = [
    "SawyerAssemblyV3Policy",
    "SawyerBasketballV3Policy",
    "SawyerBinPickingV3Policy",
    "SawyerBoxCloseV3Policy",
    "SawyerButtonPressTopdownV3Policy",
    "SawyerButtonPressTopdownWallV3Policy",
    "SawyerButtonPressV3Policy",
    "SawyerButtonPressWallV3Policy",
    "SawyerCoffeeButtonV3Policy",
    "SawyerCoffeePullV3Policy",
    "SawyerCoffeePushV3Policy",
    "SawyerDialTurnV3Policy",
    "SawyerDisassembleV3Policy",
    "SawyerDoorCloseV3Policy",
    "SawyerDoorLockV3Policy",
    "SawyerDoorOpenV3Policy",
    "SawyerDoorUnlockV3Policy",
    "SawyerDrawerCloseV3Policy",
    "SawyerDrawerOpenV3Policy",
    "SawyerFaucetCloseV3Policy",
    "SawyerFaucetOpenV3Policy",
    "SawyerHammerV3Policy",
    "SawyerHandInsertV3Policy",
    "SawyerHandlePressSideV3Policy",
    "SawyerHandlePressV3Policy",
    "SawyerHandlePullSideV3Policy",
    "SawyerHandlePullV3Policy",
    "SawyerLeverPullV3Policy",
    "SawyerPegInsertionSideV3Policy",
    "SawyerPegUnplugSideV3Policy",
    "SawyerPickOutOfHoleV3Policy",
    "SawyerPickPlaceV3Policy",
    "SawyerPickPlaceWallV3Policy",
    "SawyerPlateSlideBackSideV3Policy",
    "SawyerPlateSlideBackV3Policy",
    "SawyerPlateSlideSideV3Policy",
    "SawyerPlateSlideV3Policy",
    "SawyerPushBackV3Policy",
    "SawyerPushV3Policy",
    "SawyerPushWallV3Policy",
    "SawyerReachV3Policy",
    "SawyerReachWallV3Policy",
    "SawyerShelfPlaceV3Policy",
    "SawyerSoccerV3Policy",
    "SawyerStickPullV3Policy",
    "SawyerStickPushV3Policy",
    "SawyerSweepIntoV3Policy",
    "SawyerSweepV3Policy",
    "SawyerWindowOpenV3Policy",
    "SawyerWindowCloseV3Policy",
    "ENV_POLICY_MAP",
]
