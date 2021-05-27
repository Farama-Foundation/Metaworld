from typing import Set

import numpy as np

from .heuristics import (
    AlongBackWall,
    Ceiling,
    GreaterThanXValue,
    InFrontOf,
    LessThanXValue,
    LessThanYValue,
    OnTopOf,
)
from .tools import (
    Basketball,
    BasketballHoop,
    BinA,
    BinB,
    BinLid,
    ButtonBox,
    CoffeeMachine,
    CoffeeMug,
    Dial,
    Door,
    Drawer,
    ElectricalPlug,
    ElectricalOutlet,
    FaucetBase,
    HammerBody,
    Lever,
    NailBox,
    Puck,
    ScrewEye,
    ScrewEyePeg,
    Shelf,
    SoccerGoal,
    Thermos,
    ToasterHandle,
    Window,
)
from .solver import Solver
from .voxelspace import VoxelSpace


def standard(solver: Solver, world: VoxelSpace, toolset: Set[str]) -> bool:
    def construct(cls):
        return cls(enabled=cls.__name__ in toolset)

    basketball = construct(Basketball)
    hoop = construct(BasketballHoop)
    bin_a = construct(BinA)
    bin_b = construct(BinB)
    bin_lid = construct(BinLid)
    button = construct(ButtonBox)
    coffee_machine = construct(CoffeeMachine)
    coffee_mug = construct(CoffeeMug)
    dial = construct(Dial)
    door = construct(Door)
    drawer = construct(Drawer)
    plug = construct(ElectricalPlug)
    outlet = construct(ElectricalOutlet)
    faucet = construct(FaucetBase)
    hammer = construct(HammerBody)
    lever = construct(Lever)
    nail = construct(NailBox)
    puck = construct(Puck)
    screw_eye = construct(ScrewEye)
    screw_eye_peg = construct(ScrewEyePeg)
    shelf = construct(Shelf)
    soccer_goal = construct(SoccerGoal)
    thermos = construct(Thermos)
    toaster = construct(ToasterHandle)
    window = construct(Window)

    # Place large artifacts along the back of the table
    back_lineup = [door, nail, hoop, window, coffee_machine, drawer]
    # Place certain artifacts on top of one another to save space
    stacked = (
        (coffee_machine, toaster),
        (drawer, shelf),
        (nail, outlet),
        (door, button),
    )
    # If the lowermost item in the stack is disabled, place the upper item
    # along the back of the table instead
    back_lineup += [upper for lower, upper in stacked if not lower.enabled]

    solver.apply(AlongBackWall(0.95 * world.size[1]), back_lineup, tries=2)
    for lower, upper in stacked:
        if lower.enabled:
            solver.apply(OnTopOf(lower), [upper])

    # Put the faucet under the basketball hoop
    if faucet.enabled and hoop.enabled:
        faucet.specified_pos = hoop.specified_pos + np.array([.0, -.1, .0])
        faucet.specified_pos[2] = faucet.resting_pos_z * world.resolution
        solver.did_manual_set(faucet)

    # The ceiling means that taller objects get placed along the edges
    # of the table. We place them first (note that `shuffle=False` so list
    # order matters) so that the shorter objects don't take up space along
    # the edges until tall objects have had a chance to fill that area.
    def ceiling(i, j):
        # A bowl-shaped ceiling, centered at the Sawyer
        i -= world.mat.shape[0] // 2
        return (0.02 * i * i) + (0.005 * j * j) + 20

    # Place certain objects under a ceiling
    under_ceiling = [thermos, lever, screw_eye_peg, dial]
    # Place certain artifacts in front of one another to simplify config
    layered = (
        (window, soccer_goal),
        (nail, bin_a),
        (bin_a, bin_b),
    )
    # If rearmost item is disabled, place other item under ceiling instead
    under_ceiling += [fore for aft, fore in layered if not aft.enabled]
    # If faucet wasn't placed earlier, put it under the ceiling as well
    if faucet.enabled and not hoop.enabled:
        under_ceiling.append(faucet)

    solver.apply(Ceiling(ceiling), under_ceiling, tries=20, shuffle=False)
    for aft, fore in layered:
        if aft.enabled:
            solver.apply(InFrontOf(aft), [fore])

    # At this point we only have a few objects left to place. They're all
    # tools (not artifacts) which means they can move around freely. As
    # such, they can ignore the immense bounding boxes of the the door and
    # drawer (NOTE: in the future, it may make sense to have separate
    # `bounding` and `clearance` boxes so that freejointed tools can
    # automatically ignore clearance boxes). For now, to ignore the
    # existing bounding boxes, we manually reset their voxels:
    if door.enabled:
        world.fill_tool(door, value=False)
    if drawer.enabled:
        world.fill_tool(drawer, value=False)

    edge_buffer = (world.size[0] - 1.0) / 2.0
    free_of_overlaps = solver.apply([
        # Must have this Y constraint to avoid placing inside the door
        # and drawer whose voxels got erased
        LessThanYValue(0.6 * world.size[1]),
        GreaterThanXValue(edge_buffer),
        LessThanXValue(world.size[0] - edge_buffer)
    ], [
        hammer, bin_lid, screw_eye, coffee_mug, puck, basketball, plug
    ], tries=100, shuffle=False)

    return free_of_overlaps
