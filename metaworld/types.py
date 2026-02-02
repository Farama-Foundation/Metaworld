from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import NotRequired, TypeAlias, TypedDict


class TaskSet(NamedTuple):
    """
    A collection of tasks.
    """

    tasks_dict: dict[str, list[Task]]
    """
    Mapping from environment name to list of tasks.
    """
    env_names: list[str]
    """
    List of all environment names in the task set.
    """


class Task(NamedTuple):
    """
    All data necessary to fully describe a single environment.
    """

    env_name: str
    env_seed: int


XYZ: TypeAlias = "Tuple[float, float, float]"
"""A 3D coordinate."""


class EnvironmentStateDict(TypedDict):
    state: dict[str, Any]
    mjb: str
    mocap: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


class ObservationDict(TypedDict):
    state_observation: npt.NDArray[np.float64]
    state_desired_goal: npt.NDArray[np.float64]
    state_achieved_goal: npt.NDArray[np.float64]


class InitConfigDict(TypedDict):
    obj_init_angle: NotRequired[float]
    obj_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]


class HammerInitConfigDict(TypedDict):
    hammer_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]


class StickInitConfigDict(TypedDict):
    stick_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]
