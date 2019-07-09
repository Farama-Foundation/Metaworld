import abc
import pprint

import gym

from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from metaworld.world import ParametricWorld


class TaskBased(SawyerXYZEnv, ParametricWorld, metaclass=abc.ABCMeta):
    """An environment whose task can be described using a dictionary.

    Attributes:
        task_schema(:obj:`gym.Space`): A space representing valid tasks
            in this environment
    """

    @property
    def task(self):
        """The task with which this environment calculates rewards.

        Setting this property immediately changes the rewards calculated by the
        environment.

        Notes:
            If you require users `reset()` the environment after setting this
            property, you should document this explicitly.
        """
        return self._task

    @task.setter
    def task(self, t):
        self._validate_task(t)
        self._task = t

    @abc.abstractstaticmethod
    def goal_from_task(task):
        """Derives a state goal from a task descriptor.

        Args:
            task(object): A task conforming to `task_schema`

        Returns:
            np.ndarray: A state-space goal compatible with rlkit-based
                algorithms.
        """

        pass

    def validate_task(self, task):
        """
        Verify that a task conforms to this World's task schema.

        Args:
            task(object): A task which may or may not conform to task_schema

        Raises:
            ValueError: If `task` does not conform to `self.task_schema`
        """
        if not self.task_schema.contains(task):
            raise ValueError(
                'Task must be in the space \n{}, but provided \n{}'.format(
                    pprint.pformat(self.task_schema.spaces),
                    pprint.pformat(task)))
