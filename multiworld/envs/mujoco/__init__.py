from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld mujoco gym environments")
    # TODO
    # register(
    #     id='TODO',
    #     entry_point='multiworld.envs.mujoco.todo:TODO',
    #     tag={
    #         'git-commit-hash': 'TODO',
    #         'author': 'TODO'
    #     },
    #     kwargs={
    #
    #     },
    # )


register_custom_envs()
