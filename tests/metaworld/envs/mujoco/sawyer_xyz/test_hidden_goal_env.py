import pytest
from tests.helpers import step_env
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN


def test_hidden_goal_envs():
    
    for env_key, env_cls in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.items():       
        
        assert "goal-hidden" in env_key
        assert "GoalHidden" in env_cls.__name__

        env = env_cls()
        step_env(env, max_path_length=150, iterations=3, render=False)
        
        first_target = env._target_pos
        env.reset()
        second_target = env._target_pos

        assert (first_target == second_target).all()
