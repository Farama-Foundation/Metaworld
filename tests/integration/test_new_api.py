import pytest

from metaworld import ML1, ML10, ML45, MT10, MT50
from tests.helpers import step_env


STEPS = 3


# @pytest.mark.parametrize('env_name', ML1.ENV_NAMES)
# def test_all_ml1(env_name):
#     ml1 = ML1(env_name)
#     train_env_instances = {env_id: env_cls()
#                            for (env_id, env_cls) in ml1.train_classes.items()}
#     for task in ml1.train_tasks:
#         env = train_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in train_env_instances.values():
#         env.close()
#     del train_env_instances

#     test_env_instances = {env_id: env_cls()
#                           for (env_id, env_cls) in ml1.test_classes.items()}
#     for task in ml1.test_tasks:
#         env = test_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in test_env_instances.values():
#         env.close()
#     del test_env_instances


# def test_all_ml10():
#     ml10 = ML10()
#     train_env_instances = {env_id: env_cls()
#                            for (env_id, env_cls) in ml10.train_classes.items()}
#     for task in ml10.train_tasks:
#         env = train_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in train_env_instances.values():
#         env.close()
#     del train_env_instances

#     test_env_instances = {env_id: env_cls()
#                           for (env_id, env_cls) in ml10.test_classes.items()}
#     for task in ml10.test_tasks:
#         env = test_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in test_env_instances.values():
#         env.close()
#     del test_env_instances


# def test_all_ml45():
#     ml45 = ML45()
#     train_env_instances = {env_id: env_cls()
#                            for (env_id, env_cls) in ml45.train_classes.items()}
#     for task in ml45.train_tasks:
#         env = train_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in train_env_instances.values():
#         env.close()
#     del train_env_instances

#     test_env_instances = {env_id: env_cls()
#                           for (env_id, env_cls) in ml45.test_classes.items()}
#     for task in ml45.test_tasks:
#         env = test_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in test_env_instances.values():
#         env.close()
#     del test_env_instances


def test_all_mt10():
    mt10 = MT10()
    train_env_instances = {env_id: env_cls()
                           for (env_id, env_cls) in mt10.train_classes.items()}
    for task in mt10.train_tasks:
        env = train_env_instances[task.env_id]
        env.set_task(task)
        import ipdb; ipdb.set_trace()
        step_env(env, max_path_length=STEPS)
    for env in train_env_instances.values():
        env.close()
    del train_env_instances

    assert len(mt10.test_classes) == 0
    assert len(mt10.test_tasks) == 0


# def test_all_mt50():
#     mt50 = MT50()
#     train_env_instances = {env_id: env_cls()
#                            for (env_id, env_cls) in mt50.train_classes.items()}
#     for task in mt50.train_tasks:
#         env = train_env_instances[task.env_id]
#         env.set_task(task)
#         step_env(env, max_path_length=STEPS)
#     for env in train_env_instances.values():
#         env.close()
#     del train_env_instances

#     assert len(mt50.test_classes) == 0
#     assert len(mt50.test_tasks) == 0
