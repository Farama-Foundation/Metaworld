from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_basketball_v2 import SawyerBasketballEnvV2 as env
from metaworld.policies.sawyer_basketball_v2_policy import SawyerBasketballV2Policy as policy
from metaworld import MT50
mt50 = MT50()
print(mt50.train_classes)
tasks = [task for task in mt50.train_tasks if task.env_name == 'basketball_v2']

