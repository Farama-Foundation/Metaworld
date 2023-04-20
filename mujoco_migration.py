import numpy as np
import metaworld
import random

from metaworld.policies.sawyer_assembly_v2_policy import SawyerAssemblyV2Policy as policy
seed = 42
env_name = "assembly-v2"

random.seed(seed)

ml1 = metaworld.MT50(seed=seed)
env = ml1.train_classes[env_name]()
task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
env.set_task(task)

env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

obs = env.reset()
print(env.action_space)
action = -np.ones(4)*0.05
while True:
    env.render()
    env.step(env.action_space.sample())