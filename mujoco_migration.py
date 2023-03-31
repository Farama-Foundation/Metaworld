from metaworld.policies.sawyer_assembly_v2_policy import SawyerAssemblyV2Policy as policy

import metaworld
import random
import mujoco
import time
seed = 42
env_name = "assembly-v2"

random.seed(seed)

ml1 = metaworld.MT50(seed=seed)
env = ml1.train_classes[env_name]()
task = random.choice([t for t in ml1.train_tasks if t.env_name == env_name])
env.set_task(task)

env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)


obs = env.reset()
p = policy()
print(obs)
env.render()
for _ in range(500):
    a = p.get_action(obs)
    obs, reward, done, info = env.step(a)
    env.render()
    time.sleep(0.02)

print(obs, reward, done, info)