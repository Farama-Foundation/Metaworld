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
p = policy()
count = 0
done = False
while not done and count < 500:
    #print(p.parse_obs(obs))
    a = p.get_action(obs)
    obs, reward, done, info = env.step(a)
    print(env.render())
    count += 1
    if done:
        break
print(obs, reward, done, info)