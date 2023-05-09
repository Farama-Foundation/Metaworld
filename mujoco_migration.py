import metaworld
env_name = 'hammer-v2'
import random
from gymnasium.wrappers import TimeLimit
seed = 42
random.seed(seed)

ml1 = metaworld.MT50(seed=seed)
env = ml1.train_classes[env_name]()
env.set_task([t for t in ml1.train_tasks if t.env_name == env_name][0])
env = TimeLimit(env, max_episode_steps=500)

env.reset()

for x in range(500):
    print(x)
    a = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(a)
    env.render()
    print(terminated, truncated, info)
    if terminated or truncated:
        print('env should reset ... ')
        env.reset()

env.close()
