import numpy as np
import metaworld
import random
from metaworld.policies.sawyer_basketball_v2_policy import SawyerBasketballV2Policy as policy
from PIL import Image
import time
seed = 42
env_name = "basketball-v2"

random.seed(seed)
ml1 = metaworld.MT50(seed=seed)
env = ml1.train_classes[env_name]()
task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
env.set_task(task)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
obs = env.reset()
print(obs)
print("here")
print(env.data.qpos)
print(env.data.qvel)
'''
env.data.qpos = [ 0.00000000e+00,  6.00000000e-01, 2.98721632e-02, 1.00000000e+00,
  0.00000000e+00 , 0.00000000e+00,  0.00000000e+00,  1.88261575e+00,
 -5.93460002e-01, -9.59474274e-01,  1.64832847e+00,  9.23299010e-01,
  1.02688965e+00,  2.32340576e+00, -1.71909125e-04,  1.71744846e-04]

env.data.qvel = [ 0.00000000e+00,  0.00000000e+00, 0.0,  0.00000000e+00,
  0.0000000e+00,  0.00000000e+00,  0.0,  5.62470645e-01,
 -6.20149668e-01,  1.94811348e-01,  9.38798805e-01,  2.09966159e-01,
  2.09523279e+00,  1.02772565e-06,  1.02595610e-06]
Wenv.set_state(env.data.qpos, env.data.qvel)'''
print(env.data.qpos)
print(env.data.qvel)

p = policy()
count = 0
done = False
while count < 500 and not done:
    env.render()
    print(count)
    #img = Image.fromarray(env.render(offscreen=True))
    #img.save(f"frame_{count}.png")
    #print(p._parse_obs(obs))
    next_obs, _, _, info = env.step(p.get_action(obs))
    if int(info['success']) == 1:
        done = True
    obs = next_obs
    print(info)
    time.sleep(0.02)
    count += 1


print(info)