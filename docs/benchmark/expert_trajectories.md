---
layout: "contents"
title: Expert Trajectories
firstpage:
---

# Expert Trajectories

## Expert Policies
For each individual environment in Meta-World (i.e. ```reach-v3```, ```basketball-v3```, ```sweep-v3```) there are expert policies that solve the task. These policies can be used to generate expert data for imitation learning tasks.

## Using Expert Policies
The below example provides sample code for the reach environment. This code can be extended to the ML10/ML45/MT10/MT50 sets if a list of policies is maintained.


```python
import gymnasium as gym
import metaworld
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

env = gym.make('Meta-World/MT1', env_name='reach-v3')

obs, info = env.reset()

policy = SawyerReachV3Policy()

done = False

while not done:
    a = policy.get_action(obs)
    obs, _, _, _, info = env.step(a)
    done = int(info['success']) == 1
```
