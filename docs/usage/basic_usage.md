---
layout: "contents"
title: Generate data with expert policies
firstpage:
---

# Generate data with expert policies

## Expert Policies
For each individual environment in Meta-World (i.e. reach, basketball, sweep) there are expert policies that solve the task. These policies can be used to generate expert data for imitation learning tasks.

## Using Expert Policies
The below example provides sample code for the reach environment. This code can be extended to the ML10/ML45/MT10/MT50 sets if a list of policies is maintained.


```python
from metaworld import MT1

from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy as p

mt1 = MT1('reach-v2', seed=42)
env = mt1.train_classes['reach-v2']()
env.set_task(mt1.train_tasks[0])
obs, info = env.reset()

policy = p()

done = False

while not done:
    a = policy.get_action(obs)
    obs, _, _, _, info = env.step(a)
    done = int(info['success']) == 1


```
