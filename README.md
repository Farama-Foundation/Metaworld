# multiworld
Multitask Environments for RL

## Basic Usage
This library contains a variety of `MultitaskEnv`s.
`MultitaskEnv`s can be seen as regular gym environments.
However, they're designed with multitasks in mind.
As a running example, let's say we have a `CarEnv`.
Like normal gym envs, we can do
```
env = CarEnv(fixed_goal=True)
obs = env.reset()
next_obs, reward, done, info = env.step(action)
```
One difference from usual gym envs however is that the reward function depends
on the goal.
Furthermore, *the goal is not appended to the observation*.
The goal can be retrieved with
```
goal = env.get_goal()
```
We passed in `fixed_goal=True` to the constructor, meaning that the goal is
always the same even if we call reset.
We can change this by changing `fixed_goal`:
```
env = CarEnv(fixed_goal=False)
env.reset()
goal1 = env.get_goal()
env.reset()
goal2 = env.get_goal()
assert goal1 != goal2
```

However, remember that the goal is not part of the observation.
Hence, you'll probably want to give the goal to the policy as well:
```
env = CarEnv(fixed_goal=False)
obs = env.reset()
goal = env.get_goal()
action = policy(obs, goal)
```
This can be annoying to integrate with existing RL code, so as a helper, we have
`MultitaskEnvToFlatEnv` which tasks care of appending the goal to the
observation.
```
env = CarEnv(fixed_goal=False)
env = MultitaskToFlatEnv(env)
obs = env.reset()  # goal is concatenated to the observation
action = policy(obs)
```

## Wrappers, Extensions, and Info dict
Another big difference from gym envs is that we make liberal use of the `info`
dictionary returned by step.
```
env = CarEnv(fixed_goal=True)
obs = env.reset()
next_obs, reward, done, info = env.step(action)
print(info)

# {
#     'observation': ...,
#     'desired_goal': ...,
#     'achieved_goal': ...,
# }
```
As you see, the info dictionary now has the keys `observation`, `desired_goal`,
and `achieved_goal`. This is
The reason we do this is to make the class easily extendable.

Because the `info` dict is so useful, all the environments also implement the
`get_info` method, so you can do
```
env = CarEnv(fixed_goal=False)
first_obs = env.reset()
info = env.get_info()  # returns the info dict associated with first_obs
```
