---
layout: "contents"
title: Reward Functions 
firstpage:
---

# Reward Functions

Metaworld currently implements two types of reward functions that can be selected
by passing the `reward_func_version` keyword argument to the `gym.make(...)` call.

Supported are currently two versions.

## Version 1

Passing `reward_func_version=v1` configures the benchmark with the primary
reward function of Metaworld, which is actually a version of the
`pick-place-wall` task that is modified to also work for the other tasks.


## Version 2

TBA
