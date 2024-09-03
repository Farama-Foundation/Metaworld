---
layout: "contents"
title: Benchmark Descriptions
firstpage:
---

# Benchmark Descriptions

The benchmark provides a selection of tasks used to study generalization in reinforcement learning (RL).
Different combinations of tasks provide benchmark scenarios suitable for multi-task RL and meta-RL.
Unlike usual RL benchmarks, the training of the agent is strictly split into a training and testing phase.

## Multi-Task Problems

The multi-task setting challenges the agent to learn a predefined set of skills simultaneously.
Below, different levels of difficulty are described.

### Multi-Task (MT1)

In the easiest setting, **MT1**, a single task needs to be learned where the agent must *reach*, *push*, or *pick and place* a goal object.
There is no testing of generalization involved in this setting.

```{figure} _static/mt1.gif
   :alt: Multi-Task 1 
   :width: 500
```

### Multi-Task (MT10)

The **MT10** setting involves learning to solve a diverse set of 10 tasks, as depicted below.
There is no testing of generalization involved in this setting.



```{figure} _static/mt10.gif
   :alt: Multi-Task 10 
   :width: 500
```

### Multi-Task (MT50)

In the **MT50** setting, the agent is challenged to solve the full suite of 50 tasks contained in metaworld.
This is the most challenging multi-task setting and involves no evaluation on test tasks.


## Meta-Learning Problems

Meta-RL attempts to evaluate the [transfer learning](https://en.
wikipedia.org/wiki/Transfer_learning) capabilities of agents learning skills based on a predefined set of training tasks, by evaluating generalization using a hold-out set of test tasks.
In other words, this setting allows for benchmarking an algorithm's ability to adapt to or learn new tasks.

### Meta-RL (ML1)

The simplest meta-RL setting, **ML1**, involves a single manipulation task, such as *pick and place* of an object with a changing goal location.
For the test evaluation, unseen goal locations are used to measure generalization capabilities.



```{figure} _static/ml1.gif
   :alt: Meta-RL 1 
   :width: 500
```


### Meta-RL (ML10)

The meta-learning setting with 10 tasks, **ML10**, involves training on 10 manipulation tasks and evaluating on 5 unseen tasks during the test phase.

```{figure} _static/ml10.gif
   :alt: Meta-RL 10 
   :width: 500
```

### Meta-RL (ML45)

The most difficult environment setting of metaworld, **ML45**, challenges the agent to be trained on 45 distinct manipulation tasks and evaluated on 5 test tasks.


```{figure} _static/ml45.gif
   :alt: Meta-RL 10 
   :width: 500
```
