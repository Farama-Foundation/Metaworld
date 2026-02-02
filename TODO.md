## DONEs with reason
* Renamed `max_path_length` to `max_episode_steps` in several places to align with Gymnasium's terminology.
* * Removed `Meta-World/goal_hidden` and `Meta-World/goal_observable` as one can just use `Meta-World/MT1` with the appropriate `goal_observable` kwarg.

## TODOs

* Scripts
* tests
* Docs
* Evaluation
* Add test for one hot
* Add test for all benchmarks
* Introduce wrapper for goal observability instead of handling it downstream!
