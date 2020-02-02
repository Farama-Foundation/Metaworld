from metaworld.benchmarks import ML45

class ML45WithPinnedGoal(ML45):
    """A wrapper of ML1 environment that retains goals across pickling and unpickling.

    `env = ML1.get_train_tasks('task-name')` gives an environment that internally keeps 50 pre-generated variants of
    this task, and `env.sample_task(1)` will return one of these variants. However, these variants cannot survive
    pickling. That is, `pickle.loads(pickle.dumps(env))` will give an environment with a new set of variants, which is
    not desired when doing vectorized and parallel sampling. This wrapper solves this caveat by saving and restoring
    the parameter of a variant, i.e. the goal of the task, explicitly.

    See discussion at https://github.com/rlworkgroup/metaworld/issues/24#issuecomment-576996005

    """

    def __getstate__(self):
        state = super().__getstate__()
        state['goals'] = self._discrete_goals
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.discretize_goal_space(state.get('goals', {}))
