def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                "You must call env.set_task before using env." + func.__name__
            )
        return func(*args, **kwargs)

    return inner
