
class Benchmark:

    @classmethod
    def get_train_tasks(cls, sample_all=False):
        return cls(env_type='train', sample_all=sample_all)
    
    @classmethod
    def get_test_tasks(cls, sample_all=False):
        return cls(env_type='test', sample_all=sample_all)

    @classmethod
    def from_task(cls, task_name):
        """Construct a Benchmark instance with one task.

        Args:
            cls (metaworld.benchmarks.Benchmark): Class of the instance.
            task_name (str): Task name. Subclasses should check if the
                task exists, raise ValueError if not.

        Returns:
            metaworld.benchmarks.Benchmark: Instance that contains
                the required task.

        """
        raise NotImplementedError