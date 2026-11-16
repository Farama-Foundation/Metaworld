"""Optional multiprocessing configuration for resource-constrained CI runs."""

import multiprocessing
import os


def pytest_configure():
    """Allow CI to retain its historical Linux worker context explicitly."""
    start_method = os.environ.get("METAWORLD_TEST_START_METHOD")
    if start_method:
        multiprocessing.set_start_method(start_method, force=True)
