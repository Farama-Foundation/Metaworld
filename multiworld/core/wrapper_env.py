class ProxyEnv(object):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

