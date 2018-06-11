from multiworld.core.serializable import Serializable


class ProxyEnv(Serializable):
    def __init__(self, wrapped_env):
        self.quick_init(locals())
        self._wrapped_env = wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def __getattr__(self, attrname):
        if attrname == '_serializable_initialized':
            return None
        return getattr(self._wrapped_env, attrname)

