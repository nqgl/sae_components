class CachedSequential(nn.Module):  # Is this a cachelayer maybe? not certain best name
    def __init__(self, *layers):
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cache):
        cache.x = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CacheModule):
                x = layer(x, cache=cache[i])
            else:
                x = layer(x)
        cache.y_pred = x
        return x


class NameRef:
    def __init__(self, obj, write_enabled=True):
        self.__obj = obj
        self.__write_enabled = write_enabled

    def __getattr__(self, __name):
        if __name in ["__obj", "__write_enabled"]:
            return super().__getattr__(__name)

        @property
        def prop():
            return getattr(self.__obj, __name)

        if self.__write_enabled:

            @prop.setter
            def prop(x):
                setattr(self.__obj, __name, x)

        return prop
