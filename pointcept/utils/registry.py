# Copyright (c) OpenMMLab. All rights reserved.
# Modified for PoinTTT: reduced to what this repository uses (no scopes, parent registries,
# default arguments, forced re-registration or the deprecated registration API).
import inspect


def build_from_cfg(cfg, registry):
    """Build a module from configs dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f'`cfg` must contain the key "type", but got {cfg}')

    args = cfg.copy()
    obj_type = args.pop("type")
    obj_cls = registry.get(obj_type)
    if obj_cls is None:
        raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A registry to map strings to classes.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class, or None.
        """
        return self._module_dict.get(key)

    def build(self, cfg):
        return build_from_cfg(cfg, self)

    def _register_module(self, module_class, module_name=None):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")
        if module_name is None:
            module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered " f"in {self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(module=ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            module (type): Module class to be registered.
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name must be None or a str, but got {type(name)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name)
            return cls

        return _register
