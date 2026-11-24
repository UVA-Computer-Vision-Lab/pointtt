import torch.nn as nn
import spconv.pytorch as spconv


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """


class PointSequential(PointModule):
    r"""A sequential container for modules that act on a Point.
    Modules will be added to it in the order they are passed in the constructor
    (keyword arguments give named modules).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                input.sparse_conv_feat = module(input.sparse_conv_feat)
                input.feat = input.sparse_conv_feat.features
            # PyTorch module
            else:
                input.feat = module(input.feat)
                if "sparse_conv_feat" in input.keys():
                    input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                        input.feat
                    )
        return input
