from typing import List, Optional

from torch import nn
import torch.nn.functional as F
from .configs import ForwardConfig, check_backward_or_forward, FeatureModel


class BDLinear(nn.Linear):
    """
    Bidirectional Linear Layer - modifies the original nn.Linear class
    """

    def forward(self, x, zero_bias=False, config: ForwardConfig = ForwardConfig.FORWARD):
        if not check_backward_or_forward(config):
            return x

        active_bias = self.bias
        active_weight = self.weight
        if zero_bias:
            active_bias = None

        if config == ForwardConfig.BACKWARD:
            # transpose the weight and set the bias to None if needed.
            active_weight = active_weight.permute(1, 0)
            if not zero_bias:
                x += active_bias
                active_bias = None
        return F.linear(x, active_weight, active_bias)


class BDNN(FeatureModel):
    """
    Class for defining a Bidirectional Neural Network.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 forward_only: bool,
                 hidden_layer_sizes: Optional[List[int]] = None):
        # in features usually 192
        super().__init__(forward_only, out_features)

        self.in_features = in_features

        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
        self.hidden_layer_sizes = hidden_layer_sizes

        self.size_list = [in_features] + hidden_layer_sizes + [out_features]
        self.lr = nn.LeakyReLU()
        self.layers = []
        for i in range(len(self.size_list) - 1):
            self.layers.append(BDLinear(self.size_list[i], self.size_list[i + 1]))
        self.layers: nn.ModuleList[BDLinear] = nn.ModuleList(self.layers)

    def forward(self, x, config: ForwardConfig = ForwardConfig.FORWARD):
        if not check_backward_or_forward(config):
            return x

        iter_layers = self.layers
        if config == ForwardConfig.BACKWARD:
            iter_layers = reversed(iter_layers)

        for i, layer in enumerate(iter_layers):
            kwargs = {"config": config}

            if i == 0 and not self.forward_only:
                kwargs["zero_bias"] = True

            x = layer(x, **kwargs)

            if i < len(self.layers) - 1:
                x = self.lr(x)

        return x
