from abc import ABC
import json
import os
from time import time
from copy import deepcopy
from pprint import pformat

import torch
import torch.nn as nn
import torch.nn.functional as F

from librubiks import gpu, envs
from pelutils import log


class ModelConfig:

    activation_functions = {
        "elu":   nn.ELU(),
        "relu":  nn.ReLU(),
        "relu6": nn.ReLU6(),
        "lrelu": nn.LeakyReLU(),
    }

    def __init__(
        self,
        env_key: str,
        layer_sizes: list   = [4096, 2048, 512],
        activation_function = "elu",
        batchnorm: bool     = True,
        dropout: float      = 0,
        architecture: str   = "fc",
        init: str           = "glorot",
        id: int             = None
    ):
        assert isinstance(layer_sizes, list),\
            f"Layers must be a list of integer sizes, not {type(layer_sizes)}"
        assert activation_function in self.activation_functions.keys(),\
            "Activation function must be one of " + ", ".join(self.activation_functions.keys()) + f", not {activation_function}"
        assert isinstance(batchnorm, bool),\
            f"Batchnorm must a boolean, not {batchnorm}"
        assert 0 <= dropout <= 1,\
            f"Dropout must be in [0, 1], not {dropout}"
        assert architecture in ["fc", "res"],\
            f"Architecture must be fc (fully-connected) or res (fc with residual blocks), not {architecture}"
        assert init == "glorot" or init == "he" or type(init) == int or type(init) == float,\
            f"Init must be glorot, he, or a number, not {init}"

        self.env_key = env_key
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.batchnorm = batchnorm
        self.architecture = architecture
        self.dropout = dropout
        self.init = init
        self.id = id or hash(time())

    def get_activfunc(self):
        return self.activation_functions[self.activation_function]

    def as_json_dict(self):
        return deepcopy(self.__dict__)

    @classmethod
    def from_json_dict(cls, conf: dict):
        return ModelConfig(**conf)

    def __str__(self):
        return pformat(self.__dict__)


class Model(nn.Module, ABC):
    """
    A fully connected, feed forward Neural Network.
    Also the instantiator class of other network architectures through `create`.
    """

    layers: nn.Sequential

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.env = envs.get_env(self.config.env_key)

        self._construct_net()
        self.log(f"Created network\n{self.config}\n{self}")

    def _construct_net(self):
        """Constructs self.layers based on self.config"""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def _create_fc_layers(self, thiccness: list):
        """
        Helper function to return fully connected feed forward layers given a list of layer sizes and a final output size.
        """
        layers = []
        for i in range(len(thiccness)-1):
            l = nn.Linear(thiccness[i], thiccness[i+1])
            if self.config.init == 'glorot':
                torch.nn.init.xavier_uniform_(l.weight)
            elif self.config.init == 'he':
                torch.nn.init.kaiming_uniform_(l.weight)
            else:
                torch.nn.init.constant_(l.weight, float(self.config.init))
            layers.append(l)

            if i != len(thiccness) - 2:
                layers.append(self.config.get_activfunc())
                if self.config.batchnorm:
                    layers.append(nn.BatchNorm1d(thiccness[i+1]))
                if self.config.dropout > 0:
                    layers.append(nn.Dropout(p=self.config.dropout))

        return layers

    def clone(self) -> nn.Module:
        new_state_dict = {}
        for kw, v in self.state_dict().items():
            new_state_dict[kw] = v.clone()
        new_net = create_net(self.config)
        new_net.load_state_dict(new_state_dict)
        return new_net

    def get_params(self) -> torch.Tensor:
        return torch.cat([x.float().flatten() for x in self.state_dict().values()]).clone()

    def __len__(self):
        return sum(x.numel() for x in self.state_dict().values())


class _FullyConnected(Model):
    """
    A fully-connected network with no fancyness
    """
    def _construct_net(self):
        layer_sizes = [self.env.oh_size, *self.config.layer_sizes, 1]
        layers = self._create_fc_layers(layer_sizes)
        self.layers = nn.Sequential(*layers)


class _ResBlock(nn.Module):
    """
    A residual block of two linear layers with the same size.
    """
    def __init__(self, layer_size: int, activation: nn.Module, with_batchnorm: bool):
        super().__init__()
        self.layer1, self.layer2 = nn.Linear(layer_size, layer_size), nn.Linear(layer_size, layer_size)
        self.activate = activation
        self.with_batchnorm = with_batchnorm
        if self.with_batchnorm:
            # Uses two batchnorms as PyTorch trains some running momentum parameters for each bnorm
            self.batchnorm1 = nn.BatchNorm1d(layer_size)
            self.batchnorm2 = nn.BatchNorm1d(layer_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Layer 1
        x = self.layer1(x)
        if self.with_batchnorm: x = self.batchnorm1(x)
        x = self.activate(x)
        # Layer 2
        x = self.layer2(x)
        if self.with_batchnorm: x = self.batchnorm2(x)
        # Residual added
        x += residual
        x = self.activate(x)
        return x


class _ResNet(Model):
    """
    A fully-connected network with residual blocks
    """
    def _construct_net(self):
        # FIXME: This is currently broken
        # TODO: Better implementation of res blocks, so they can have different sizes
        for i in range(self.config.res_blocks):
            resblock = _ResBlock(self.config.res_size, self.config.activation_function, self.config.batchnorm)
            self.shared_net.add_module(f'resblock{i}', resblock)


def create_net(config: ModelConfig) -> Model:
    """
    Allows this class to be used to instantiate other Network architectures based on the content
    of the configuartion file.
    """
    if config.architecture == "fc":
        return _FullyConnected(config).to(gpu)
    if config.architecture == "res":
        return _ResNet(config).to(gpu)
    raise KeyError(f"Network architecture should be 'fc' or 'res', but '{config.architecture}' was given")


def save_net(net: Model, save_dir: str, is_best = False):
    """
    Save the model and configuration to the given directory
    The folder will include a pytorch model, and a json configuration file
    """

    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        model_path = os.path.join(save_dir, "model-best.pt")
        torch.save(net.state_dict(), model_path)
        log(f"Saved best model to {model_path}")
        return
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(net.state_dict(), model_path)
    conf_path = os.path.join(save_dir, "config.json")
    with open(conf_path, "w", encoding="utf-8") as conf:
        json.dump(net.config.as_json_dict(), conf, indent=4)
    log(f"Saved model to {model_path} and configuration to {conf_path}")


def load_net(load_dir: str, load_best = False) -> Model:
    """Load a model from a configuration directory"""

    model_path = os.path.join(load_dir, "model.pt" if not load_best else "model-best.pt")
    conf_path = os.path.join(load_dir, "config.json")
    with open(conf_path, encoding="utf-8") as conf:
        try:
            state_dict = torch.load(model_path, map_location=gpu)
        except FileNotFoundError:
            model_path = os.path.join(load_dir, "model.pt")
            state_dict = torch.load(model_path, map_location=gpu)
        config = ModelConfig.from_json_dict(json.load(conf))

    model = create_net(config)
    model.load_state_dict(state_dict)
    model.to(gpu)
    # First time the net is loaded, a feedforward is performed, as the first time is slow
    # This avoids skewing evaluation results
    env = envs.get_env(config.env_key)
    with torch.no_grad():
        model.eval()
        model(env.as_oh(env.get_solved()))
        model.train()
    return model

