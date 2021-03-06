import json
import os
from time import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from librubiks import cube
from librubiks import gpu
from librubiks.utils import NullLogger


class ModelConfig:

	_fc_small_arch  = { "shared_sizes": [4096, 2048], "part_sizes": [512] }
	_fc_big_arch    = { "shared_sizes": [8192, 4096, 2048], "part_sizes": [1024, 512] }
	_res_small_arch = { "shared_sizes": [4096, 1024], "part_sizes": [512], "res_blocks": 4, "res_size": 1024 }
	_res_big_arch   = { "shared_sizes": [8192, 4096, 2048], "part_sizes": [1024, 512], "res_blocks": 6, "res_size": 2048 }
	_conv_arch      = { "shared_sizes": [4096, 2048], "part_sizes": [512], "conv_channels": [32, 64, 128], "cat_sizes": [2048] }

	def __init__(self,
				 activation_function=torch.nn.ELU(),
				 batchnorm=True, architecture="fc_small",
				 init="glorot",
				 is2024=True,
				 **kwargs,  # For backwardscompatibility
			):
		self.activation_function = activation_function
		self.batchnorm = batchnorm
		self.architecture = self._backward_comp_arch(architecture)  # Options: 'fc_small', 'fc_big', 'res_small', 'res_big', 'conv'
		self.init = init  # Options: glorot, he or a number
		self.is2024 = is2024

		self.id = hash(time())

		# General purpose values
		self.shared_sizes = self._get_arch()["shared_sizes"]
		self.part_sizes = self._get_arch()["part_sizes"]

		# ResNet values
		if self.architecture.startswith("res"):
			self.res_blocks = self._get_arch()["res_blocks"]
			self.res_size = self._get_arch()["res_size"]

		# CNN values
		if self.architecture.startswith("conv"):
			self.conv_channels = self._get_arch()["conv_channels"]
			self.cat_sizes = self._get_arch()["cat_sizes"]
	
	def _backward_comp_arch(self, arch: str):
		# Ensure backwards compatibility with older model configs
		if arch in ["fc", "res"]:
			return arch + "_small"
		return arch

	def _get_arch(self):
		return getattr(self, f"_{self.architecture}_arch")

	@classmethod
	def _get_non_serializable(cls):
		return { "activation_function": cls._get_activation_function }

	def as_json_dict(self):
		d = deepcopy(self.__dict__)
		for key in ["shared_sizes", "part_sizes", "res_blocks", "res_size", "conv_channels", "cat_sizes"]:
			d.pop(key, None)
		for a, f in self._get_non_serializable().items():
			d[a] = f(d[a], False)
		return d

	@classmethod
	def from_json_dict(cls, conf: dict):
		for a, f in cls._get_non_serializable().items():
			conf[a] = f(conf[a], True)
		return ModelConfig(**conf)

	@staticmethod
	def _get_activation_function(val, from_key: bool):
		afs = {"elu": torch.nn.ELU(), "relu": torch.nn.ReLU()}
		if from_key:
			return afs[val]
		else:
			return [x for x in afs if type(afs[x]) == type(val)][0]


class Model(nn.Module):
	"""
	A fully connected, feed forward Neural Network.
	Also the instantiator class of other network architectures through `create`.
	"""
	shared_net: nn.Sequential
	policy_net: nn.Sequential
	value_net: nn.Sequential

	def __init__(self, config: ModelConfig, logger=NullLogger()):
		super().__init__()
		self.config = config
		self.log = logger

		self._construct_net()
		self.log(f"Created network\n{self.config}\n{self}")

	@staticmethod
	def create(config: ModelConfig, logger=NullLogger()):
		"""
		Allows this class to be used to instantiate other Network architectures based on the content
		of the configuartion file.
		"""
		if config.architecture.startswith("fc"):  return Model(config, logger).to(gpu)
		if config.architecture.startswith("res"): return ResNet(config, logger).to(gpu)
		if config.architecture == "conv":         return ConvNet(config, logger).to(gpu)

		raise KeyError(f"Network architecture should be 'fc_small', 'fc_big', 'res_small', 'res_big', 'conv', but '{config.architecture}' was given")

	def _construct_net(self, pv_input_size: int=None):
		"""
		Constructs a feed forward fully connected DNN.
		"""
		pv_input_size =  self.config.shared_sizes[-1] if pv_input_size is None else pv_input_size

		shared_thiccness = [cube.get_oh_shape(), *self.config.shared_sizes]
		policy_thiccness = [pv_input_size, *self.config.part_sizes, cube.action_dim]
		value_thiccness = [pv_input_size, *self.config.part_sizes, 1]

		self.shared_net = nn.Sequential(*self._create_fc_layers(shared_thiccness, False))
		self.policy_net = nn.Sequential(*self._create_fc_layers(policy_thiccness, True))
		self.value_net = nn.Sequential(*self._create_fc_layers(value_thiccness, True))

	def forward(self, x, policy=True, value=True):
		assert policy or value
		x = self.shared_net(x)
		return_values = []
		if policy:
			policy = self.policy_net(x)
			return_values.append(policy)
		if value:
			value = self.value_net(x)
			return_values.append(value)
		return return_values if len(return_values) > 1 else return_values[0]

	def _create_fc_layers(self, thiccness: list, final: bool):
		"""
		Helper function to return fully connected feed forward layers given a list of layer sizes and
		a final output size.
		"""
		layers = []
		for i in range(len(thiccness)-1):
			l = nn.Linear(thiccness[i], thiccness[i+1])
			if self.config.init == 'glorot': torch.nn.init.xavier_uniform_(l.weight)
			elif self.config.init == 'he': torch.nn.init.kaiming_uniform_(l.weight)
			else: torch.nn.init.constant_(l.weight, float(self.config.init))
			layers.append(l)

			if not (final and i == len(thiccness) - 2):
				layers.append(self.config.activation_function)
				if self.config.batchnorm:
					layers.append(nn.BatchNorm1d(thiccness[i+1]))

		return layers

	def clone(self):
		new_state_dict = {}
		for kw, v in self.state_dict().items():
			new_state_dict[kw] = v.clone()
		new_net = Model.create(self.config)
		new_net.load_state_dict(new_state_dict)
		return new_net

	def get_params(self):
		return torch.cat([x.float().flatten() for x in self.state_dict().values()]).clone()

	def save(self, save_dir: str, is_min=False):
		"""
		Save the model and configuration to the given directory
		The folder will include a pytorch model, and a json configuration file
		"""

		os.makedirs(save_dir, exist_ok=True)
		if is_min:
			model_path = os.path.join(save_dir, "model-best.pt")
			torch.save(self.state_dict(), model_path)
			self.log(f"Saved min model to {model_path}")
			return
		model_path = os.path.join(save_dir, "model.pt")
		torch.save(self.state_dict(), model_path)
		conf_path = os.path.join(save_dir, "config.json")
		with open(conf_path, "w", encoding="utf-8") as conf:
			json.dump(self.config.as_json_dict(), conf, indent=4)
		self.log(f"Saved model to {model_path} and configuration to {conf_path}")

	@staticmethod
	def load(load_dir: str, logger=NullLogger(), load_best=False):
		"""
		Load a model from a configuration directory
		"""

		model_path = os.path.join(load_dir, "model.pt" if not load_best else "model-best.pt")
		conf_path = os.path.join(load_dir, "config.json")
		with open(conf_path, encoding="utf-8") as conf:
			try:
				state_dict = torch.load(model_path, map_location=gpu)
			except FileNotFoundError:
				model_path = os.path.join(load_dir, "model.pt")
				state_dict = torch.load(model_path, map_location=gpu)
			config = ModelConfig.from_json_dict(json.load(conf))

		model = Model.create(config, logger)
		model.load_state_dict(state_dict)
		model.to(gpu)
		# First time the net is loaded, a feedforward is performed, as the first time is slow
		# This avoids skewing evaluation results
		with torch.no_grad():
			model.eval()
			model(cube.as_oh(cube.get_solved()))
			model.train()
		return model


class NonConvResBlock(nn.Module):
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

	def forward(self, x):
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

class ResNet(Model):
	"""
	A Linear Residual Neural Network.
	"""
	#				    /-> policy fc layer(s)
	#  x-> fc layers -> residual blocks
	#				    \-> value fc layer(s)
	def _construct_net(self):
		# Resblock class is very simple currently (does not change size), so its input must match the res_size
		assert self.config.shared_sizes[-1] == self.config.res_size or (not self.config.shared_sizes and self.config.res_size == cube.get_oh_shape())

		# Uses FF constructor to set up feed forward nets. Resblocks are added only to shared net
		super()._construct_net( pv_input_size = self.config.res_size )
		for i in range(self.config.res_blocks):
			resblock = NonConvResBlock(self.config.res_size, self.config.activation_function, self.config.batchnorm)
			self.shared_net.add_module(f'resblock{i}', resblock)


class _CircularPad(nn.Module):
	"""
	Circular padding is broken in convolutional modules in pytorch 1.4 (supposedly fixed 1.5). Therefore this manual implementation
	See https://github.com/pytorch/pytorch/issues/20981 and https://github.com/kornia/kornia/pull/478
	"""
	def __init__(self, padding: list):
		super().__init__()
		self.padding = padding

	def forward(self, x):
		return F.pad(x, self.padding, mode="circular")

class ConvNet(Model):

	shared_conv_net: nn.Sequential
	cat_net: nn.Sequential

	# x -> conv layers                 policy layers
	#                   > cat layer <
	# x -> fc layers                   value layers

	def _construct_net(self):

		# Creates all convolutional layers
		channels_list = [6, *self.config.conv_channels]
		cat_input_size = channels_list[-1] * 8 + self.config.shared_sizes[-1]
		# First convolutional layer has stride of two and special padding
		conv_layers = [_CircularPad([1, 1]), nn.Conv1d(channels_list[0], channels_list[1], kernel_size=3, stride=1)]
		if self.config.batchnorm:
			conv_layers.append(nn.BatchNorm1d(channels_list[1]))
		# Rest have stride of one and normal padding
		for in_channels, out_channels in zip(channels_list[1:-1], channels_list[2:]):
			conv_layers.append(_CircularPad([1, 1]))
			conv_layers.append(nn.Conv1d(in_channels, out_channels, 3))
			conv_layers.append(self.config.activation_function)
			if self.config.batchnorm:
				conv_layers.append(nn.BatchNorm1d(out_channels))
		self.shared_conv_net = nn.Sequential(*conv_layers)

		# Creates concatenation layers
		# Its input size is the raveled conv size + fc output size
		# Its output size is fc output size, as this is also the input size for the policy and value nets
		cat_layers = []
		cat_sizes = [cat_input_size] + self.config.cat_sizes
		for in_size, out_size in zip(cat_sizes[:-1], cat_sizes[1:]):
			cat_layers.append(nn.Linear(in_size, out_size))
			cat_layers.append(self.config.activation_function)
			if self.config.batchnorm:
				cat_layers.append(nn.BatchNorm1d(out_size))
		self.cat_net = nn.Sequential(*cat_layers)

		# Constructs the rest of the network
		super()._construct_net(cat_sizes[-1])

	def forward(self, x, policy=True, value=True):
		assert policy or value

		# Shared part of network
		fc_out = self.shared_net(x)
		conv_out = self.shared_conv_net(cube.as_correct(x)).view(len(x), -1)
		x = torch.cat([fc_out, conv_out], dim=1)
		x = self.cat_net(x)

		# Policy and value parts
		return_values = []
		if policy:
			policy = self.policy_net(x)
			return_values.append(policy)
		if value:
			value = self.value_net(x)
			return_values.append(value)
		return return_values if len(return_values) > 1 else return_values[0]

