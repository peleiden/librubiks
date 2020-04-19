import json
import os
import torch
import torch.nn as nn
from copy import deepcopy

from src.rubiks.cube.cube import Cube
from src.rubiks import cpu, gpu
from src.rubiks.utils.logger import Logger, NullLogger

from dataclasses import dataclass


@dataclass
class ModelConfig:
	activation_function: torch.nn.functional = torch.nn.ELU()
	dropout: float = 0
	batchnorm: bool = True

	@classmethod
	def _get_non_serializable(cls):
		return {"activation_function": cls._conv_activation_function}

	def as_json_dict(self):
		d = deepcopy(self.__dict__)
		for a, f in self._get_non_serializable().items():
			d[a] = f(d[a], False)
		return d
	
	@classmethod
	def from_json_dict(cls, conf: dict):
		for a, f in cls._get_non_serializable().items():
			conf[a] = f(conf[a], True)
		return ModelConfig(**conf)

	@staticmethod
	def _conv_activation_function(val, from_key: bool):
		afs = {"elu": torch.nn.ELU(), "relu": torch.nn.ReLU()}
		if from_key:
			return afs[val]
		else:
			return [x for x in afs if type(afs[x]) == type(val)][0]


class Model(nn.Module):
	
	def __init__(self, config: ModelConfig, logger=NullLogger(),):
		super().__init__()
		self.config = config
		self.log = logger

		shared_thiccness = [Cube.get_oh_shape(), 4096, 2048]
		policy_thiccness = [shared_thiccness[-1], 512, 12]
		value_thiccness = [shared_thiccness[-1], 512, 1]
		self.shared_net = nn.Sequential(*self._create_fc_layers(shared_thiccness, False))
		self.policy_net = nn.Sequential(*self._create_fc_layers(policy_thiccness, True))
		self.value_net = nn.Sequential(*self._create_fc_layers(value_thiccness, True))

		self.log(f"Created network\n{self.config}\n{self}")

	def _create_fc_layers(self, thiccness: list, final: bool):
		layers = []
		for i in range(len(thiccness)-1):
			layers.append(nn.Linear(thiccness[i], thiccness[i+1]))
			if not (final and i == len(thiccness) - 2):
				layers.append(self.config.activation_function)
				layers.append(nn.Dropout(self.config.dropout))
				if self.config.batchnorm:
					layers.append(nn.BatchNorm1d(thiccness[i+1]))
		
		return layers
	
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

	def clone(self):
		new_state_dict = {}
		for kw, v in self.state_dict().items():
			new_state_dict[kw] = v.cpu().clone()
		new_net = Model(self.config)
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
			model_path = os.path.join(save_dir, "model-min.pt")
			torch.save(self.state_dict(), model_path)
			self.log(f"Saved min model to {model_path}")
			return
		model_path = os.path.join(save_dir, "model.pt")
		torch.save(self.state_dict(), model_path)
		conf_path = os.path.join(save_dir, "config.json")
		with open(conf_path, "w", encoding="utf-8") as conf:
			json.dump(self.config.as_json_dict(), conf)
		self.log(f"Saved model to {model_path} and configuration to {conf_path}")
	
	@staticmethod
	def load(load_dir: str):
		"""
		Load a model from a configuration directory
		"""
		
		model_path = os.path.join(load_dir, "model.pt")
		conf_path = os.path.join(load_dir, "config.json")
		with open(conf_path, encoding="utf-8") as conf:
			state_dict = torch.load(model_path, map_location=gpu)
			config = ModelConfig.from_json_dict(json.load(conf))
		
		model = Model(config)
		model.load_state_dict(state_dict)
		model.to(gpu)
		# First time the net is loaded, a feedforward is performed, as the first time is slow
		# This avoids skewing evaluation results
		with torch.no_grad():
			model.eval()
			model(Cube.as_oh(Cube.get_solved()).to(gpu))
			model.train()
		return model



