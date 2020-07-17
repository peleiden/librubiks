import os
from dataclasses import dataclass
import functools
import json
import numpy as np
import torch

cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rc_params = { "font.size": 24, "legend.fontsize": 22, "legend.framealpha": 0.5 }  # matplotlib settings
rc_params_small = { **rc_params, "font.size": 20, "legend.fontsize": 18 }  # Same but with smaller font


def reset_cuda():
	torch.cuda.empty_cache()
	if torch.cuda.is_available():
		torch.cuda.synchronize()


def no_grad(fun):
	functools.wraps(fun)
	def wrapper(*args, **kwargs):
		with torch.no_grad():
			return fun(*args, **kwargs)
	return wrapper


@dataclass
class DataStorage:
	subfolder = ""
	json_name = "data.json"

	def save(self, loc: str):
		loc = self._get_loc(loc)
		os.makedirs(loc, exist_ok=True)

		# Split data by whether it should be saved .json and to .npy
		to_json = dict()
		to_npy = dict()
		for key, data in self.__dict__.items():
			if isinstance(data, np.ndarray):
				to_npy.update({key: data})
			else:
				to_json.update({key: data})
		
		# Save data
		paths = [os.path.join(loc, self.json_name)]
		with open(paths[0], "w", encoding="utf-8") as f:
			json.dump(to_json, f, indent=4)
		for key, arr in to_npy.items():
			paths.append(os.path.join(loc, f"{key}.npy"))
			np.save(paths[-1], arr)
		return paths
	
	@classmethod
	def load(cls, loc: str):
		loc = cls._get_loc(loc)
		# Load .json
		with open(os.path.join(loc, cls.json_name), encoding="utf-8") as f:
			non_numpy = json.load(f)

		# Get .npy filenames and load these
		npys = dict()
		for key, field in cls.__dict__["__dataclass_fields__"].items():
			if field.type == np.ndarray:
				npys[key] = np.load(os.path.join(loc, f"{key}.npy"))
		
		return cls(**non_numpy, **npys)
	
	@classmethod
	def _get_loc(cls, loc: str):
		return os.path.join(loc, cls.subfolder) if cls.subfolder else loc

