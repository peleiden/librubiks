import os
import json
import torch

from src.tests import MainTest

from src.rubiks import cpu, gpu
from src.rubiks.model import Model, ModelConfig
from src.rubiks.utils.logger import NullLogger


class TestModel(MainTest):
	def test_model(self):
		config = ModelConfig()
		model = Model(config).to(gpu)
		assert next(model.parameters()).device.type == gpu.type
		model.eval()
		x = torch.randn(2, 480).to(gpu)
		model(x)
		model.train()
		model(x)

	def test_save_and_load(self):
		torch.manual_seed(42)

		config = ModelConfig()
		model = Model(config, logger=NullLogger()).to(gpu)
		model_dir = "local_tests/local_model_test"
		model.save(model_dir)
		assert os.path.exists(f"{model_dir}/config.json")
		assert os.path.exists(f"{model_dir}/model.pt")

		model = Model.load(model_dir).to(gpu)
		assert next(model.parameters()).device.type == gpu.type


	def test_model_config(self):
		cf = ModelConfig(torch.nn.ReLU())
		with open("local_tests/test_config.json", "w") as f:
			json.dump(cf.as_json_dict(), f)
		with open("local_tests/test_config.json") as f:
			cf = ModelConfig.from_json_dict(json.load(f))
		assert type(cf.activation_function) == type(torch.nn.ReLU())
