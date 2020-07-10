import os
import json
import torch

from tests import MainTest

from librubiks import gpu
from librubiks.model import Model, ModelConfig, create_net, load_net, save_net
from librubiks.utils import NullLogger


class TestModel(MainTest):
	def test_model(self):
		config = ModelConfig("cube2024")
		model = create_net(config)
		assert next(model.parameters()).device.type == gpu.type
		model.eval()
		x = torch.randn(2, 480).to(gpu)
		model(x)
		model.train()
		model(x)

	# TODO: Reintroduce this test once resnet has been fixed
	# def test_resnet(self):
	# 	config = ModelConfig("cube2024", architecture = 'res_big')
	# 	model = create_net(config)
	# 	assert next(model.parameters()).device.type == gpu.type
	# 	model.eval()
	# 	x = torch.randn(2, 480).to(gpu)
	# 	model(x)
	# 	model.train()
	# 	model(x)

	def test_save_and_load(self):
		torch.manual_seed(42)

		config = ModelConfig("cube2024")
		model = create_net(config, logger=NullLogger())
		model_dir = "local_tests/local_model_test"
		save_net(model, model_dir)
		assert os.path.exists(f"{model_dir}/config.json")
		assert os.path.exists(f"{model_dir}/model.pt")

		model = load_net(model_dir).to(gpu)
		assert next(model.parameters()).device.type == gpu.type

	def test_model_config(self):
		cf = ModelConfig("cube2024", activation_function="relu")
		with open("local_tests/test_config.json", "w", encoding="utf-8") as f:
			json.dump(cf.as_json_dict(), f)
		with open("local_tests/test_config.json", encoding="utf-8") as f:
			cf = ModelConfig.from_json_dict(json.load(f))
		assert type(cf.get_af()) == type(torch.nn.ReLU())

	def test_init(self):
		for init in ['glorot', 'he', 0, 1.123123123e-3]:
			cf = ModelConfig("cube2024", init=init)
			model = create_net(cf)
			x = torch.randn(2, 480).to(gpu)
			model(x)
