import sys, os
from shutil import rmtree
from glob import glob as glob  # glob
from typing import List

import json

import numpy as np
import torch

from librubiks import envs
from librubiks.utils import get_commit, Logger

from librubiks.model import Model, ModelConfig, create_net, save_net, load_net
from librubiks.train import Train

from librubiks.solving import agents as ag
from librubiks.solving.agents import ValueSearch, DeepAgent, Agent
from librubiks.solving.evaluation import Evaluator, EvalData


class TrainJob:
	eval_games = 200  # Not given as arguments to __init__, as they should be accessible in runtime_estim
	max_time = 0.05

	def __init__(
		self,
		# Set by parser, should correspond to options in runtrain
		name: str,
		location: str,
		env_key: str,
		rollouts: int,
		rollout_games: int,
		rollout_depth: int,
		batch_size: int,
		alpha_update: float,
		lr: float,
		gamma: float,
		tau: float,
		update_interval: int,
		optim_fn: str,
		reward_method: str,
		arch: str,
		nn_init: str,
		evaluation_interval: int,
		analysis: bool,

		# Not set by argparser/configparser
		# TODO: Evaluate on several different depths or maybe reintroduce S
		agent = ValueSearch(net=None),
		scrambling_depths: tuple = (12,),
	):

		self.name = name
		assert isinstance(self.name, str)
		
		self.env = envs.get_env(env_key)

		self.rollouts = rollouts
		assert self.rollouts > 0
		self.rollout_games = rollout_games
		assert self.rollout_games > 0
		self.rollout_depth = rollout_depth
		assert rollout_depth > 0
		self.batch_size = batch_size
		assert 0 < self.batch_size <= self.rollout_games * self.rollout_depth

		self.alpha_update = alpha_update
		assert 0 <= alpha_update <= 1
		self.lr = lr
		assert type(lr) == float and lr > 0
		self.gamma = gamma
		assert 0 < gamma <= 1
		self.tau = tau
		assert 0 < tau <= 1
		self.update_interval = update_interval
		assert isinstance(self.update_interval, int) and 0 <= self.update_interval
		self.optim_fn = getattr(torch.optim, optim_fn)
		assert issubclass(self.optim_fn, torch.optim.Optimizer)

		self.location = location
		self.log = Logger(f"{self.location}/train.log", name)  # Already creates logger at init to test whether path works
		self.log(f"Initialized {self.name}")

		self.evaluator = Evaluator(n_games=self.eval_games, max_time=self.max_time, scrambling_depths=scrambling_depths, logger=self.log)
		self.evaluation_interval = evaluation_interval
		assert isinstance(self.evaluation_interval, int) and 0 <= self.evaluation_interval
		self.agent = agent
		assert isinstance(self.agent, DeepAgent)

		assert nn_init in ["glorot", "he"] or (float(nn_init) or True),\
			f"Initialization must be glorot, he or a number, but was {nn_init}"
		assert arch in ["fc", "res"]
		self.model_cfg = ModelConfig(env_key, architecture=arch, init=nn_init)

		self.analysis = analysis
		assert isinstance(self.analysis, bool)

		self.reward_method = reward_method
		assert self.reward_method in ["paper", "lapanfix", "schultzfix", "reward0"]

	def execute(self):

		# Sets representation
		self.log.section(f"Starting job:\n{self.name} using {self.env} environment\nLocation {self.location}\nCommit: {get_commit()}")

		train = Train(
			env                 = self.env,
			rollouts            = self.rollouts,
			rollout_games       = self.rollout_games,
			rollout_depth       = self.rollout_depth,
			batch_size          = self.batch_size,
			optim_fn            = self.optim_fn,
			lr                  = self.lr,
			gamma               = self.gamma,
			alpha_update        = self.alpha_update,
			tau                 = self.tau,
			update_interval     = self.update_interval,
			reward_method       = self.reward_method,
			agent               = self.agent,
			evaluator           = self.evaluator,
			evaluation_interval = self.evaluation_interval,
			with_analysis       = self.analysis,
			logger              = self.log,
		)

		net = create_net(self.model_cfg, self.log)
		net, best_net, traindata, analysisdata = train.train(net)
		save_net(net, self.location)
		if self.evaluation_interval:
			save_net(best_net, self.location, is_best=True)
		paths = traindata.save(self.location)
		self.log("Saved training data to\n\t" + "\n\t".join(paths))

		if analysisdata is not None:
			paths = analysisdata.save(os.path.join(self.location, "analysis"))
			self.log("Saved anaylsis data to\n\t" + "\n\t".join(paths))

	@staticmethod
	def clean_dir(loc: str):
		"""
		Cleans a training directory except for train_config.ini, the content of which is also returned
		"""
		tcpath = f"{loc}/train_config.ini"
		with open(tcpath, encoding="utf-8") as f:
			content = f.read()
		rmtree(loc)
		os.mkdir(loc)
		with open(tcpath, "w", encoding="utf-8") as f:
			f.write(content)
		return content


class EvalJob:

	def __init__(
		self,
		# Set by parser, should correspond to options in runeval
		name: str,
		location: str,
		use_best: bool,
		env_key: str,
		agents: List[str],
		games: int,
		max_time: float,
		max_states: int,
		scrambling: str,
		optimized_params: bool,
		astar_lambdas: List[float],
		astar_expansions: List[int],

		# Currently not set by parser
		in_subfolder: bool,  # Should be true if there are multiple experiments
	):

		self.name = name
		self.location = location
		self.env = envs.get_env(env_key)

		assert isinstance(games, int) and games
		assert max_time >= 0
		assert max_states >= 0
		assert max_time or max_states
		scrambling = range(*scrambling)
		assert isinstance(optimized_params, bool)

		# Create evaluator
		self.log = Logger(f"{self.location}/{self.name}.log", name)  # Already creates logger at init to test whether path works
		self.evaluator = Evaluator(n_games=games, max_time=max_time, max_states=max_states, scrambling_depths=scrambling, logger=self.log)
		self.log("Collecting agents")

		# Create agents
		astar_lambdas = iter(astar_lambdas)
		astar_expansions = iter(astar_expansions)
		self.agents = list()  # List of all agents that will be evaluated
		for agent_str in agents:
			agent = getattr(ag, agent_str)
			assert issubclass(agent, ag.Agent) and not agent == ag.Agent and not agent == ag.DeepAgent,\
				f"Agent must be a subclass of agents.Agent or agents.DeepAgent, but {agent_str} was given"

			if issubclass(agent, ag.DeepAgent):
				self.agents, agent_args = {}, {}

				# Some DeepAgents need specific arguments
				if agent == ag.AStar:
					astar_lambda, astar_expansion = next(astar_lambdas), next(astar_expansions)
					assert isinstance(astar_lambda, float) and 0 <= astar_lambda <= 1, "AStar lambda must be float in [0, 1]"
					assert isinstance(astar_expansion, int) and astar_expansion >= 1 and (not max_states or astar_expansion < max_states),\
						"Expansions must be int < max states"
					agent_args = { 'lambda_': astar_lambda, 'expansions': astar_expansion }
				else:  # Non-parametric methods go brrrr
					agent_args = {}

				# Use parent folder, if parser has generated multiple folders
				search_location = os.path.dirname(os.path.abspath(self.location)) if in_subfolder else self.location
				# DeepAgent might have to test multiple NN's
				for folder in glob(f"{search_location}/*/") + [search_location]:
					if not os.path.isfile(os.path.join(folder, 'model.pt')):
						continue
					if optimized_params and agent in [ag.AStar]:
						parampath = os.path.join(folder, f'{agent_str}_params.json')
						if os.path.isfile(parampath):
							with open(parampath, 'r') as paramfile:
								agents_args = json.load(paramfile)
						else:
							self.log.throw(FileNotFoundError(
								f"Optimized params was set to true, but no file {parampath} was found, proceding with arguments for this {agent_str}."
							))
					a = agent.from_saved(folder, use_best=use_best, **agent_args)
					if a.env.__class__ == self.env.__class__:
						if in_subfolder and len(agents) > 1:  # Add distinguishing names to agents
							a.name += f" - {a.net}"
						agents.append(a)
						self.log(f"Added agent '{a}' to agent list")
					else:
						self.log(f"Agent '{a}' was not added to list, as the network uses the {a.env} environment")

				if not self.agents:
					self.log.throw(FileNotFoundError(f"No model.pt found in folder or subfolder of {self.location}"))

			else:
				self.agents.append(agent(self.env))
				self.log(f"Added agent '{self.agents[-1]}' to agent list")

			self.log(f"Initialized {self.name} with agents " + '\n'.join(str(s) for s in self.agents) + "\nEnvironment: {self.env}")
			self.log(f"Time estimate: {len(self.agents) * self.evaluator.approximate_time() / 60:.2f} min. (Rough upper bound)")

	def execute(self):
		self.log(f"Beginning evaluator {self.name}\nLocation {self.location}\nCommit: {get_commit()}")
		for agent in self.agents:
			self.log.section(f'Evaluationg agent {agent}')
			evaldata = self.evaluator.eval(agent)
			subfolder = os.path.join(self.location, "evaluation_results")
			paths = evaldata.save(subfolder)
			self.log("Saved evaluation results to\n\t" + "\n\t".join(paths))
