import sys, os
from shutil import rmtree
from glob import glob as glob  # glob

import json

import numpy as np
import torch

from librubiks import envs
from librubiks.utils import get_commit, Logger

from librubiks.model import Model, ModelConfig, create_net, save_net, load_net
from librubiks.train import Train

from librubiks.solving import agents
from librubiks.solving.agents import ValueSearch, DeepAgent, Agent
from librubiks.solving.evaluation import Evaluator


class TrainJob:
	eval_games = 200  # Not given as arguments to __init__, as they should be accessible in runtime_estim
	max_time = 0.05

	def __init__(self,
				 # Set by parser, should correspond to options in runtrain
				 name: str,
				 location: str,
				 env: str,
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
				 # TODO: Evaluate on several different depths
				 agent = ValueSearch(net=None),
				 scrambling_depths: tuple = (12,),
			):

		self.name = name
		assert isinstance(self.name, str)
		
		self.env = envs.get_env(env)

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
		self.logger = Logger(f"{self.location}/train.log", name)  # Already creates logger at init to test whether path works
		self.logger.log(f"Initialized {self.name}")

		self.evaluator = Evaluator(n_games=self.eval_games, max_time=self.max_time, scrambling_depths=scrambling_depths, logger=self.logger)
		self.evaluation_interval = evaluation_interval
		assert isinstance(self.evaluation_interval, int) and 0 <= self.evaluation_interval
		self.agent = agent
		assert isinstance(self.agent, DeepAgent)

		assert nn_init in ["glorot", "he"] or ( float(nn_init) or True ),\
				f"Initialization must be glorot, he or a number, but was {nn_init}"
		assert arch in ["fc", "res"]
		self.model_cfg = ModelConfig(env, architecture=arch, init=nn_init)

		self.analysis = analysis
		assert isinstance(self.analysis, bool)

		self.reward_method = reward_method
		assert self.reward_method in ["paper", "lapanfix", "schultzfix", "reward0"]

	def execute(self):

		# Sets representation
		self.logger.section(f"Starting job:\n{self.name} using {self.env.name} environment\nLocation {self.location}\nCommit: {get_commit()}")

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
			logger              = self.logger,
		)

		net = create_net(self.model_cfg, self.logger)
		net, best_net, traindata, analysisdata = train.train(net)
		save_net(net, self.location)
		if self.evaluation_interval:
			save_net(best_net, self.location, is_best=True)
		traindata.save(self.location)

		analysispath = os.path.join(self.location, "analysis")
		datapath = os.path.join(self.location, "train-data")
		os.mkdir(datapath)
		os.mkdir(analysispath)

		if analysisdata is not None:
			analysisdata.save(datapath)

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
			name: str,
			# Set by parser, should correspond to options in runeval
			location: str,
			use_best: bool,
			agent: str,
			games: int,
			max_time: float,
			max_states: int,
			scrambling: str,
			optimized_params: bool,
			mcts_c: float,
			mcts_graph_search: bool,
			policy_sample: bool,
			astar_lambda: float,
			astar_expansions: int,
			egvm_epsilon: float,
			egvm_workers: int,
			egvm_depth: int,

			# Currently not set by parser
			in_subfolder: bool,  # Should be true if there are multiple experiments
		):

		self.name = name
		self.location = location

		assert isinstance(games, int) and games
		assert max_time >= 0
		assert max_states >= 0
		assert max_time or max_states
		scrambling = range(*scrambling)
		assert isinstance(optimized_params, bool)

		# Create evaluator
		self.logger = Logger(f"{self.location}/{self.name}.log", name)  # Already creates logger at init to test whether path works
		self.evaluator = Evaluator(n_games=games, max_time=max_time, max_states=max_states, scrambling_depths=scrambling, logger=self.logger)

		# Create agents
		agent_string = agent
		agent = getattr(agents, agent_string)
		assert issubclass(agent, agents.Agent)

		if issubclass(agent, agents.DeepAgent):
			self.agents, self.reps, agents_args = {}, {}, {}

			# DeepAgents need specific arguments
			if agent == agents.AStar:
				assert isinstance(astar_lambda, float) and 0 <= astar_lambda <= 1, "AStar lambda must be float in [0, 1]"
				assert isinstance(astar_expansions, int) and astar_expansions >= 1 and (not max_states or astar_expansions < max_states) , "Expansions must be int < max states"
				agents_args = { 'lambda_': astar_lambda, 'expansions': astar_expansions }
			else:  # Non-parametric methods go brrrr
				agents_args = {}

			search_location = os.path.dirname(os.path.abspath(self.location)) if in_subfolder else self.location # Use parent folder, if parser has generated multiple folders
			# DeepAgent might have to test multiple NN's
			for folder in glob(f"{search_location}/*/") + [search_location]:
				if not os.path.isfile(os.path.join(folder, 'model.pt')): continue
				with open(f"{folder}/config.json") as f:
					cfg = json.load(f)
				if optimized_params and agent in [agents.AStar]:
					parampath = os.path.join(folder, f'{agent_string}_params.json')
					if os.path.isfile(parampath):
						with open(parampath, 'r') as paramfile:
							agents_args = json.load(paramfile)
					else:
						self.logger.log(f"Optimized params was set to true, but no file {parampath} was found, proceding with arguments for this {agent_string}.")

				agent = agent.from_saved(folder, use_best=use_best, **agents_args)
				key = f'{agent}{"" if folder == search_location else " " + os.path.basename(folder.rstrip(os.sep))}'

				self.reps[key] = cfg["is2024"]
				self.agents[key] = agent

			if not self.agents:
				raise FileNotFoundError(f"No model.pt found in folder or subfolder of {self.location}")
			self.logger.log(f"Loaded model from {search_location}")

		else:
			agent = agent()
			self.agents = { str(agent): agent }
			self.reps   = { str(agent): True  }

		self.agent_results = {}
		self.logger.log(f"Initialized {self.name} with agents {', '.join(str(s) for s in self.agents)}")
		self.logger.log(f"TIME ESTIMATE: {len(self.agents) * self.evaluator.approximate_time() / 60:.2f} min.\t(Rough upper bound)")

	def execute(self):
		self.logger.log(f"Beginning evaluator {self.name}\nLocation {self.location}\nCommit: {get_commit()}")
		for (name, agent), representation in zip(self.agents.items(), self.reps.values()):
			self.agent_results[name] = self._single_exec(name, agent)

	def _single_exec(self, name: str, agent: Agent):
		self.logger.section(f'Evaluationg agent {name}')
		evaldata = self.evaluator.eval(agent)
		subfolder = os.path.join(self.location, "evaluation_results")
		os.makedirs(subfolder, exist_ok=True)
		paths = evaldata.save(subfolder)
		self.logger.log("Saved evaluation results to\n" + "\n".join(paths))
		return res, states, times

	@staticmethod
	def plot_all_jobs(jobs: list, save_location: str):
		results, states, times, settings = dict(), dict(), dict(), dict()
		export_settings = dict()
		for job in jobs:
			for agent, (result, states_, times_) in job.agent_results.items():
				key = agent if len(jobs) == 1 else f"{job.name} - {agent}"
				results[key] = result
				states[key] = states_
				times[key] = times_
				settings[key] = {
					"n_games": job.evaluator.n_games,
					"max_time": job.evaluator.max_time,
					"max_states": job.evaluator.max_states,
					"scrambling_depths": job.evaluator.scrambling_depths,
				}
				export_settings[key] = { **settings[key], "scrambling_depths": job.evaluator.scrambling_depths.tolist() }
		eval_settings_path = os.path.join(save_location, "eval_settings.json")
		with open(eval_settings_path, "w", encoding="utf-8") as f:
			json.dump(export_settings, f, indent=4)
		savepaths = Evaluator.plot_evaluators(results, states, times, settings, save_location)
		joinedpaths = "\n".join(savepaths)
		job.logger(f"Saved settings to {eval_settings_path} and plots to\n{joinedpaths}")

