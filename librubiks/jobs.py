import os
from shutil import rmtree
from collections import Counter
from glob import glob as glob  # glob
from typing import List, Tuple

import json

import numpy as np
import torch

from librubiks import envs
from pelutils import get_commit, log

from librubiks.model import ModelConfig, create_net, save_net
from librubiks.train import Train, TrainData
from librubiks.analysis import AnalysisData

from librubiks.solving import agents as ag
from librubiks.solving.agents import ValueSearch, DeepAgent
from librubiks.solving.evaluation import Evaluator, EvalData

import librubiks.plots.trainplot as tp
import librubiks.plots.analysisplot as ap
import librubiks.plots.evalplot as ep


class TrainJob:
    eval_games = 200
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

        self.location = location
        log.configure(f"{self.location}/train.log", name)

        self.name = name
        assert isinstance(self.name, str) and len(self.name) > 0

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

        self.evaluator = Evaluator(n_games=self.eval_games, max_time=self.max_time, scrambling_depths=scrambling_depths)
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

        log(f"Initialized training job '{self.name}'")

    def execute(self):

        # Sets representation
        log.section(f"Starting job:\n{self.name} using {self.env} environment\nLocation {self.location}\nCommit: {get_commit()}")

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
            name                = self.name,
            with_analysis       = self.analysis,
        )

        net = create_net(self.model_cfg)
        log(f"Created network\n{net.config}\n{net}")
        net, best_net, traindata, analysisdata = train.train(net)
        p, c = save_net(net, self.location)
        log("Saved model to %s and configuration to %s" % (p, c))
        if self.evaluation_interval:
            p, __ = save_net(best_net, self.location, is_best=True)
            log("Saved best model %s" % p)
        paths = traindata.save(self.location)
        log("Saved training data to", *paths, sep="\n- ")

        if analysisdata is not None:
            paths = analysisdata.save(os.path.join(self.location))
            log("Saved anaylsis data to", *paths, sep="\n- ")

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
        log.configure(f"{self.location}/{self.name}.log", name)
        self.env = envs.get_env(env_key)

        assert isinstance(games, int) and games
        assert max_time >= 0
        assert max_states >= 0
        assert max_time or max_states
        scrambling = range(*scrambling)
        assert isinstance(optimized_params, bool)

        # Create evaluator
        self.evaluator = Evaluator(n_games=games, max_time=max_time, max_states=max_states, scrambling_depths=scrambling)

        # Create agents
        log("Collecting agents")
        # TODO: If one lambda or expansion is given, this should apply to all
        astar_lambdas = iter(astar_lambdas)
        astar_expansions = iter(astar_expansions)
        self.agents = list()  # List of all agents that will be evaluated
        for agent_str in agents:
            agent = getattr(ag, agent_str)
            assert issubclass(agent, ag.Agent) and not agent == ag.Agent and not agent == ag.DeepAgent,\
                f"Agent must be a subclass of agents.Agent or agents.DeepAgent, but {agent_str} was given"

            if issubclass(agent, ag.DeepAgent):

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
                                agent_args = json.load(paramfile)
                        else:
                            log.throw(FileNotFoundError(
                                f"Optimized params was set to true, but no file {parampath} was found, "
                                f"proceding with arguments for this {agent_str}."
                            ))
                    a = agent.from_saved(folder, use_best=use_best, **agent_args)
                    if a.env.__class__ == self.env.__class__:
                        self.agents.append(a)
                        log(f"Added agent '{a}' to agent list")
                    else:
                        log(f"Agent '{a}' was not added to list, as the network uses the {a.env} environment")

                # Handl agents with the same name
                # TODO Better way to uniquely identify searchers than layer sizes - perhaps name parameter?
                counts = Counter([agent.name for agent in self.agents])
                for agent in self.agents:
                    if counts[agent.name] > 1:
                        agent.name += " - " + ", ".join([str(x) for x in agent.net.config.layer_sizes])
            else:
                self.agents.append(agent(self.env))
                log(f"Added agent '{self.agents[-1]}' to agent list")

            log(f"Initialized {self.name} with agents", *self.agents, f"\nEnvironment: {self.env}")
            log(f"Time estimate: {len(self.agents) * self.evaluator.approximate_time() / 60:.2f} min. (Rough upper bound)")

    def execute(self):
        log(f"Beginning evaluator {self.name}\nLocation {self.location}\nCommit: {get_commit()}")
        evaldata = self.evaluator.eval(self.agents)
        paths = evaldata.save(self.location)
        log("Saved evaluation results to", *paths, sep="\n- ")


class PlotJob:

    _standard = (15, 10)
    _wide = (22, 10)

    def __init__(self, loc: str, train: bool, analysis: bool, eval_: bool):
        self.loc = loc
        self.train = train
        self.analysis = analysis
        self.eval = eval_

        log.configure(os.path.join(self.loc, "plots.log"), "Plotting")
        log(
            "Plotting training"   if self.train    else "Not plotting training",
            "Plotting analysis"   if self.analysis else "Not plotting analysis",
            "Plotting evaluation" if self.eval     else "Not plotting evaluation",
        )

    def execute(self):
        train_data, analysis_data, eval_data = self._get_data()

        if train_data:
            log.section("Plotting training data")
            paths = []
            for loc, t_d in train_data.items():  # Make train_data iterable again!
                loc = os.path.join(loc, "train-plots")
                os.makedirs(loc, exist_ok=True)
                paths.append(tp.plot_training(loc, t_d, self._standard))
            log("Saved training plots to", *paths, sep="\n- ")

        if analysis_data:
            log.section("Plotting analysis data")
            paths = []
            for loc, a_d in analysis_data.items():
                loc = os.path.join(loc, "analysis-plots")
                os.makedirs(loc, exist_ok=True)
                paths.append(ap.plot_substate_distributions(loc, a_d, self._standard))
                paths.append(ap.visualize_first_states(loc, a_d, self._standard))
                paths.append(ap.plot_value_targets(loc, a_d, self._standard))
                # paths.append(ap.plot_net_changes(loc, a_d, self._standard))
            log("Saved analysis plots to", *paths, sep="\n- ")

        if eval_data:
            log.section("Plotting evaluation data")
            paths = []
            for loc, e_d in eval_data.items():
                loc = os.path.join(loc, "eval-plots")
                os.makedirs(loc, exist_ok=True)
                paths.append(ep.plot_depth_win(loc, e_d, self._standard))
                paths.append(ep.plot_sol_length_boxplots(loc, e_d))
                paths.append(ep.plot_time_states_winrate(loc, e_d, self._standard, is_times=False))
                paths.append(ep.plot_time_states_winrate(loc, e_d, self._standard, is_times=True))
                paths.extend(ep.plot_distributions(loc, e_d, self._standard))
            log("Saved evaluation plots to", *paths, sep="\n- ")

    def _get_data(self) -> Tuple[dict, dict, dict]:
        """
        Return three dicts (or None), the keys of which are locations and the values data
        First for train, second for analysis, and third for evaluation
        """

        log.section("Searching for data directories")
        directories   = self._list_dirs(self.loc)
        train_dirs    = self._filter_by_name(TrainData.subfolder,    directories)
        analysis_dirs = self._filter_by_name(AnalysisData.subfolder, directories)
        eval_dirs     = self._filter_by_name(EvalData.subfolder,     directories)

        sep = "\n- "
        if self.train:
            if train_dirs:
                log("Found the following directories with training data", *train_dirs, sep=sep)
            else:
                log("Found no directories with training data")
        if self.analysis:
            if analysis_dirs:
                log("Found the following directories with analysis data", *analysis_dirs, sep=sep)
            else:
                log("Found no directories with analysis data")
        if self.eval:
            if eval_dirs:
                log("Found the following directories with evaluation data", *eval_dirs, sep=sep)
            else:
                log("Found no directories with evaluation data")

        train_data    = { x: TrainData.load(x)    for x in train_dirs    } if self.train    else None
        analysis_data = { x: AnalysisData.load(x) for x in analysis_dirs } if self.analysis else None
        eval_data     = { x: EvalData.load(x)     for x in eval_dirs     } if self.eval     else None

        return train_data, analysis_data, eval_data

    @staticmethod
    def _list_dirs(loc: str) -> List[str]:
        return [x[0] for x in os.walk(loc)]

    @staticmethod
    def _filter_by_name(name: str, directories: List[str]) -> List[str]:
        """Return parent folders of all directories with given name"""
        dirs = list()
        for directory in directories:
            split = os.path.split(directory)
            if name in split:
                dirs.append(os.path.join(*split[:split.index(name)]))
        return np.unique(dirs).tolist()
