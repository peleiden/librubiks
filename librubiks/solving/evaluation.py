import os

import numpy as np
from scipy import stats
import matplotlib.colors as mcolour
import matplotlib.pyplot as plt

from librubiks import cube, rc_params, rc_params_small
from librubiks.utils import NullLogger, Logger, TickTock, TimeUnit, bernoulli_error
from librubiks.solving import agents


class EvaluationData:
	# TODO Similar to TrainData
	raise NotImplementedError


class Evaluator:
	def __init__(self,
		         n_games,
		         scrambling_depths: range or list,
		         max_time = None,  # Max time to completion per game
		         max_states = None,  # The max number of states to explore per game
		         logger: Logger = NullLogger()
		):

		self.n_games = n_games
		self.max_time = max_time
		self.max_states = max_states

		self.tt = TickTock()
		self.log = logger
		# Use array of scrambling of scrambling depths if not deep evaluation else just a one element array with 0
		self.scrambling_depths = np.array(scrambling_depths) if scrambling_depths != range(0) else np.array([0])

		self.log("\n".join([
			"Creating evaluator",
			f"Games per scrambling depth: {self.n_games}",
			f"Scrambling depths: {scrambling_depths if self._isdeep() else 'Uniformly sampled in [100, 999]'}",
		]))

	def _isdeep(self):
		return self.scrambling_depths.size == 1 and self.scrambling_depths[0] == 0

	def approximate_time(self):
		return self.max_time * self.n_games * len(self.scrambling_depths)

	def _eval_game(self, agent: agents.Agent, depth: int, profile: str):
		turns_to_complete = -1  # -1 for unfinished
		state, _, _ = cube.scramble(depth, True)
		self.tt.profile(profile)
		solution_found = agent.search(state, self.max_time, self.max_states)
		dt = self.tt.end_profile(profile)
		if solution_found: turns_to_complete = len(agent.action_queue)
		return turns_to_complete, dt

	def eval(self, agent: agents.Agent) -> (np.ndarray, np.ndarray, np.ndarray):
		"""
		Evaluates an agent
		Returns results which is an a len(self.scrambling_depths) x self.n_games matrix
		Each entry contains the number of steps needed to solve the scrambled cube or -1 if not solved
		"""
		self.log.section(f"Evaluation of {agent}")
		self.log("\n".join([
			f"{self.n_games*len(self.scrambling_depths)} cubes",
			f"Maximum solve time per cube is {TickTock.stringify_time(self.max_time, TimeUnit.second)} "
			f"and estimated total time <= {TickTock.stringify_time(self.approximate_time(), TimeUnit.minute)}" if self.max_time else "No time limit given",
			f"Maximum number of explored states is {TickTock.thousand_seps(self.max_states)}" if self.max_states else "No max states given",
		]))
		
		res = []
		states = []
		times = []
		for d in self.scrambling_depths:
			for _ in range(self.n_games):
				if self._isdeep():  # Randomly sample evaluation depth for deep evaluations
					d = np.random.randint(100, 1000)
				p = f"Evaluation of {agent}. Depth {'100 - 999' if self._isdeep() else d}"
				r, dt = self._eval_game(agent, d, p)

				res.append(r)
				states.append(len(agent))
				times.append(dt)
			if not self._isdeep():
				self.log.verbose(f"Performed evaluation at depth: {d}/{self.scrambling_depths[-1]}")

		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))
		states = np.reshape(states, (len(self.scrambling_depths), self.n_games))
		times = np.reshape(times, (len(self.scrambling_depths), self.n_games))

		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			self.log_this_depth(res[i], states[i], times[i], d)

		self.log.verbose(f"Evaluation runtime\n{self.tt}")

		return res, states, times

	def log_this_depth(self, res: np.ndarray, states: np.ndarray, times: np.ndarray, depth: int):
		"""Logs summary statistics for given depth

		:param res:  Vector of results
		:param states: Vector of seen states for each game
		:param times: Vector of runtimes for each game
		:param depth:  Scrambling depth at which results were generated
		"""
		share_completed = np.count_nonzero(res!=-1)*100/len(res)
		won_games = res[res!=-1]
		self.log(f"Scrambling depth {depth if depth else 'deep'}", with_timestamp=False)
		self.log(
			f"\tShare completed: {share_completed:.2f} % {bernoulli_error(share_completed/100, len(res), 0.05, stringify=True)} (approx. 95 % CI)",
			with_timestamp=False
		)
		if won_games.size:
			mean_turns = won_games.mean()
			median_turns = np.median(won_games)
			std_turns = won_games.std()
			self.log(
				f"\tTurns to win: {mean_turns:.2f} +/- {std_turns:.1f} (std.), Median: {median_turns:.0f}",
				with_timestamp=False
			)

		safe_times = times != 0
		states_per_sec = states[safe_times] / times[safe_times]
		self.log(
			f"\tStates seen: Pr. game: {states.mean():.2f} +/- {states.std():.0f} (std.), "\
			f"Pr. sec.: {states_per_sec.mean():.2f} +/- {states_per_sec.std():.0f} (std.)", with_timestamp=False)
		self.log(f"\tTime:  {times.mean():.2f} +/- {times.std():.2f} (std.)", with_timestamp=False)

	@staticmethod
	def _get_a_value(obj: dict):
		"""Returns a vaue from the object"""
		return obj[list(obj.keys())[0]]

	@staticmethod
	def check_equal_settings(eval_settings: dict):
		"""Super simple looper just to hide the ugliness"""
		games, times = list(), list()
		for setting in eval_settings.values():
			games.append(setting['max_time'])
			times.append(setting['n_games'])
		return games.count(games[0]) == len(games), times.count(times[0]) == len(times)
