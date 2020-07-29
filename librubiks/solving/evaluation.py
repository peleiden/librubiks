import os
from dataclasses import dataclass
from typing import List

import numpy as np

from pelutils import DataStorage

from librubiks import envs
from pelutils import NullLogger, Logger, TickTock, TimeUnit
from librubiks.solving.agents import Agent
from librubiks.analysis.statscompare import bernoulli_error

def _isdeep(scrambling_depths: np.ndarray) -> bool:
	scrambling_depths = np.array(scrambling_depths)
	return scrambling_depths.size == 1 and scrambling_depths[0] == 0


@dataclass
class EvalData(DataStorage):
	games: int
	scrambling_depths: List[int]
	max_time: float
	max_states: int
	agents: List[str]
	sol_lengths: np.ndarray
	times: np.ndarray
	states: np.ndarray

	subfolder = "eval-data"
	json_name = "eval-data.json"


class Evaluator:
	deep_depth = range(100, 1000)  # How much to scramble under deep evaluation

	def __init__(
			self,
			n_games: int,
			scrambling_depths: range or list,
			max_time: float = None,  # Max time to completion per game
			max_states: int = None,  # The max number of states to explore per game
			logger: Logger = NullLogger(),
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
			f"Scrambling depths: {scrambling_depths}" if _isdeep(self.scrambling_depths)
			else f"Uniformly sampled in [{self.deep_depth.start}, {self.deep_depth.stop - 1}]",
		]))

	def approximate_time(self):
		return self.max_time * self.n_games * len(self.scrambling_depths)

	def _eval_game(self, agent: Agent, depth: int, profile: str):
		turns_to_complete = -1  # -1 for unfinished
		state, _ = agent.env.scramble(depth, True)
		self.tt.profile(profile)
		solution_found = agent.search(state, self.max_time, self.max_states)
		dt = self.tt.end_profile(profile)
		if solution_found:
			turns_to_complete = len(agent.action_queue)
		return turns_to_complete, dt

	def eval(self, agents: List[Agent]) -> EvalData:
		"""
		Evaluates an agent
		Returns results which is an a len(self.scrambling_depths) x self.n_games matrix
		Each entry contains the number of steps needed to solve the scrambled state or -1 if not solved
		"""
		sollengths = []
		states_explored = []
		time_used = []
		for agent in agents:
			self.log.section(f"Evaluating {agent}")
			self.log("\n".join([
				f"{self.n_games * len(self.scrambling_depths)} cubes",
				f"Maximum solve time per cube is {TickTock.stringify_time(self.max_time, TimeUnit.second)} "
				f"and estimated total time <= {TickTock.stringify_time(self.approximate_time(), TimeUnit.minute)}"
				if self.max_time else "No time limit given",
				f"Maximum number of explored states is {TickTock.thousand_seps(self.max_states)}"
				if self.max_states else "No max states given",
			]))

			res = []
			states = []
			times = []
			for d in self.scrambling_depths:
				for _ in range(self.n_games):
					if _isdeep(self.scrambling_depths):  # Randomly sample evaluation depth for deep evaluations
						d = np.random.randint(self.deep_depth.start, self.deep_depth.stop)
					p = f"Evaluation of {agent} at depth [{self.deep_depth.start}, {self.deep_depth.stop - 1}]"\
						if _isdeep(self.scrambling_depths) else f"Evaluation of {agent} at depth {d}"
					r, dt = self._eval_game(agent, d, p)

					res.append(r)
					states.append(len(agent))
					times.append(dt)
				if not _isdeep(self.scrambling_depths):
					self.log.verbose(f"Performed evaluation at depth: {d}/{self.scrambling_depths[-1]}")

			res = np.reshape(res, (len(self.scrambling_depths), self.n_games))
			states = np.reshape(states, (len(self.scrambling_depths), self.n_games))
			times = np.reshape(times, (len(self.scrambling_depths), self.n_games))
			sollengths.append(res)
			states_explored.append(res)
			time_used.append(times)

			self.log(f"Evaluation results")
			for i, d in enumerate(self.scrambling_depths):
				self.log_this_depth(res[i], states[i], times[i], d)

			self.log.verbose(f"Evaluation runtime\n{self.tt}")

		data = EvalData(
			games = self.n_games,
			scrambling_depths = self.scrambling_depths.tolist(),
			max_time = self.max_time,
			max_states = self.max_states,
			agents = [str(agent) for agent in agents],
			sol_lengths = np.array(sollengths),
			times = np.array(time_used),
			states = np.array(states_explored),
		)

		return data

	def log_this_depth(self, res: np.ndarray, states: np.ndarray, times: np.ndarray, depth: int):
		"""
		Logs summary statistics for given depth

		:param res:  Vector of results
		:param states: Vector of seen states for each game
		:param times: Vector of runtimes for each game
		:param depth:  Scrambling depth at which results were generated
		"""

		pct_completed = np.count_nonzero(res != -1) * 100 / len(res)
		won_games = res[res != -1]
		self.log(f"Scrambling depth {depth if depth else 'deep'}", with_timestamp=False)
		self.log(
			f"\tShare completed: {pct_completed:.2f} % {bernoulli_error(pct_completed / 100, len(res), 0.05, stringify=True)} (approx. 95 % CI)",
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
			f"\tStates seen: Pr. game: {states.mean():.2f} +/- {states.std():.0f} (std.), " \
			f"Pr. sec.: {states_per_sec.mean():.2f} +/- {states_per_sec.std():.0f} (std.)", with_timestamp=False)
		self.log(f"\tTime:  {times.mean():.2f} +/- {times.std():.2f} (std.)", with_timestamp=False)
