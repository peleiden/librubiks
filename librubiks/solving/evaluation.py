import os

import numpy as np
from scipy import stats
import matplotlib.colors as mcolour
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 22})

from librubiks.utils import NullLogger, Logger, TickTock, bernoulli_error

from librubiks.solving import agents

from librubiks import cube


class Evaluator:
	def __init__(self,
			n_games,
			scrambling_depths: range or list,
			max_time = None,  # Max time to completion per game
			max_states = None,
			logger: Logger = NullLogger()
		):

		self.n_games = n_games
		self.max_time = max_time
		self.max_states = max_states

		self.tt = TickTock()
		self.log = logger
		self.scrambling_depths = np.array(scrambling_depths)

		self.log("\n".join([
			"Creating evaluator",
			f"Games per scrambling depth: {self.n_games}",
			f"Scrambling depths: {scrambling_depths}",
		]))

	def approximate_time(self):
		return self.max_time * self.n_games * len(self.scrambling_depths)

	def _eval_game(self, agent: agents.Agent, depth: int):
		turns_to_complete = -1  # -1 for unfinished
		state, _, _ = cube.scramble(depth, True)
		solution_found = agent.search(state, self.max_time, self.max_states)
		if solution_found: turns_to_complete = len(agent.action_queue)
		return turns_to_complete

	def eval(self, agent: agents.Agent):
		"""
		Evaluates an agent
		Returns results which is an a len(self.scrambling_depths) x self.n_games matrix
		Each entry contains the number of steps needed to solve the scrambled cube or -1 if not solved
		"""
		self.log.section(f"Evaluation of {agent}")
		self.log(f"{self.n_games*len(self.scrambling_depths)} games with max time per game {self.max_time}\nExpected time <~ {self.approximate_time()/60:.2f} min")

		res = []
		states = []
		times = []
		for d in self.scrambling_depths:
			for _ in range(self.n_games):
				self.tt.profile(f"Evaluation of {agent}. Depth {d}")
				r = self._eval_game(agent, d)
				t = self.tt.end_profile(f"Evaluation of {agent}. Depth {d}")

				res.append(r)
				states.append(len(agent))
				times.append(t)
			self.log.verbose(f"Performed evaluation at depth: {d}/{self.scrambling_depths[-1]}")

		res = np.reshape(res, (len(self.scrambling_depths), self.n_games))
		states = np.reshape(states, (len(self.scrambling_depths), self.n_games))
		times = np.reshape(times, (len(self.scrambling_depths), self.n_games))

		self.log(f"Evaluation results")
		for i, d in enumerate(self.scrambling_depths):
			self.log_this_depth(res[i], states[i], times[i], d)

		S_mu, S_conf = self.S_confidence(self.S_dist(res, self.scrambling_depths))
		self.log(f"S: {S_mu:.2f} p/m {S_conf:.2f}", with_timestamp=False)
		self.log.verbose(f"Evaluation runtime\n{self.tt}")

		return res, states

	def log_this_depth(self, res: np.ndarray, states: np.ndarray, times: np.ndarray, depth: int):
		"""Logs summary statistics for given deth

		:param res:  Vector of results
		:param states: Vector of seen states for each game
		:param times: Vector of runtimes for each game
		:param depth:  Scrambling depth at which results were generated
		"""
		share_completed = np.count_nonzero(res!=-1)*100/len(res)
		won_games = res[res!=-1]
		self.log(f"Scrambling depth {depth}", with_timestamp=False)
		self.log(f"\tShare completed: {share_completed:.2f} % {bernoulli_error(share_completed/100, len(res), 0.05, stringify=True)} (approx. 95 % CI)", with_timestamp=False)
		if won_games.size:
			mean_turns = won_games.mean()
			median_turns = np.median(won_games)
			std_turns = won_games.std()
			self.log(
				f"\tTurns to win: "\
				f"{mean_turns:.2f} +/- {std_turns:.1f} (std.), Median: {median_turns:.0f}"
				, with_timestamp=False
			)

		safe_times = times != 0
		states_per_sec = states[safe_times] / times[safe_times]
		self.log(
			f"\tStates seen: Pr. game: {states.mean():.2f} +/- {states.std():.0f} (std.), "\
			f"Pr. sec.: {states_per_sec.mean():.2f} +/- {states_per_sec.std():.0f} (std.)", with_timestamp=False)
		self.log(f"\tTime:  {times.mean():.2f} +/- {times.std():.2f} (std.)", with_timestamp=False)
	@staticmethod
	def S_dist(res: np.ndarray, scrambling_depths: np.ndarray) -> np.ndarray:
		"""
		Computes sum score game wise, that is it returns an array of length self.n_games
		It assumes that all srambling depths lower that self.scrambling_depths[0] are always solved
		and that all depths above self.scrambling_depths[-1] are never solved
		Overall S is the mean of the returned array
		:param res: Numpy array of evaluation results as returned by self.eval
		:param scrambling_depths: The scrambling depths used for the evaluation
		:return: Numpy array of length self.n_games
		"""
		solved = res != -1
		lower_depths = scrambling_depths[0] - 1
		return solved.sum(axis=0) + lower_depths

	@staticmethod
	def S_confidence(S_dist: np.ndarray, alpha=0.05):
		"""
		Calculates mean and confidence interval on a distribution of S values as given by Evaluator.S_dist
		:param S_dist: numpy array of S values
		:param alpha: confidence interval
		:return: mean and z * sigma
		"""
		mu = np.mean(S_dist)
		std = np.std(S_dist)
		z = stats.norm.ppf(1 - alpha / 2)
		return mu, z * std / np.sqrt(len(S_dist))

	def plot_this_eval(self, eval_results: dict, save_dir: str,  **kwargs):
		self.log("Creating plot of evaluation")
		settings = {
			'n_games': self.n_games,
			'max_time': self.max_time,
			'scrambling_depths': self.scrambling_depths
		}
		save_paths = self.plot_evaluators(eval_results, save_dir, [settings] * len(eval_results), **kwargs)
		self.log(f"Saved evaluation plots to {save_paths}")

	@staticmethod
	def plot_evaluators(eval_results: dict, save_dir: str, eval_settings: list, show: bool=False, title: str=''):
		"""
		{agent: results from eval}
		"""
		save_paths = []
		#depth, win%-graph
		games_equal, times_equal = Evaluator.check_equal_settings(eval_settings)
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		ax.set_ylabel(f"Percentage of {eval_settings[0]['n_games']} games won" if games_equal else "Percentage of games won")
		ax.set_xlabel(f"Scrambling depth: Number of random rotations applied to cubes")
		ax.locator_params(axis='x', integer=True, tight=True)

		tab_colours = list(mcolour.TABLEAU_COLORS)
		colours = [tab_colours[i%len(tab_colours)] for i in range(len(eval_results))]

		for i, (agent, results) in enumerate(eval_results.items()):
			used_settings = eval_settings[i]
			color = colours[i]
			win_percentages = (results != -1).mean(axis=1) * 100

			ax.plot(used_settings['scrambling_depths'], win_percentages, linestyle='dashdot', color=color)
			ax.scatter(used_settings['scrambling_depths'], win_percentages, color=color, label=f"Win % of {agent}")
		ax.legend()
		ax.set_ylim([-5, 105])
		ax.grid(True)
		ax.set_title(title if title else (f"Cubes solved in {eval_settings[0]['max_time']:.2f} seconds" if times_equal else "Cubes solved") )
		fig.tight_layout()

		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "eval_winrates.png")
		plt.savefig(path)
		save_paths.append(path)

		if show: plt.show()
		plt.clf()

		# solution length boxplots
		plt.rcParams.update({"font.size": 18})
		max_width = 2
		width = min(len(eval_results), max_width)
		height = (len(eval_results)+1) // width if width == max_width else 1
		positions = [(i, j) for i in range(height) for j in range(width)]
		fig, axes = plt.subplots(height, width, figsize=(width*10, height*6))

		max_sollength = 50
		agents, agent_results = list(zip(*eval_results.items()))
		agent_results = np.array(agent_results).copy()
		agent_results[agent_results > max_sollength] = max_sollength  # Clips outlier results to prevent skewing plots
		ylim = np.array([-0.02, 1.02]) * agent_results.max()
		min_, max_ = used_settings["scrambling_depths"].min(), used_settings["scrambling_depths"].max()
		xticks = np.arange(min_, max_+1, max(np.ceil((max_-min_+1)/8).astype(int), 1))
		for i, position in enumerate(positions):
			# Make sure axes are stored in a matrix, so they are easire to work with
			# Select axes object
			if len(eval_results) == 1:
				axes = np.array([[axes]])
			elif len(eval_results) <= width and i == 0:
				axes = np.expand_dims(axes, 0)
			ax = axes[position]
			if position[1] == 0:
				ax.set_ylabel(f"Solution length")
			if position[0] == height - 1 or len(eval_results) <= width:
				ax.set_xlabel(f"Scrambling depth")
			ax.locator_params(axis="y", integer=True, tight=True)

			try:
				agent, results = agents[i], agent_results[i]
				used_settings = eval_settings[i]
				ax.set_title(str(agent) if axes.size > 1 else "Solution lengths for " + str(agent))
				results = [depth[depth != -1] for depth in results]
				ax.boxplot(results)
				ax.grid(True)
			except IndexError:
				pass
			ax.set_ylim(ylim)
			ax.set_xlim([used_settings["scrambling_depths"].min()-1, used_settings["scrambling_depths"].max()+1])

		plt.setp(axes, xticks=xticks, xticklabels=[str(x) for x in xticks])
		plt.rcParams.update({"font.size": 22})
		if axes.size > 1:
			fig.suptitle("Solution lengths")
		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		os.makedirs(save_dir, exist_ok=True)
		path = os.path.join(save_dir, "eval_sollengths.png")
		plt.savefig(path)
		save_paths.append(path)

		if show: plt.show()
		plt.clf()

		# Histograms of S
		normal_pdf = lambda x, mu, sigma: np.exp(-1/2 * ((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		sss = np.array([Evaluator.S_dist(results, eval_settings[i]['scrambling_depths']) for i, results in enumerate(eval_results.values())])
		mus, confs = zip(*[Evaluator.S_confidence(ss) for ss in sss])
		stds = [ss.std() for ss in sss]
		lower, higher = sss.min() - 2, sss.max() + 2
		bins = np.arange(lower, higher+1)
		ax.hist(x			= sss.T,
				bins		= bins,
				density		= True,
				color		= colours,
				edgecolor	= "black",
				linewidth	= 2,
				align		= "left",
				label		= [f"{agent}: S = {mus[i]:.2f} p/m {confs[i]:.2f}" for i, agent in enumerate(eval_results.keys())])
		highest_y = 0
		for i in range(len(eval_results)):
			if stds[i] > 0:
				x = np.linspace(lower, higher, 500)
				y = normal_pdf(x, mus[i], stds[i])
				x = x[~np.isnan(y)]
				y = y[~np.isnan(y)]
				plt.plot(x, y, color="black", linewidth=9)
				plt.plot(x, y, color=colours[i], linewidth=5)
				highest_y = max(highest_y, y.max())
		ax.set_xlim([lower, higher])
		ax.set_xticks(bins)
		ax.set_title(f"Single game S distributions (evaluated on {eval_settings[0]['max_time']:.2f} s per game)" if times_equal else "Single game S distributions")
		ax.set_xlabel("S")
		ax.set_ylim([0, highest_y*(1+0.1*max(3, len(eval_results)))])  # To make room for labels
		ax.set_ylabel("Frequency")
		ax.legend(loc=2)
		path = os.path.join(save_dir, "eval_S.png")
		plt.savefig(path)
		save_paths.append(path)

		if show: plt.show()
		plt.clf()

		return save_paths

	@staticmethod
	def check_equal_settings(eval_settings: list):
		"""Super simple looper just to hide the ugliness from above function
		"""
		games, times = list(), list()
		for setting in eval_settings:
			games.append(setting['max_time'])
			times.append(setting['n_games'])
		return games.count(games[0]) == len(games), times.count(times[0]) == len(times)
