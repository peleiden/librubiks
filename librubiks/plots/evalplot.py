import numpy as np
import matplotlib.colors as mcolour
import matplotlib.pyplot as plt

from librubiks import rc_params, rc_params_small
from librubiks.utils import Logger, NullLogger, TickTock, bernoulli_error
from librubiks.train import TrainData

plt.rcParams.update(rc_params)
base_colours = list(mcolour.BASE_COLORS)
tab_colours = list(mcolour.TABLEAU_COLORS)
all_colours = base_colours[:-1] + tab_colours[:-2]

class EvalPlot:

	def __init__(self, logger = NullLogger()):
		self.log = logger

	@classmethod
	def plot_evaluators(cls, eval_results: dict, eval_states: dict, eval_times: dict, eval_settings: dict, save_dir: str, title: str='') -> list:
		"""
		Plots evaluation results
		:param eval_results:   { agent name: [steps to solve, -1 for unfinished] }
		:param eval_states:    { agent name: [states seen during solving] }
		:param eval_times:     { agent name: [time spent solving] }
		:param eval_settings:  { agent name: { 'n_games': int, 'max_time': float, 'max_states': int, 'scrambling_depths': np.ndarray } }
		:param save_dir:       Directory in which to save plots
		:param title:          If given, overrides auto generated title in (depth, winrate) plot
		:return:               Locations of saved plots
		"""
		assert eval_results.keys() == eval_results.keys() == eval_times.keys() == eval_settings.keys(), "Keys of evaluation dictionaries should match"
		os.makedirs(save_dir, exist_ok=True)

		tab_colours = list(mcolour.TABLEAU_COLORS)
		colours = [tab_colours[i%len(tab_colours)] for i in range(len(eval_results))]

		save_paths = [
			cls._plot_depth_win(eval_results, save_dir, eval_settings, colours, title),
			cls._sol_length_boxplots(eval_results, save_dir, eval_settings, colours),
		]
		# Only plot (time, winrate), (states, winrate), and their distributions if settings are the same
		if all(cls.check_equal_settings(eval_settings)):
			d = cls._get_a_value(eval_settings)["scrambling_depths"][-1]
			save_paths.extend([
				cls._time_states_winrate_plot(eval_results, eval_times, True, d, save_dir, eval_settings, colours),
				cls._time_states_winrate_plot(eval_results, eval_states, False, d, save_dir, eval_settings, colours),
			])
			p = cls._distribution_plots(eval_results, eval_times, eval_states, d, save_dir, eval_settings, colours)
			if p != "ERROR":
				save_paths.extend(p)

		return save_paths
	
	@classmethod
	def _plot_depth_win(cls, eval_results: dict, save_dir: str, eval_settings: dict, colours: list, title: str='') -> str:
		# depth, win%-graph
		games_equal, times_equal = cls.check_equal_settings(eval_settings)
		fig, ax = plt.subplots(figsize=(19.2, 10.8))
		ax.set_ylabel(f"Percentage of {cls._get_a_value(eval_settings)['n_games']} games won" if games_equal else "Percentage of games won")
		ax.set_xlabel(f"Scrambling depth: Number of random rotations applied to cubes")
		ax.locator_params(axis='x', integer=True, tight=True)

		for i, (agent, results) in enumerate(eval_results.items()):
			used_settings = eval_settings[agent]
			color = colours[i]
			win_percentages = (results != -1).mean(axis=1) * 100

			ax.plot(used_settings['scrambling_depths'], win_percentages, linestyle='dashdot', color=color)
			ax.scatter(used_settings['scrambling_depths'], win_percentages, color=color, label=agent)
		ax.legend()
		ax.set_ylim([-5, 105])
		ax.grid(True)
		ax.set_title(title if title else (f"Percentage of cubes solved in {cls._get_a_value(eval_settings)['max_time']:.2f} seconds" if times_equal else "Cubes solved"))
		fig.tight_layout()

		path = os.path.join(save_dir, "eval_winrates.png")
		plt.savefig(path)
		plt.clf()

		return path

	@classmethod
	def _sol_length_boxplots(cls, eval_results: dict, save_dir: str, eval_settings: dict, colours: list) -> str:
		# Solution length boxplots
		plt.rcParams.update(rc_params_small)
		max_width = 2
		width = min(len(eval_results), max_width)
		height = (len(eval_results)+1) // width if width == max_width else 1
		positions = [(i, j) for i in range(height) for j in range(width)]
		fig, axes = plt.subplots(height, width, figsize=(width*10, height*6))

		max_sollength = 50
		agents, agent_results = list(zip(*eval_results.items()))
		agent_results = tuple(x.copy() for x in agent_results)
		for res in agent_results:
			res[res > max_sollength] = max_sollength
		ylim = np.array([-0.02, 1.02]) * max([res.max() for res in agent_results])
		min_ = min([x["scrambling_depths"][0] for x in eval_settings.values()])
		max_ = max([x["scrambling_depths"][-1] for x in eval_settings.values()])
		xticks = np.arange(min_, max_+1, max(np.ceil((max_-min_+1)/8).astype(int), 1))
		for used_settings, (i, position) in zip(eval_settings.values(), enumerate(positions)):
			# Make sure axes are stored in a matrix, so they are easire to work with, and select axes object
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
				assert type(agent) == str, str(type(agent))
				ax.set_title(agent if axes.size > 1 else "Solution lengths for " + agent)
				results = [depth[depth != -1] for depth in results]
				ax.boxplot(results)
				ax.grid(True)
			except IndexError:
				pass
			ax.set_ylim(ylim)
			ax.set_xlim([used_settings["scrambling_depths"].min()-1, used_settings["scrambling_depths"].max()+1])

		plt.setp(axes, xticks=xticks, xticklabels=[str(x) for x in xticks])
		plt.rcParams.update(rc_params)
		if axes.size > 1:
			fig.suptitle("Solution lengths")
		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		path = os.path.join(save_dir, "eval_sollengths.png")
		plt.savefig(path)
		plt.clf()

		return path

	@classmethod
	def _time_states_winrate_plot(cls, eval_results: dict, eval_times_or_states: dict, is_times: bool,
	                              depth: int, save_dir: str, eval_settings: dict, colours: list) -> str:
		# Make a (time spent, winrate) plot if is_times else (states explored, winrate)
		# Only done for the deepest configuration
		plt.figure(figsize=(19.2, 10.8))
		max_value = 0
		for (agent, res), values, colour in zip(eval_results.items(), eval_times_or_states.values(), colours):
			sort_idcs = np.argsort(values.ravel())  # Use values from all different depths - mainly for deep evaluation
			wins, values = (res != -1).ravel()[sort_idcs], values.ravel()[sort_idcs]
			max_value = max(max_value, values.max())
			cumulative_winrate = np.cumsum(wins) / len(wins) * 100
			plt.plot(values, cumulative_winrate, "o-", linewidth=3, color=colour, label=agent)
		plt.xlabel("Time used [s]" if is_times else "States explored")
		plt.ylabel("Winrate [%]")
		plt.xlim([-0.05*max_value, 1.05*max_value])
		plt.ylim([-5, 105])
		plt.legend()
		plt.title(f"Winrate against {'time used for' if is_times else 'states seen during'} solving at depth {depth if depth else '100 - 999'}")
		plt.grid(True)
		plt.tight_layout()
		path = os.path.join(save_dir, "time_winrate.png" if is_times else "states_winrate.png")
		plt.savefig(path)
		plt.clf()
		
		return path
		
	@classmethod
	def _distribution_plots(cls, eval_results: dict, eval_times: dict, eval_states: dict, depth: int,
	                        save_dir: str, eval_settings: dict, colours: list) -> str:
		"""Histograms of solution length, time used, and states explored for won games"""

		normal_pdf = lambda x, mu, sigma: np.exp(-1/2 * ((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

		won_games    = { agent: (res != -1).ravel() for agent, res in eval_results.items() }
		if all(w.sum() <= 1 for w in won_games.values()):
			return "ERROR"
		eval_results = { agent: res.ravel()[won_games[agent]]    for agent, res    in eval_results.items() if won_games[agent].sum() > 1 }
		eval_times   = { agent: times.ravel()[won_games[agent]]  for agent, times  in eval_times.items()   if won_games[agent].sum() > 1 }
		eval_states  = { agent: states.ravel()[won_games[agent]] for agent, states in eval_states.items()  if won_games[agent].sum() > 1 }

		eval_data    = [eval_results, eval_times, eval_states]
		x_labels     = ["Solution length", "Time used [s]", "States seen"]
		titles       = ["Distribution of solution lengths for solved cubes",
		                "Distribution of time used for solved cubes",
						"Distribution of states seen for solved cubes"]
		paths        = [os.path.join(save_dir, x) + ".png" for x in ["solve_length_dist", "time_dist", "state_dist"]]
		paths_iter   = iter(paths)

		for data, xlab, title, path in zip(eval_data, x_labels, titles, paths):
			plt.figure(figsize=(19.2, 10.8))
			agents = list(data.keys())
			values = [data[agent] for agent in agents]
			apply_to_values = lambda fun: fun([fun(v) for v in values])
			mus, sigmas = np.array([v.mean() for v in values]), np.array([v.std() for v in values])
			min_, max_ = apply_to_values(np.min), apply_to_values(np.max)
			if xlab == "Solution length":
				lower, higher = min_ - 2, max_ + 2
			else:
				lower = min_ - (max_ - min_) * 0.1
				higher = max_ + (max_ - min_) * 0.1
			highest_y = 0
			for i, (agent, v) in enumerate(zip(agents, values)):
				bins = np.arange(lower, higher+1) if xlab == "Solution length" else int(np.sqrt(len(v))*2) + 1
				heights, _, _ = plt.hist(x=v, bins=bins, density=True, color=colours[i], edgecolor="black", linewidth=2,
				                         alpha=0.5, align="left" if xlab == "Solution length" else "mid", label=f"{agent}: {mus[i]:.2f}")
				highest_y = max(highest_y, np.max(heights))
			if xlab == "Solution length":
				for i in range(len(data)):
					if sigmas[i] > 0:
						x = np.linspace(lower, higher, 1000)
						y = normal_pdf(x, mus[i], sigmas[i])
						x = x[~np.isnan(y)]
						y = y[~np.isnan(y)]
						plt.plot(x, y, color="black", linewidth=9)
						plt.plot(x, y, color=colours[i], linewidth=5)
						highest_y = max(highest_y, y.max())
			plt.xlim([lower, higher])
			plt.ylim([0, highest_y*(1+0.1*max(3, len(eval_results)))])  # To make room for labels
			plt.xlabel(xlab)
			plt.ylabel("Frequency")
			plt.title(f"{title} at depth {depth if depth else '100 - 999'}")
			plt.legend()
			plt.savefig(next(paths_iter))
			plt.clf()

		return paths


