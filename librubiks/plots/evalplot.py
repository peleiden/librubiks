import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from librubiks.utils import Logger, NullLogger, TickTock, bernoulli_error
from librubiks.solving.evaluation import EvalData, Evaluator
from librubiks.plots.defaults import rc_params, rc_params_small, colours


def plot_depth_win(loc: str, data: EvalData, size: tuple) -> str:
	# depth, win%-graph
	fig, ax = plt.subplots(figsize=size)
	ax.set_ylabel(f"Percentage of {data.games} games won")
	ax.set_xlabel(f"Scrambling depth: Number of random rotations applied to cubes")
	ax.locator_params(axis='x', integer=True, tight=True)

	for i, sol_lengths in enumerate(data.sol_lengths):
		win_percentages = (sol_lengths != -1).mean(axis=1) * 100

		ax.plot(data.scrambling_depths, win_percentages, linestyle='dashdot', color=colours[i])
		ax.scatter(data.scrambling_depths, win_percentages, color=colours[i], label=colours[i])
	ax.legend()
	ax.set_ylim([-5, 105])
	ax.grid(True)
	ax.set_title(f"Percentage of cubes solved in {data.max_time} seconds")
	fig.tight_layout()

	path = os.path.join(loc, "eval_winrates.png")
	plt.savefig(path)
	plt.close()

	return path


def plot_sol_length_boxplots(loc: str, data: EvalData) -> str:
	# Solution length boxplots
	plt.rcParams.update(rc_params_small)
	max_width = 3
	width = min(len(data.agents), max_width)
	height = (len(data.agents)+1) // width if width == max_width else 1
	positions = [(i, j) for i in range(height) for j in range(width)]
	fig, axes = plt.subplots(height, width, figsize=(width*10, height*6))

	max_sollength = 50
	agent_results = tuple(x.copy() for x in data.sol_lengths)
	for res in agent_results:
		res[res > max_sollength] = max_sollength
	ylim = np.array([-0.02, 1.02]) * max([res.max() for res in agent_results])
	min_, max_ = data.scrambling_depths[0], data.scrambling_depths[-1]
	xticks = np.arange(min_, max_+1, max(np.ceil((max_-min_+1)/8).astype(int), 1))
	for i, position in enumerate(positions):
		# Make sure axes are stored in a matrix, so they are easire to work with, and select axes object
		if len(data.sol_lengths) == 1:
			axes = np.array([[axes]])
		elif len(data.sol_lengths) <= width and i == 0:
			axes = np.expand_dims(axes, 0)
		ax = axes[position]
		if position[1] == 0:
			ax.set_ylabel(f"Solution length")
		if position[0] == height - 1 or len(data.sol_lengths) <= width:
			ax.set_xlabel(f"Scrambling depth")
		ax.locator_params(axis="y", integer=True, tight=True)

		try:
			agent, results = data.agents[i], agent_results[i]
			assert type(agent) == str, str(type(agent))
			ax.set_title(agent if axes.size > 1 else "Solution lengths for " + agent)
			results = [depth[depth != -1] for depth in results]
			ax.boxplot(results)
			ax.grid(True)
		except IndexError:
			pass
		ax.set_ylim(ylim)
		ax.set_xlim([min(data.scrambling_depths)-1, max(data.scrambling_depths)+1])

	plt.setp(axes, xticks=xticks, xticklabels=[str(x) for x in xticks])
	plt.rcParams.update(rc_params)
	if axes.size > 1:
		fig.suptitle("Solution lengths")
	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	path = os.path.join(loc, "eval_sollengths.png")
	plt.savefig(path)
	plt.close()

	return path


def _get_depth_str(scrambling_depths: np.ndarray) -> str:
	if len(scrambling_depths) == 1 and scrambling_depths[0] != 0:
		return str(scrambling_depths[0])
	elif len(scrambling_depths) == 1:
		return f"{Evaluator.deep_depth.start} - {Evaluator.deep_depth.stop-1}"
	else:
		return f"{min(scrambling_depths)} - {max(scrambling_depths)}"


def plot_time_states_winrate(loc: str, data: EvalData, size: tuple, is_times: bool) -> str:
	# Make a (time spent, winrate) plot if is_times else (states explored, winrate)
	# Only done for the deepest configuration
	plt.figure(figsize=size)
	max_value = 0
	times_or_states = data.times if is_times else data.states
	for agent, res, values, colour in zip(data.agents, data.sol_lengths, times_or_states, colours):
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
	depth = _get_depth_str(data.scrambling_depths)
	plt.title(f"Winrate against {'time used for' if is_times else 'states seen during'} solving at depth {depth}")
	plt.grid(True)
	plt.tight_layout()
	path = os.path.join(loc, "time_winrate.png" if is_times else "states_winrate.png")
	plt.savefig(path)
	plt.close()

	return path


def plot_distributions(loc: str, data: EvalData, size: tuple) -> List[str]:
	"""Histograms of solution length, time used, and states explored for won games"""

	normal_pdf = lambda x, mu, sigma: np.exp(-1/2 * ((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))

	won_games    = { agent: (res != -1).ravel() for agent, res in zip(data.agents, data.sol_lengths) }
	if all(w.sum() <= 1 for w in won_games.values()):
		return "ERROR"
	eval_results = { agent: res.ravel()[won_games[agent]]    for agent, res    in zip(data.agents, data.sol_lengths) if won_games[agent].sum() > 1 }
	eval_times   = { agent: times.ravel()[won_games[agent]]  for agent, times  in zip(data.agents, data.times)       if won_games[agent].sum() > 1 }
	eval_states  = { agent: states.ravel()[won_games[agent]] for agent, states in zip(data.agents, data.states)      if won_games[agent].sum() > 1 }

	eval_data    = [eval_results, eval_times, eval_states]
	x_labels     = ["Solution length", "Time used [s]", "States seen"]
	titles       = ["Distribution of solution lengths for solved cubes",
					"Distribution of time used for solved cubes",
					"Distribution of states seen for solved cubes"]
	paths        = [os.path.join(loc, x) + ".png" for x in ["solve_length_dist", "time_dist", "state_dist"]]
	paths_iter   = iter(paths)

	for stat_data, xlab, title in zip(eval_data, x_labels, titles):
		plt.figure(figsize=size)
		agents = list(stat_data.keys())
		values = [stat_data[agent] for agent in agents]
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
			for i in range(len(stat_data)):
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
		plt.title(f"{title} at depth {_get_depth_str(data.scrambling_depths)}")
		plt.legend()
		plt.savefig(next(paths_iter))
		plt.close()

	return paths


