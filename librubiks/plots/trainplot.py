import os
import numpy as np
import matplotlib.pyplot as plt

from librubiks.utils import TickTock, bernoulli_error
from librubiks.train import TrainData
from librubiks.plots.defaults import rc_params, all_colours

plt.rcParams.update(rc_params)


def plot_training(loc: str, data: TrainData, size: tuple) -> str:
	"""
	Visualizes training by showing training loss + evaluation reward in same plot
	"""
	fig, loss_ax = plt.subplots(figsize=size)

	colour = "tab:blue"
	loss_ax.set_ylabel("Training loss")
	loss_ax.plot(np.arange(data.rollouts), data.losses, linewidth=3, color=colour, label="Training loss")
	loss_ax.tick_params(axis='y', labelcolor=colour)
	loss_ax.set_xlabel(f"Rollout, each of {TickTock.thousand_seps(data.states_per_rollout)} states")
	loss_ax.set_ylim(np.array([-0.05*1.35, 1.35]) * data.losses.max())
	h1, l1 = loss_ax.get_legend_handles_labels()
	if data.evaluation_rollouts.size:
		color = 'tab:orange'
		reward_ax = loss_ax.twinx()
		reward_ax.set_ylim([-5, 105])
		reward_ax.set_ylabel("Solve rate (~95 % CI) [%]")
		bernoulli_errors = bernoulli_error(data.evaluation_results, data.eval_games, alpha=0.05)
		reward_ax.errorbar(data.evaluation_rollouts, data.evaluation_results*100, bernoulli_errors*100, fmt="-o",
			capsize=10, color=color, label="Policy performance", errorevery=2, alpha=0.8)
		reward_ax.tick_params(axis='y', labelcolor=color)
		h2, l2 = reward_ax.get_legend_handles_labels()
		h1 += h2
		l1 += l2
	loss_ax.legend(h1, l1, loc=2)

	title = (f"Training - {TickTock.thousand_seps(data.rollouts*data.states_per_rollout)} states")
	plt.title(title)
	fig.tight_layout()
	plt.grid(True)

	path = os.path.join(loc, f"training_{data.name}.png")
	plt.savefig(path)
	plt.close()

	return path

