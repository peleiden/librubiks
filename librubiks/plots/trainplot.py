import os
import numpy as np
import matplotlib.colors as mcolour
import matplotlib.pyplot as plt

from librubiks import rc_params
from librubiks.utils import Logger, NullLogger, TickTock, bernoulli_error
from librubiks.train import TrainData

try:
	import networkx
	import imageio
	has_image_tools = True
except ModuleNotFoundError:
	has_image_tools = False

plt.rcParams.update(rc_params)
base_colours = list(mcolour.BASE_COLORS)
tab_colours = list(mcolour.TABLEAU_COLORS)
all_colours = base_colours[:-1] + tab_colours[:-2]


def plot_training(self, save_dir: str, name: str):
	# FIXME: Load a traindata
	"""
	Visualizes training by showing training loss + evaluation reward in same plot
	"""
	self.log("Making plot of training")
	fig, loss_ax = plt.subplots(figsize=(23, 10))

	colour = "red"
	loss_ax.set_ylabel("Training loss")
	loss_ax.plot(self.train_rollouts, self.train_losses,  linewidth=3,                        color=colour,   label="Training loss")
	loss_ax.plot(self.train_rollouts, self.policy_losses, linewidth=2, linestyle="dashdot",   color="orange", label="Policy loss")
	loss_ax.plot(self.train_rollouts, self.value_losses,  linewidth=2, linestyle="dashed",    color="green",  label="Value loss")
	loss_ax.tick_params(axis='y', labelcolor=colour)
	loss_ax.set_xlabel(f"Rollout, each of {TickTock.thousand_seps(self.states_per_rollout)} states")
	loss_ax.set_ylim(np.array([-0.05*1.35, 1.35]) * self.train_losses.max())
	h1, l1 = loss_ax.get_legend_handles_labels()

	if len(self.evaluation_rollouts):
		color = 'blue'
		reward_ax = loss_ax.twinx()
		reward_ax.set_ylim([-5, 105])
		reward_ax.set_ylabel("Solve rate (~95 % CI) [%]")
		sol_shares = np.array(self.sol_percents)
		bernoulli_errors = bernoulli_error(sol_shares, self.evaluator.n_games, alpha=0.05)
		reward_ax.errorbar(self.evaluation_rollouts, sol_shares*100, bernoulli_errors*100, fmt="-o",
			capsize=10, color=color, label="Policy performance", errorevery=2, alpha=0.8)
		reward_ax.tick_params(axis='y', labelcolor=color)
		h2, l2 = reward_ax.get_legend_handles_labels()
		h1 += h2
		l1 += l2
	loss_ax.legend(h1, l1, loc=2)

	title = (f"Training - {TickTock.thousand_seps(self.rollouts*self.rollout_games*self.rollout_depth)} states")
	plt.title(title)
	fig.tight_layout()
	plt.grid(True)

	os.makedirs(save_dir, exist_ok=True)
	path = os.path.join(save_dir, f"training_{name}.png")
	plt.savefig(path)
	self.log(f"Saved loss and evaluation plot to {path}")

	plt.close()

# ANALYSIS PLOTS
def plot_substate_distributions(self, loc: str):
	# TODO: Remove policy from this plot
	self.log("Making plot of policy entropy and ADI value stds")

	fig, entropy_ax = plt.subplots(figsize=(19.2, 10.8))
	entropy_ax.set_xlabel(f"Rollout number")

	colour = "red"
	entropy_ax.set_ylabel(f"Rollout mean Shannon entropy", color=colour)
	entropy_ax.plot(self.policy_entropies, linestyle="dashdot", label="Entropy of training policy output for cubes", color=colour)
	entropy_ax.tick_params(axis='y', labelcolor = colour)
	h1, l1 = entropy_ax.get_legend_handles_labels()

	colour = 'blue'
	std_ax = entropy_ax.twinx()
	std_ax.set_ylabel(f"Rollout mean std.", color=colour)
	std_ax.plot(self.substate_val_stds, linestyle="dashdot", color=colour, label="Std. for ADI substates for cubes")
	std_ax.tick_params(axis='y', labelcolor=colour)

	h2, l2 = std_ax.get_legend_handles_labels()

	entropy_ax.legend(h1+h2, l1+l2)

	fig.tight_layout()
	plt.title(f"Analysis of substate distributions over time")
	plt.grid(True)

	path = os.path.join(loc, "substate_dists.png")
	plt.savefig(path)
	plt.close()

	self.log(f"Saved substate probability plot to {path}")

def visualize_first_states(self, loc: str):
	if has_image_tools and self.evaluations.size:
		self.log("Making visualization of first state values")
		gif_frames = []

		# Build graph structure
		G = networkx.DiGraph()
		edge_labels = {}
		G.add_nodes_from(range(len(self.first_state_values[0])))
		positions = {0: (50, 85)}
		label_positions = {0: (50, 80)}
		# Labels must be
		for i in range(cube.action_dim):
			x_pos = 100*( i / (cube.action_dim - 1) )
			positions[i+1] = (x_pos, 5)
			label_positions[i+1] = (x_pos, 12.5)

		for i, (face, pos) in enumerate(cube.action_space):
			G.add_edge(0, i+1)
			edge_labels[(0, i+1)] =	cube.action_names[face].lower() if pos else cube.action_names[face].upper()



		fig = plt.figure(figsize=(10, 7.5))
		for i, values in enumerate(self.first_state_values):

			plt.title(f"Values at rollout:  {self.evaluations[i]}")

			labels = {j: f"{float(val):.2f}" for j, val in enumerate(values)}
			colors = [float(val) for val in values] #Don't ask
			networkx.draw(G, pos=positions, alpha=0.8, node_size=1000, \
					cmap = plt.get_cmap('cool'), node_color=colors, vmin=-1, vmax=1.5)

			networkx.draw_networkx_labels(G, pos=label_positions, labels=labels, font_size = 15)
			networkx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels,\
					font_size = 22, label_pos=0.25)

			plt.axis('off')
			fig.tight_layout()
			# https://stackoverflow.com/a/57988387, but is there an easier way?
			fig.canvas.draw()
			image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
			image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			gif_frames.append(image_from_plot)

			plt.close()

		if len(gif_frames) > 3: gif_frames.extend(gif_frames[-1] for i in range(10)) # Hacky way to pause gif at end
		savepath = os.path.join(loc, "value_development.gif")
		imageio.mimsave(savepath, gif_frames, format='GIF', duration=0.25)
		self.log(f"Saved visualizations of first state values to {savepath}")
	elif not has_image_tools:
		self.log(f"Visualizaiton of first state values could not be saved: Install imageio and networkx to do this")

def plot_value_targets(self, loc: str):
	if not len(self.evaluations): return
	self.log("Plotting average value targets")
	plt.figure(figsize=(19.2, 10.8))
	focus_rollouts = self._get_evaluations_for_value()
	colours = iter(all_colours)
	filter_by_bools = lambda list_, bools: [x for x, b in zip(list_, bools) if b]
	for target, rollout in zip(filter_by_bools(self.avg_value_targets, ~focus_rollouts), filter_by_bools(self.evaluations, ~focus_rollouts)):
		plt.plot(self.depths + (self.reward_method != "lapanfix"), target, "--", color="grey", alpha=.4)
	for target, rollout in zip(filter_by_bools(self.avg_value_targets, focus_rollouts), filter_by_bools(self.evaluations, focus_rollouts)):
		plt.plot(self.depths + (self.reward_method != "lapanfix"), target, linewidth=3, color=next(colours), label=f"{rollout+1} Rollouts")
	plt.legend(loc=1)
	plt.xlim(np.array([-.05, 1.05]) * (self.depths[-1]+1))
	plt.xlabel("Scrambling depth")
	plt.ylabel("Average target value")
	plt.title("Average target value")
	path = os.path.join(loc, "avg_target_values.png")
	plt.grid(True)
	plt.savefig(path)
	plt.close()
	self.log(f"Saved value target plot to {path}")

def plot_net_changes(self, loc: str):
	self.log("Plotting changes to network parameters")
	plt.figure(figsize=(19.2, 10.8))
	plt.plot(self.evaluations, np.cumsum(self.param_changes), label="Cumulative change in network parameters")
	plt.plot(self.evaluations, self.param_total_changes, linestyle="dashdot", label="Change in parameters since original network")
	plt.legend(loc=2)
	plt.xlabel(f"Rollout number")
	plt.ylabel("Euclidian distance")
	plt.grid(True)
	path = os.path.join(loc, "parameter_changes.png")
	plt.savefig(path)
	plt.close()
	self.log(f"Saved network change plot to {path}")

