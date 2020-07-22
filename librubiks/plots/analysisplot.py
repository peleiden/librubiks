
import os
import numpy as np
import matplotlib.pyplot as plt

from librubiks.utils import TickTock
from librubiks.analysis import AnalysisData
from librubiks.plots.defaults import rc_params, all_colours

try:
	import networkx
	import imageio
	has_image_tools = True
except ModuleNotFoundError:
	has_image_tools = False


def plot_substate_distributions(loc: str, data: AnalysisData, size: tuple) -> str:
	self.log("Making plot of policy entropy and ADI value stds")

	fig, entropy_ax = plt.subplots(figsize=size)
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

	return path


def visualize_first_states(loc: str, data: AnalysisData, size: tuple) -> str:
	if not has_image_tools:
		return "Visualizaiton of first state values could not be saved: Install imageio and networkx to do this"
	
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



	fig = plt.figure(figsize=size)
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
	return savepath


def plot_value_targets(loc: str, data: AnalysisData, size: tuple) -> str:
	if not len(self.evaluations):
		return
	
	plt.figure(figsize=size)
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
	
	return path


def plot_net_changes(loc: str, data: AnalysisData, size: tuple) -> str:
	# TODO: Consider making this a 3D plot with the three most important axes from PCA
	plt.figure(figsize=size)
	plt.plot(self.evaluations, np.cumsum(self.param_changes), label="Cumulative change in network parameters")
	plt.plot(self.evaluations, self.param_total_changes, linestyle="dashdot", label="Change in parameters since original network")
	plt.legend(loc=2)
	plt.xlabel(f"Rollout number")
	plt.ylabel("Euclidian distance")
	plt.grid(True)
	path = os.path.join(loc, "parameter_changes.png")
	plt.savefig(path)
	plt.close()

	return path

