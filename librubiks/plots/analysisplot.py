
import os
import numpy as np
import matplotlib.pyplot as plt

from librubiks.envs import get_env
from pelutils import TickTock
from librubiks.analysis import AnalysisData
from librubiks.plots.defaults import rc_params, colours

try:
    import networkx
    import imageio
    has_image_tools = True
except ModuleNotFoundError:
    has_image_tools = False


def plot_substate_distributions(loc: str, data: AnalysisData, size: tuple) -> str:
    colour = "tab:blue"

    plt.figure(figsize=size)
    plt.plot(data.substate_val_stds, linestyle="dashdot", color=colour, label="Std. for ADI for cubes")
    plt.xlabel("Rollout number")
    plt.ylabel("Rollout mean std.")
    plt.tick_params(axis="y", labelcolor=colour)
    plt.legend()
    plt.tight_layout()
    plt.title(f"Analysis of substate distributions over time")
    plt.grid(True)

    path = os.path.join(loc, "substate_dists.png")
    plt.savefig(path)
    plt.close()

    return path


def visualize_first_states(loc: str, data: AnalysisData, size: tuple) -> str:
    if not has_image_tools:
        return "Visualizaiton of first state values could not be saved: Install imageio and networkx to do this"
    
    env = get_env(data.env_key)
    gif_frames = []

    # Build graph structure
    G = networkx.DiGraph()
    edge_labels = {}
    G.add_nodes_from(range(data.first_state_values.shape[1]))
    positions = {0: (50, 85)}
    label_positions = {0: (50, 80)}
    for i in range(env.action_dim):
        x_pos = 100 * ( i / (env.action_dim - 1) )
        positions[i+1] = (x_pos, 5)
        label_positions[i+1] = (x_pos, 12.5)

    for action in env.action_space:
        G.add_edge(0, i+1)
        edge_labels[(0, i+1)] =	env.action_names[action]

    fig = plt.figure(figsize=size)
    for i, values in enumerate(data.first_state_values):

        plt.title(f"Values at rollout: {data.evaluations[i]}")

        labels = {j: f"{float(val):.2f}" for j, val in enumerate(values)}
        colors = [float(val) for val in values]  # Don't ask
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

    if len(gif_frames) > 3:
        gif_frames.extend(gif_frames[-1] for i in range(10))  # Hacky way to pause gif at end
    savepath = os.path.join(loc, "value_development.gif")
    imageio.mimsave(savepath, gif_frames, format='GIF', duration=0.25)
    return savepath


def _get_evaluations_for_value(data: AnalysisData) -> np.ndarray:
    """
    Returns a boolean vector of length len(self.data.evaluations) containing whether or not the curve should be in focus
    """
    focus_rollouts = np.zeros(len(data.evaluations), dtype=bool)
    if len(data.evaluations) > 15:
        early_rollouts = 5
        late_rollouts = 10
        early_indices = [0, *np.unique(np.round(np.logspace(0, np.log10(data.extra_evals*2/3), early_rollouts-1)).astype(int))]
        late_indices = np.unique(np.linspace(data.extra_evals, len(data.evaluations)-1, late_rollouts, dtype=int))
        focus_rollouts[early_indices] = True
        focus_rollouts[late_indices] = True
    else:
        focus_rollouts[...] = True
    return focus_rollouts


def plot_value_targets(loc: str, data: AnalysisData, size: tuple) -> str:
    if not data.evaluations.size:
        return

    plt.figure(figsize=size)
    focus_rollouts = _get_evaluations_for_value(data)
    colours_iter = iter(colours)
    filter_by_bools = lambda list_, bools: [x for x, b in zip(list_, bools) if b]
    for target, rollout in zip(filter_by_bools(data.avg_value_targets, ~focus_rollouts), filter_by_bools(data.evaluations, ~focus_rollouts)):
        plt.plot(data.depths + (data.reward_method != "lapanfix"), target, "--", color="grey", alpha=.4)
    for target, rollout in zip(filter_by_bools(data.avg_value_targets, focus_rollouts), filter_by_bools(data.evaluations, focus_rollouts)):
        plt.plot(data.depths + (data.reward_method != "lapanfix"), target, linewidth=3, color=next(colours_iter), label=f"{rollout+1} Rollouts")
    plt.legend(loc=1)
    plt.xlim(np.array([-.05, 1.05]) * (data.depths[-1]+1))
    plt.xlabel("Scrambling depth")
    plt.ylabel("Average target value")
    plt.title("Average target value")
    path = os.path.join(loc, "avg_target_values.png")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    return path


def plot_net_changes(loc: str, data: AnalysisData, size: tuple) -> str:
    # This method is rather useless and is disabled for now
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

