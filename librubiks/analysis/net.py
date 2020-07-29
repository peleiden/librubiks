import matplotlib.pyplot as plt
import numpy as np
import torch

from librubiks import gpu, no_grad
from librubiks import envs
from librubiks.model import Model, load_net
from pelutils import TickTock, Logger

tt = TickTock()
log = Logger("data/local_analyses/net.log", "Analyzing MCTS")
net = load_net("data/local_method_comparison/asgerfix").eval().to(gpu)
env = net.env


def _get_adi_ff_slices(b, n):
	slice_size = n // b + 1
	# Final slice may have overflow, however this is simply ignored when indexing
	slices = [slice(i * slice_size, (i + 1) * slice_size) for i in range(b)]
	return slices


def _ff(oh_states):
	batches = 1
	while True:
		try:
			value_parts = [net(oh_states[slice_]).squeeze() for slice_ in _get_adi_ff_slices(batches, len(oh_states))]
			values = torch.cat(value_parts).cpu()
			break
		except RuntimeError as e:  # Usually caused by running out of vram. If not, the error is still raised, else batch size is reduced
			if "alloc" not in str(e):
				raise e
			batches *= 2
	return values


@no_grad
def value(n, d):
	depths = np.tile(np.arange(1, d+1), n).reshape(n, d)
	states, states_oh = env.sequence_scrambler(n, d, False)
	values = _ff(states_oh).squeeze().numpy().reshape(n, d)
	plt.plot(depths.T, values.T, "o-")
	plt.grid(True)
	plt.show()


if __name__ == "__main__":
	value(10, 50)
