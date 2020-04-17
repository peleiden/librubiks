import numpy as np

from src.rubiks.cube.cube import Cube
from src.rubiks.solving.search import Searcher, DeepSearcher


class Agent:

	def __init__(self, searcher: Searcher):
		"""
		time_limit: Number of seconds that the tree search part of the algorithm is allowed to searc
		"""
		self._searcher = searcher

	def generate_action_queue(self, state: np.ndarray, time_limit: float) -> (bool, int):
		solution_found = self._searcher.search(state, time_limit)
		return solution_found, len(self._searcher.action_queue)

	def action(self) -> (int, bool):
		return Cube.action_space[self._searcher.action_queue.popleft()]

	def allow_mt(self):
		# NN based agents see very little gain but much higher compute usage with standard mt implementation
		return self._searcher.with_mt

	def __str__(self):
		return str(self._searcher)
	
	def __len__(self):
		# Returns number of explored states
		# This will not work properly with searchers using multiple workers if they do not maintain a states object
		return len(self._searcher.states) if hasattr(self._searcher, "states") else len(self._searcher.action_queue)
		

class DeepAgent(Agent):
	def __init__(self, searcher: DeepSearcher):
		super().__init__(searcher)

	def update_net(self, net):
		self._searcher.net = net



