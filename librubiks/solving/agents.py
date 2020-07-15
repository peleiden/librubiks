from abc import ABC
from collections import deque
import heapq

import numpy as np

from librubiks import gpu, no_grad, envs
from librubiks.utils import TickTock
from librubiks.model import Model, load_net


class Agent(ABC):
	eps = np.finfo("float").eps
	_explored_states = 0

	def __init__(self, env: envs.Environment):
		self.env = env
		self.action_queue = deque()
		self.tt = TickTock()

	@no_grad
	def search(self, state: np.ndarray, time_limit: float = None, max_states: int = None) -> bool:
		# Returns whether a path was found and generates action queue
		# Implement _step method for agents that look one step ahead, otherwise overwrite this method
		time_limit, max_states = self.reset(time_limit, max_states)
		self.tt.tick()

		if self.env.is_solved(state): return True
		while self.tt.tock() < time_limit and len(self) < max_states:
			action, state, solution_found, states = self._step(state)
			self._explored_states += states
			self.action_queue.append(action)
			if solution_found:
				return True

		self._explored_states = len(self.action_queue)
		return False

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool, int):
		"""
		Takes a step given a state
		:param state: numpy array containing a state
		:return: Action index, new state, is solved, states explored
		"""
		raise NotImplementedError

	def reset(self, time_limit: float, max_states: int):
		self._explored_states = 0
		self.action_queue = deque()
		self.tt.reset()
		if hasattr(self, "net"):
			getattr(self, "net").eval()
		assert time_limit or max_states
		time_limit = time_limit or 1e10
		max_states = max_states or int(1e10)
		return time_limit, max_states

	def __str__(self):
		raise NotImplementedError

	def __len__(self):
		# Returns number of states explored
		return self._explored_states


class DeepAgent(Agent, ABC):
	def __init__(self, net: Model):
		super().__init__(None)
		self.net = net
		self.env = net.env if net is not None else None

	@classmethod
	def from_saved(cls, loc: str, use_best: bool):
		net = load_net(loc, load_best=use_best)
		net.to(gpu)
		return cls(net)

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		raise NotImplementedError

	def update_net(self, net: Model):
		self.net = net
		self.env = net.env


class RandomSearch(Agent):
	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		action = np.random.randint(self.env.action_dim)
		state = self.env.act(state, action)
		return action, state, self.env.is_solved(state), 1

	def __str__(self):
		return "Random depth-first search"


class BFS(Agent):

	states = dict()

	def search(self, state: np.ndarray, time_limit: float = None, max_states: int = None) -> (np.ndarray, bool):
		# TODO: Use self.env.multi_act for le big speed
		time_limit, max_states = self.reset(time_limit, max_states)
		self.tt.tick()

		if self.env.is_solved(state):
			return True

		# Each element contains the state from which it came and the action taken to get to it
		self.states = { state.tostring(): (None, None) }
		queue = deque([state])
		while self.tt.tock() < time_limit and len(self) < max_states:
			state = queue.popleft()
			tstate = state.tostring()
			for i, action in enumerate(self.env.action_space):
				new_state = self.env.act(state, action)
				new_tstate = new_state.tostring()
				if new_tstate in self.states:
					continue
				elif self.env.is_solved(new_state):
					self.action_queue.appendleft(i)
					while self.states[tstate][0] is not None:
						self.action_queue.appendleft(self.states[tstate][1])
						tstate = self.states[tstate][0]
					return True
				else:
					self.states[new_tstate] = (tstate, i)
					queue.append(new_state)

		return False

	def __str__(self):
		return "Breadth-first search"

	def __len__(self):
		return len(self.states)


class ValueSearch(DeepAgent):

	def _step(self, state: np.ndarray) -> (int, np.ndarray, bool):
		substates = self.env.multi_act(self.env.repeat_state(state, self.env.action_dim), self.env.iter_actions())
		solutions = self.env.multi_is_solved(substates)
		if np.any(solutions):
			action = np.where(solutions)[0][0]
			return action, substates[action], True, self.env.action_dim
		else:
			substates_oh = self.env.as_oh(substates)
			v = self.net(substates_oh).squeeze().cpu().numpy()
			action = np.argmax(v)
			return action, substates[action], False, self.env.action_dim

	def __str__(self):
		return "Greedy value"


class AStar(DeepAgent):
	"""
	Batch Weighted A* Search
	As per Agostinelli, McAleer, Shmakov, Baldi:
	"Solving the Rubik's cube with deep reinforcement learning and search".

	Expands the `self.expansions` best nodes at a time according to cost
	f(node) = `self.lambda_` * g(node) + h(node)
	where h(node) is given as the negative value (cost-to-go) of the DNN and g(x) is the path cost
	"""
	# Expansion priority queue
	# 	Min heap. An element contains tuple of (cost, index)
	# 	This priority queue uses python std. lib heapq which is based on the python list.
	# 	We should maybe consider whether this could be done faster if we build our own implementation.
	open_queue: list

	# State data structures
	# The length of all arrays are dynamic and controlled by `reset` and `expand_stack_size`
	# Index 0 is not used in these to allow for
	# 	states
	# 		Contains all states currently visited in the representation set in cube
	# 	indices:
	# 		Dictionary mapping state.tostring() to index in the states array.
	# 	G_
	# 		A* distance approximation of distance to starting node
	# 	parents
	# 		parents[i] is index of currently found parent with lowest G of state i
	# 	parent_actions
	# 		parent_actions[i] is action idx taken FROM the lightest parent to state i

	indices = dict
	states: np.ndarray
	G: np.ndarray
	parents: np.ndarray
	parent_actions: np.ndarray

	_stack_expand = 1000

	def __init__(self, net: Model, lambda_: float, expansions: int):
		"""Init data structure, save params

		:param net: Neural network whose value output is used as heuristic h
		:param lambda_: The weighting factor in [0,1] that weighs the cost from start node g(x)
		:param expansions: Number of expansions to perform at a time
		"""
		super().__init__(net)
		self.lambda_ = lambda_
		self.expansions = expansions

	@no_grad
	def search(self, state: np.ndarray, time_limit: float = None, max_states: int = None) -> bool:
		"""Seaches according to the batched, weighted A* algorithm

		While there is time left, the algorithm finds the best `expansions` open states
		(using priority queue) with lowest cost according to the A* cost heuristic (see `self.cost`).
		From these, it expands to new open states according to `self.expand_batch`.
		"""
		self.tt.tick()
		time_limit, max_states = self.reset(time_limit, max_states)
		if self.env.is_solved(state):
			return True

		# First node
		self.indices[state.tostring()], self.states[1], self.G[1] = 1, state, 0
		heapq.heappush(self.open_queue, (0, 1))  # Given cost 0: Should not matter; just to avoid np.empty weirdness

		while self.tt.tock() < time_limit and len(self) + self.expansions * self.env.action_dim <= max_states:
			self.tt.profile("Remove nodes from open priority queue")
			n_remove = min( len(self.open_queue), self.expansions )
			expand_idcs = np.array([ heapq.heappop(self.open_queue)[1] for _ in range(n_remove) ], dtype=int)
			self.tt.end_profile("Remove nodes from open priority queue")

			is_won = self.expand_batch(expand_idcs)
			if is_won:  # 🦀🦀🦀WE DID IT BOIS🦀🦀🦀
				i = self.indices[ self.env.get_solved().tostring() ]
				# Build action queue
				while i != 1:
					self.action_queue.appendleft(self.parent_actions[i])
					i = self.parents[i]
				return True
		return False

	def expand_batch(self, expand_idcs: np.ndarray) -> bool:
		"""
		Expands to the neighbors of each of the states in
		Loose pseudo code:
		```
		1. Calculate children for all the batched expansion states
		2. Check which children are seen and not seen
		3. FOR the unseen
			IF they are the goal state: RETURN TRUE
			Set the state as their parent and set their G
			Calculate their H and add to open-list with correct cost
		4. RELAX(seen) #See psudeo code under `relax_seen_states`
		5. RETURN FALSE
		```

		:param expand_idcs: Indices corresponding to states in `self.states` of states from which to expand
		:return: True iff. solution was found in this expansion
		"""
		expand_size = len(expand_idcs)
		while len(self) + expand_size * self.env.action_dim > len(self.states):
			self.increase_stack_size()

		self.tt.profile("Calculate substates")
		parent_idcs = np.repeat(expand_idcs, self.env.action_dim, axis=0)
		substates = self.env.multi_act(
			self.states[parent_idcs],
			self.env.iter_actions(expand_size)
		)
		actions_taken = np.tile(np.arange(self.env.action_dim), expand_size)
		self.tt.end_profile("Calculate substates")

		self.tt.profile("Find new substates")
		substate_strs = [s.tostring() for s in substates]
		get_substate_strs = lambda bools: [s for s, b in zip(substate_strs, bools) if b]
		seen_substates = np.array([s in self.indices for s in substate_strs])
		unseen_substates = ~seen_substates

		# Handle duplicates
		first_occurences    = np.zeros(len(substate_strs), dtype=bool)
		_, first_indeces    = np.unique(substate_strs, return_index=True)
		first_occurences[first_indeces] = True
		first_seen          = first_occurences & seen_substates
		first_unseen        = first_occurences & unseen_substates
		self.tt.end_profile("Find new substates")

		self.tt.profile("Add substates to data structure")
		new_states          = substates[first_unseen]
		new_states_idcs     = len(self) + np.arange(first_unseen.sum()) + 1
		new_idcs_dict       = { s: i for i, s in zip(new_states_idcs, get_substate_strs(first_unseen)) }
		self.indices.update(new_idcs_dict)
		substate_idcs       = np.array([self.indices[s] for s in substate_strs])
		old_states_idcs     = substate_idcs[first_seen]

		self.states[new_states_idcs] = substates[first_unseen]
		self.tt.end_profile("Add substates to data structure")

		self.tt.profile("Update new state values")
		new_parent_idcs = parent_idcs[first_unseen]
		self.G[new_states_idcs] = self.G[new_parent_idcs] + 1
		self.parent_actions[new_states_idcs] = actions_taken[first_unseen]
		self.parents[new_states_idcs] = new_parent_idcs

		# Add the new states to "open" priority queue
		costs = self.cost(new_states, new_states_idcs)
		for i, cost in enumerate(costs):
			heapq.heappush(self.open_queue, (cost, new_states_idcs[i]))
		self.tt.end_profile("Update new state values")

		self.tt.profile("Check whether won")
		solved_substates = self.env.multi_is_solved(new_states)
		if solved_substates.any():
			return True
		self.tt.end_profile("Check whether won")

		self.tt.profile("Old states: Update parents and G")
		seen_batch_idcs = np.where(first_seen)  # Old idcs corresponding to first_seen
		self.relax_seen_states( old_states_idcs, parent_idcs[seen_batch_idcs], actions_taken[seen_batch_idcs] )
		self.tt.end_profile("Old states: Update parents and G")

		return False

	def relax_seen_states(self, state_idcs: np.ndarray, parent_idcs: np.ndarray, actions_taken: np.ndarray):
		"""A* relaxation of states already seen before
		Relaxes the G A* upper bound on distance to starting node.
		Relaxation of new states is done in `expand_batch`. Relaxation of seen states is aheuristic and follows the idea
		of Djikstras algorithm closely with the exception that the new nodes also might prove to reveal a shorter path
		to their parents.

		(Very loose) pseudo code:
		```
		FOR seen children:
			1. IF G of child is lower than G of state +1:
				Update state's G and set the child as its' parent
			2. ELSE IF G of parent + 1 is lower than  G of child:
				Update substate's G and set state as its' parent
		```
		:param state_idcs: Vector, shape (batch_size,) of indeces in `self.states` of already seen states to consider for relaxation
		:param parent_idcs: Vector, shape (batch_size,) of indeces in `self.states` of the parents of these
		:param actions_taken: Vector, shape (batch_size,) where actions_taken[i], in [0, 12], corresponds\
							to action taken from parent i to get to child i
		"""
		# Case: New ways to the substates: When a faster way has been found to the substate
		new_ways = self.G[parent_idcs] + 1 < self.G[state_idcs]
		new_way_states, new_way_parents = state_idcs[new_ways], parent_idcs[new_ways]

		self.G[new_way_states]               = self.G[new_way_parents] + 1
		self.parent_actions[ new_way_states] = actions_taken[new_ways]
		self.parents[new_way_states]         = new_way_parents

		# Case: Shortcuts through the substates: When the substate is a new shortcut to its parent
		shortcuts = self.G[state_idcs] + 1 < self.G[parent_idcs]
		shortcut_states, shortcut_parents = state_idcs[shortcuts], parent_idcs[shortcuts]

		self.G[shortcut_parents] = self.G[shortcut_states] + 1
		self.parent_actions[shortcut_parents] = self.env.rev_actions(actions_taken[shortcuts])
		self.parents[shortcut_parents] = shortcut_states

	@no_grad
	def cost(self, states: np .ndarray, indeces: np.ndarray) -> np.ndarray:
		"""The A star cost of the state using the DNN heuristic
		Uses the value neural network. -value is regarded as the distance heuristic
		It is actually not really necessay to accept both the states and their indices, but
		it speeds things a bit up not having to calculate them here again.

		:param states: (batch size, *(cube_dimensions)) of states
		:param indeces: indeces in self.indeces corresponding to these states.
		"""
		states = self.env.as_oh(states)
		H = -self.net(states)
		H = H.cpu().squeeze().detach().numpy()

		return self.lambda_ * self.G[indeces] + H

	def reset(self, time_limit: float, max_states: int) -> (float, int):
		time_limit, max_states = super().reset(time_limit, max_states)

		self.open_queue     = list()
		self.indices        = dict()
		self.states         = np.empty((self._stack_expand, *self.env.shape()), dtype=self.env.dtype)
		self.parents        = np.empty(self._stack_expand, dtype=int)
		self.parent_actions = np.zeros(self._stack_expand, dtype=int)
		self.G              = np.empty(self._stack_expand)
		return time_limit, max_states

	def increase_stack_size(self):
		expand_size    = len(self.states)

		self.states	        = np.concatenate([self.states, np.empty((expand_size, *self.env.shape()), dtype=self.env.dtype)])
		self.parents        = np.concatenate([self.parents, np.zeros(expand_size, dtype=int)])
		self.parent_actions = np.concatenate([self.parent_actions, np.zeros(expand_size, dtype=int)])
		self.G              = np.concatenate([self.G, np.empty(expand_size)])

	@classmethod
	def from_saved(cls, loc: str, use_best: bool, lambda_: float, expansions: int) -> DeepAgent:
		net = load_net(loc, load_best=use_best)
		return cls(net, lambda_=lambda_, expansions=expansions)

	def __len__(self) -> int:
		return len(self.indices)

	def __str__(self) -> str:
		return f'AStar (lambda={self.lambda_}, N={self.expansions})'
