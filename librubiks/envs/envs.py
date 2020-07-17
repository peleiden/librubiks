"""
The API for the available enviroments -- for now only two representations of Rubik's cube
As the environments are used many times in many different ways, this module has some requirements:

- All functionality must be out-of-place, as doing it in-place would require significant bookkeeping outside the module
- Efficiency
- All enviroments must have the same API, which currently only allows for discrete enviroments,
  the states of which can be represented by a numpy array
  While it is possible to implement stochastic environments, the rest of the code base will not work with such environments

The solution to this is this quite large module with these features:

- The module carries NO STATE: Environment state must be maintained elsewhere when used, and this API is purely functional
- Many functions are implemented compactly using numpy or pytorch sacrificing readability for efficiency

Custom environments are easy to implement; simply let them inherit from the Environment superclass
and add an entry to the environments dict at the end of the file
"""

from abc import ABC

import numpy as np
import torch

from librubiks import gpu
from librubiks.envs.cube_maps import SimpleState, get_corner_pos, get_side_pos, get_tensor_map, get_633maps, neighbors_686


#########################
# BASE ENVIROMENT CLASS #
#########################

class Environment(ABC):
	key: str
	dtype = np.int8
	action_dim: int
	action_space: np.ndarray
	action_names: tuple
	shape: tuple
	oh_size: int
	_solved_instance: np.ndarray

	def __init__(self, key: str):
		self.key = key

	def act(self, state: np.ndarray, action: int) -> np.ndarray:
		"""
		Perform an action on a given state and return the result
		"""
		raise NotImplementedError

	def multi_act(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		"""
		Perform n actions on n states and return the n resulting states
		"""
		return np.array([self.act(state, action) for state, action in zip(states, actions)])

	def get_solved_instance(self) -> np.ndarray:
		"""
		Get an instance of the solved state
		Careful Ned, careful now -- this method does not return a copy,
		and the result is there read-only and should only be used, when speed is critical
		"""
		return self._solved_instance

	def get_solved(self) -> np.ndarray:
		"""
		The same as Environment.get_solved_instance, but returns a copy, which can be safely modified
		"""
		return self.get_solved_instance().copy()

	def is_solved(self, state: np.ndarray) -> bool:
		"""
		Returns whether or not the given state is solved
		"""
		return (state == self.get_solved_instance()).all()

	def multi_is_solved(self, states: np.ndarray) -> np.ndarray:
		"""
		Returns an n long boolean vector containing whether or not the n given states are solved
		"""
		return (states == self.get_solved_instance()).all(axis=tuple(range(1, len(self.shape) + 1)))

	def as_oh(self, state: np.ndarray) -> torch.tensor:
		"""
		One-hot encodes a state and returns the result saved on the gpu
		"""
		raise NotImplementedError

	def multi_as_oh(self, states: np.ndarray) -> torch.tensor:
		"""
		One-hot encodes n states and returns the result saved on the gpu
		"""
		return torch.cat([self.as_oh(state) for state in states])

	def rev_action(self, action: int) -> int:
		"""
		Returns the reverse of a given action, such that
		env.act(state, action) == env.act(env.act(state, action), env.rev_action(action))
		"""
		raise NotImplementedError

	def multi_rev_action(self, actions: np.ndarray) -> np.ndarray:
		"""
		Returns n actions which are the reverse of the n given actions
		"""
		return np.array([self.rev_action(action) for action in actions])

	def stringify(self, state: np.ndarray) -> str:
		"""
		Stringifies the state into a more readable manner
		"""
		raise NotImplementedError

	###################
	# Utility methods #
	###################

	def iter_actions(self, n: int = 1):
		"""
		Returns a numpy array of size n * action_dim containing tiled actions
		Useful in combination with repeat_state
		"""
		return np.tile(self.action_space, n)

	def sample_actions(self, n: int = 1):
		"""
		Randomly samples n actions in the action space
		"""
		return np.random.randint(0, self.action_dim, size=n)

	def repeat_state(self, state: np.ndarray, n: int = None) -> np.ndarray:
		"""
		Repeats state n times, such that the output array will have shape n x *Cube shape
		Useful in combination with multi_rotate
		"""
		if n is None:
			n = self.action_dim
		return np.tile(state, [n, *[1] * len(self.shape)])

	####################
	# Scrambling logic #
	####################

	def scramble(self, depth: int, force_not_solved=False) -> (np.ndarray, np.ndarray, np.ndarray):
		actions = np.random.randint(0, self.action_dim, depth)
		state = self.get_solved()
		for action in actions:
			state = self.act(state, action)

		if force_not_solved and depth != 0 and self.is_solved(state):
			return self.scramble(depth, True)

		return state, actions

	def sequence_scrambler(self, games: int, depth: int, with_solved: bool) -> (np.ndarray, torch.tensor):
		"""
		An out-of-place scrambler which returns the state to each of the scrambles useful for ADI
		Returns a games x n x 20 tensor with states as well as their one-hot representations (games * n) x 480
		:with_solved: Whether to include the solved cube in the sequence
		"""
		states = []
		current_states = self.repeat_state(self.get_solved(), games)
		actions = np.random.randint(0, self.action_dim, (depth, games))
		if with_solved: states.append(current_states)
		for d in range(depth - with_solved):
			current_states = self.multi_act(current_states, actions[d])
			states.append(current_states)
		states = np.vstack(np.transpose(states, (1, 0, *np.arange(2, len(self.shape) + 2))))
		oh_states = self.multi_as_oh(states)
		return states, oh_states

	def __str__(self):
		raise NotImplementedError


################
# ENVIRONMENTS #
################

class _Cube(Environment, ABC):
	"""
	Abstract cube class containing shared properties and methods for the different representations
	"""

	action_dim = 12
	action_space = np.arange(action_dim)
	action_names = "F", "B", "T", "D", "L", "R", "F'", "B'", "T'", "D'", "L'", "R'"

	# If the six sides are represented by an array, the order should be F, B, T, D, L, R
	F, B, T, D, L, R = 0, 1, 2, 3, 4, 5

	def rev_action(self, action: int) -> int:
		return action + 6 if action < 6 else action - 6

	def rev_actions(self, actions: np.ndarray) -> np.ndarray:
		neg_dir = actions < 6
		actions = actions - 6  # Positive revolution
		actions[neg_dir] += 12  # Negative revolution
		return actions

	def _as633(self, state: np.ndarray) -> np.ndarray:
		"""
		Encode as 6 x 3 x 3 array
		Order: F, B, T, D, L, R
		"""
		raise NotImplementedError

	def _stringify_cube(self, state633: np.ndarray) -> str:
		stringarr = np.empty((9, 12), dtype=str)
		stringarr[...] = " "
		simple = np.array([
			[-1, self.T, -1, -1],
			[self.L, self.F, self.R, self.B],
			[-1, self.D, -1, -1],
		])
		for i in range(6):
			pos = tuple(int(x) for x in np.where(simple == i))
			stringarr[pos[0] * 3: pos[0] * 3 + 3, pos[1] * 3: pos[1] * 3 + 3] = state633[i].astype(str)
		string = "\n".join([" ".join(list(y)) for y in stringarr])
		return string

	def stringify(self, state: np.ndarray) -> str:
		return self._stringify_cube(self._as633(state))

	def _get_686solved(self) -> np.ndarray:
		solved_state = np.zeros((6, 8, 6), dtype=self.dtype)
		for i in range(6):
			solved_state[i, :, i] = 1
		return solved_state

	def _get_2024solved(self) -> np.ndarray:
		solved_state = SimpleState()
		tensor_state = np.empty(20, dtype=self.dtype)
		for i in range(8):
			tensor_state[i] = get_corner_pos(solved_state.corners[i], solved_state.corner_orientations[i])
		for i in range(12):
			tensor_state[i + 8] = get_side_pos(solved_state.sides[i], solved_state.side_orientations[i])
		return tensor_state

	@staticmethod
	def _face_dir(actions):
		"""
		Converts one or more actions to legacy type actions, that is face 0-5 and direction (0 for negative, 1 for positive)
		"""
		faces = actions % 6
		directions = int(actions >= 6) if type(actions) == int else (actions >= 6).astype(int)
		return faces, directions


class _Cube2024(_Cube):
	oh_size = 480

	corner_side_idcs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	oh_idcs = np.arange(20) * 24

	def __init__(self, key: str):
		super().__init__(key)
		self._solved_instance = self._get_2024solved()
		self.shape = self._solved_instance.shape
		self.corner_633map, self.side_633map = get_633maps(self.F, self.B, self.T, self.D, self.L, self.R)
		self.maps = get_tensor_map(self.dtype)
		self.corner_maps, self.side_maps = self.maps

	def act(self, state: np.ndarray, action: int):
		return self.maps[self.corner_side_idcs, action, state].copy()

	def multi_act(self, states: np.ndarray, actions: np.ndarray):
		repeated_actions = np.broadcast_to(actions, (20, len(actions))).T.flat
		corners_sides = np.broadcast_to(self.corner_side_idcs, (len(states), 20)).flat
		states = self.maps[corners_sides, repeated_actions, states.flat].reshape((len(states), 20)).copy()
		return states

	def as_oh(self, state: np.ndarray) -> torch.tensor:
		"""Takes in a state and returns an 1 x 480 one-hot tensor"""
		oh = torch.zeros(1, 480, device=gpu)
		idcs = self.oh_idcs + state
		oh[0, idcs] = 1
		return oh

	def multi_as_oh(self, states: np.ndarray) -> torch.tensor:
		oh = torch.zeros(states.shape[0], 480, device=gpu)
		idcs = self.oh_idcs + states
		all_idcs = np.broadcast_to(np.arange(len(states)), (20, len(states)))
		oh[all_idcs.T.ravel(), idcs.ravel()] = 1
		return oh

	def _as633(self, state: np.ndarray):
		state633 = (np.ones((3, 3, 6)) * np.arange(6)).transpose(2, 1, 0).astype(int)
		for i in range(8):
			# Inserts values for corner i in position pos
			pos = state[i] // 3
			orientation = state[i] % 3
			# For these corners, "right turn" order is 0 2 1 instead of 0 1 2, so orientation is messed up without this fix
			if pos in [0, 2, 5, 7]:
				orientation *= -1
			values = np.roll([x[0] for x in self.corner_633map[i]], orientation)
			state633[self.corner_633map[pos][0]] = values[0]
			state633[self.corner_633map[pos][1]] = values[1]
			state633[self.corner_633map[pos][2]] = values[2]

		for i in range(12):
			# Inserts values for side i in position pos
			pos = state[i + 8] // 2
			orientation = state[i + 8] % 2
			values = np.roll([x[0] for x in self.side_633map[i]], orientation)
			state633[self.side_633map[pos][0]] = values[0]
			state633[self.side_633map[pos][1]] = values[1]

		return state633

	def __str__(self):
		return "Cube 20x24"


class _Cube686(_Cube):
	oh_size = 288

	# Maps an 8 long vector starting at (0, 0) in 3x3 onto a 9 long vector which can be reshaped to 3x3
	map633 = np.array([0, 3, 6, 7, 8, 5, 2, 1])
	# Number of times the 8 long vector has to be shifted to the left to start at (0, 0) in 3x3
	shifts = np.array([0, 6, 6, 4, 2, 4])

	def __init__(self, key: str):
		super().__init__(key)
		self._solved_instance = self._get_686solved()
		self.shape = self._solved_instance.shape

		# No shame
		self._Cube686_n3_03 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
		self._Cube686_n3_n13 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2])
		self.adjacents = np.array([6, 7, 0, 2, 3, 4, 4, 5, 6, 0, 1, 2])
		self.rolled_adjecents = np.roll(self.adjacents, 3)
		self.roll_left = np.array([2, 3, 4, 5, 6, 7, 0, 1])
		self.roll_right = np.array([6, 7, 0, 1, 2, 3, 4, 5])
		self.neighbor_idcs_pos = [neighbors_686[[face] * 12, self._Cube686_n3_03] for face in np.arange(6)]
		self.neighbor_idcs_neg = [neighbors_686[[face] * 12, self._Cube686_n3_n13] for face in np.arange(6)]

	def act(self, state: np.ndarray, action: int):
		"""
		Performs one move on the cube, specified by the side (0-5),
		and if the direction is negative (0) or positive (1)
		"""

		face, direction = self._face_dir(action)

		altered_state = state.copy()
		ini_state = state[neighbors_686[face]]

		if direction:
			altered_state[face] = state[face, self.roll_right]
			altered_state[self.neighbor_idcs_pos[face], self.adjacents] = ini_state[
				self._Cube686_n3_n13, self.rolled_adjecents]
		else:
			altered_state[face] = state[face, self.roll_left]
			altered_state[self.neighbor_idcs_neg[face], self.rolled_adjecents] = ini_state[
				self._Cube686_n3_03, self.adjacents]

		return altered_state

	def multi_act(self, states: np.ndarray, actions: np.ndarray):
		faces, directions = self._face_dir(actions)

		altered_states = states.copy()
		ini_states = np.array([state[n] for state, n in zip(states, neighbors_686[faces])])
		for altered_state, state, ini_state, face, direction in zip(altered_states, states, ini_states, faces,
																	directions):
			if direction:
				altered_state[face] = state[face, self.roll_right]
				altered_state[self.neighbor_idcs_pos[face], self.adjacents] = ini_state[
					self._Cube686_n3_n13, self.rolled_adjecents]
			else:
				altered_state[face] = state[face, self.roll_left]
				altered_state[self.neighbor_idcs_neg[face], self.rolled_adjecents] = ini_state[
					self._Cube686_n3_03, self.adjacents]

		return altered_states

	def as_oh(self, state: np.ndarray) -> torch.tensor:
		# This representation is already one-hot encoded, so only ravelling is done
		state = np.expand_dims(state, 0)
		oh_state = torch.from_numpy(state.reshape(len(state), 288)).to(gpu).float()
		return oh_state

	def multi_as_oh(self, states: np.ndarray) -> torch.tensor:
		oh_states = torch.from_numpy(states.reshape(len(states), 288)).to(gpu).float()
		return oh_states

	def _as633(self, state: np.ndarray):
		state68 = np.where(state == 1)[2].reshape((6, 8))
		state69 = (np.ones((9, 6)) * np.arange(6)).astype(int).T  # Nice
		for i in range(6):
			state69[i, self.map633] = np.roll(state68[i], -self.shifts[i], axis=0)
		return state69.reshape((6, 3, 3))

	def __str__(self):
		return "Cube 6x8x6"


environments = {
	"cube2024": _Cube2024,
	"cube686": _Cube686,
}


def get_env(env_key: str) -> Environment:
	assert env_key in environments, f"Environment must be one of " + ", ".join(environments.keys()) + f", but {env_key} was given"
	return environments[env_key](env_key)
