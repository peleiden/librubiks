"""
The API for our Rubik's Cube simulator.
As we use the Rubik's cube in many different ways, this module has some requirements

- Pretty much all functions must work without altering any state as they are to be used by agents.
- Efficiency
- Must support two different Rubik's representations (20x24 and 6x8x6)

The solution to this is this quite large module with these features

- The module carries NO STATE: Environment state must be maintained elsewhere when used and this API works
	mostly purely functionally
- Many functions are polymorphic between corresponding functions of the two representations
- Most functions are implemented compactly using numpy or pytorch sacrificing readability for efficiency
- Some global constants are maintained
"""
from abc import ABC
from enum import Enum
import functools
import numpy as np
import torch

from librubiks import gpu
from librubiks.envs.maps import SimpleState, get_corner_pos, get_side_pos, get_tensor_map, get_633maps, neighbors_686

#########################
# BASE ENVIROMENT CLASS #
#########################

class _Environment(ABC):

	dtype = np.int8
	action_dim: int
	action_space: np.ndarray
	action_names: tuple
	shape: tuple
	oh_size: int
	_solved_state: np.ndarray
	
	@classmethod
	def act(cls, state: np.ndarray, action: int) -> np.ndarray:
		raise NotImplementedError

	@classmethod
	def multi_act(cls, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		return np.array([cls.act(state, action) for state, action in zip(states, actions)])

	@classmethod
	def get_solved_instance(cls) -> np.ndarray:
		return cls._solved_state

	@classmethod
	def get_solved(cls) -> np.ndarray:
		return cls.get_solved_instance().copy()

	@classmethod
	def is_solved(cls, state: np.ndarray) -> np.ndarray:
		raise NotImplementedError

	@classmethod
	def multi_is_solved(cls, states: np.ndarray) -> np.ndarray:
		return np.ndarray([cls.is_solved(state) for state in states])

	@classmethod
	def as_oh(cls, state: np.ndarray) -> torch.tensor:
		raise NotImplementedError

	@classmethod
	def multi_as_oh(cls, states: np.ndarray) -> torch.tensor:
		return torch.cat([cls.as_oh(state) for state in states])
	
	@classmethod
	def rev_action(cls, action: int) -> int:
		raise NotImplementedError

	@classmethod
	def multi_rev_action(cls, actions: np.ndarray) -> np.ndarray:
		return np.array([cls.rev_action(action) for action in actions])
	
	@classmethod
	def stringify(cls, state: np.ndarray) -> str:
		raise NotImplementedError
	
	################
	# Action logic #
	################

	@classmethod
	def iter_actions(cls, n: int=1):
		"""
		Returns a numpy array of size n * action_dim containing tiled actions
		Useful in combination with repeat_states
		"""
		return np.tile(cls.action_space, n)

	@classmethod
	def repeat_state(cls, state: np.ndarray, n: int=None) -> np.ndarray:
		"""
		Repeats state n times, such that the output array will have shape n x *Cube shape
		Useful in combination with multi_rotate
		"""
		if n is None:
			n = cls.action_dim
		return np.tile(state, [n, *[1]*len(shape())])

	####################
	# Scrambling logic #
	####################
	
	@classmethod
	def scramble(cls, depth: int, force_not_solved=False) -> (np.ndarray, np.ndarray, np.ndarray):
		actions = np.random.randint(0, cls.action_dim, depth)
		state = cls.get_solved()
		for action in actions:
			state = cls.act(state, action)

		if force_not_solved and depth != 0 and cls.is_solved(state):
			return cls.scramble(depth, True)

		return state, actions

	@classmethod
	def sequence_scrambler(cls, games: int, depth: int, with_solved: bool) -> (np.ndarray, torch.tensor):
		"""
		An out-of-place scrambler which returns the state to each of the scrambles useful for ADI
		Returns a games x n x 20 tensor with states as well as their one-hot representations (games * n) x 480
		:with_solved: Whether to include the solved cube in the sequence
		"""
		states = []
		current_states = cls.repeat_state(cls.get_solved(), games)
		actions = np.random.randint(0, cls.action_dim, (depth, games))
		if with_solved: states.append(current_states)
		for d in range(depth - with_solved):
			current_states = cls.multi_act(current_states, actions[d])
			states.append(current_states)
		states = np.vstack(np.transpose(states, (1, 0, *np.arange(2, len(shape())+2))))
		oh_states = as_oh(states)
		return states, oh_states


# If the six sides are represented by an array, the order should be F, B, T, D, L, R
F, B, T, D, L, R = 0, 1, 2, 3, 4, 5

################
# ENVIRONMENTS #
################

# TODO: Consider shared cube enviroment for stuff like this
def _stringify_cube(state633: np.ndarray) -> str:
	stringarr = np.empty((9, 12), dtype=str)
	stringarr[...] = " "
	simple = np.array([
		[-1, T, -1, -1],
		[L,  F,  R,  B],
		[-1, D, -1, -1],
	])
	for i in range(6):
		pos = tuple(int(x) for x in np.where(simple==i))
		stringarr[pos[0]*3 : pos[0]*3+3, pos[1]*3 : pos[1]*3+3] = state633[i].astype(str)
	string = "\n".join([" ".join(list(y)) for y in stringarr])
	return string
# FIXME
def rev_action(action: int) -> int:
	return action + 1 if action % 2 == 0 else action - 1

def rev_actions(actions: np.ndarray) -> np.ndarray:
	rev_actions = actions - 1
	rev_actions[actions % 2 == 0] += 2
	return rev_actions

@staticmethod
def _get_2024solved(dtype):
	solved_state = SimpleState()
	tensor_state = np.empty(20, dtype=dtype)
	for i in range(8):
		tensor_state[i] = get_corner_pos(solved_state.corners[i], solved_state.corner_orientations[i])
	for i in range(12):
		tensor_state[i+8] = get_side_pos(solved_state.sides[i], solved_state.side_orientations[i])
	return tensor_state
def _get_686solved(dtype):
	solved_state = np.zeros((6, 8, 6), dtype=dtype)
	for i in range(6):
		solved_state[i, :, i] = 1
	return solved_state

_solved2024 = _get_2024solved(dtype)
_solved686 = _get_686solved(dtype)

class _Cube2024(_Environment):

	action_dim = 12
	action_space = np.arange(action_dim)
	action_names = "F", "F'", "B", "B'", "T", "T'", "D", "D'", "L", "L'", "R", "R'"
	oh_size = 480
	_solved_state = _get_2024solved(dtype)

	maps = get_tensor_map(dtype)
	corner_maps, side_maps = maps
	corner_side_idcs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
	corner_633map, side_633map = get_633maps(F, B, T, D, L, R)
	oh_idcs = np.arange(20) * 24

	@classmethod
	def act(cls, state: np.ndarray, action: int):
		"""
		Performs one move on the cube, specified by the side (0-5),
		and whether the rotation is in a positive direction (0 for negative and 1 for positive)
		"""
		
		return cls.maps[cls.corner_side_idcs, action, state].copy()

	@classmethod
	def multi_act(cls, states: np.ndarray, actions: np.ndarray):
		repeated_actions = np.tile(actions, (12, 1)).ravel("F")
		states = states.copy()
		states[:, :8] = cls.corner_maps[repeated_actions[:8*len(actions)], states[:, :8].flat].reshape(-1, 8)
		states[:, 8:] = cls.side_maps[repeated_actions, states[:, 8:].flat].reshape(-1, 12)
		return states

	@classmethod
	def as_oh(cls, states: np.ndarray) -> torch.tensor:
		# Takes in n states and returns an n x 480 one-hot tensor
		oh = torch.zeros(1, 480, device=gpu)
		idcs = cls.oh_idcs + states
		oh[0, idcs] = 1
		return oh
	
	@classmethod
	def multi_as_oh(cls, states: np.ndarray) -> torch.tensor:
		oh = torch.zeros(states.shape[0], 480, device=gpu)
		idcs = cls.oh_idcs + states
		all_idcs = np.broadcast_to(np.arange(len(states)), (20, len(states)))
		oh[all_idcs.T.ravel(), idcs.ravel()] = 1
		return oh

	@classmethod
	def _as633(cls, state: np.ndarray):
		"""
		Order: F, B, T, D, L, R
		"""
		# Starts with assembled state
		state633 = (np.ones((3, 3, 6)) * np.arange(6)).transpose(2, 1, 0).astype(int)
		for i in range(8):
			# Inserts values for corner i in position pos
			pos = state[i] // 3
			orientation = state[i] % 3
			# Mapping should probably be smarter
			# For these corners, "right turn" order is 0 2 1 instead of 0 1 2, so orientation is messed up without this fix
			if pos in [0, 2, 5, 7]:
				orientation *= -1
			values = np.roll([x[0] for x in cls.corner_633map[i]], orientation)
			state633[cls.corner_633map[pos][0]] = values[0]
			state633[cls.corner_633map[pos][1]] = values[1]
			state633[cls.corner_633map[pos][2]] = values[2]
		
		for i in range(12):
			# Inserts values for side i in position pos
			pos = state[i+8] // 2
			orientation = state[i+8] % 2
			values = np.roll([x[0] for x in cls.side_633map[i]], orientation)
			state633[cls.side_633map[pos][0]] = values[0]
			state633[cls.side_633map[pos][1]] = values[1]
		
		return state633
	
	@classmethod
	def stringify(cls, state: np.ndarray):
		return _stringify_cube(cls._as633(state))



class _Cube686(_Environment):

	action_dim = 12
	action_space = np.arange(action_dim)
	action_names = "F", "F'", "B", "B'", "T", "T'", "D", "D'", "L", "L'", "R", "R'"
	oh_size = 288
	_solved_state = _get_686solved(dtype)
	
	# No shame
	_Cube686_n3_03    = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
	_Cube686_n3_n13   = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2])
	adjacents         = np.array([6, 7, 0, 2, 3, 4, 4, 5, 6, 0, 1, 2])
	rolled_adjecents  = np.roll(adjacents, 3)
	roll_left         = np.array([2, 3, 4, 5, 6, 7, 0, 1])
	roll_right        = np.array([6, 7, 0, 1, 2, 3, 4, 5])
	neighbor_idcs_pos = [neighbors_686[[face]*12, _Cube686_n3_03] for face in np.arange(6)]
	neighbor_idcs_neg = [neighbors_686[[face]*12, _Cube686_n3_n13] for face in np.arange(6)]

	# Maps an 8 long vector starting at (0, 0) in 3x3 onto a 9 long vector which can be reshaped to 3x3
	map633 = np.array([0, 3, 6, 7, 8, 5, 2, 1])
	# Number of times the 8 long vector has to be shifted to the left to start at (0, 0) in 3x3
	shifts = np.array([0, 6, 6, 4, 2, 4])

	solved_cuda = torch.from_numpy(_get_686solved(dtype)).to(gpu)

	@classmethod
	def rotate(cls, state: np.ndarray, face: int, direction: int):
		"""
		Performs one move on the cube, specified by the side (0-5),
		and if the direction is negative (0) or positive (1)
		"""

		altered_state = state.copy()
		ini_state = state[neighbors_686[face]]

		if direction:
			altered_state[face] = state[face, cls.roll_right]
			altered_state[cls.neighbor_idcs_pos[face], cls.adjacents] = ini_state[cls._Cube686_n3_n13, cls.rolled_adjecents]
		else:
			altered_state[face] = state[face, cls.roll_left]
			altered_state[cls.neighbor_idcs_neg[face], cls.rolled_adjecents] = ini_state[cls._Cube686_n3_03, cls.adjacents]

		return altered_state

	@classmethod
	def multi_rotate(cls, states: np.ndarray, faces: np.ndarray, directions: np.ndarray):
		altered_states = states.copy()
		ini_states = np.array([state[n] for state, n in zip(states, neighbors_686[faces])])
		for altered_state, state, ini_state, face, direction in zip(altered_states, states, ini_states, faces, directions):
			if direction:
				altered_state[face] = state[face, cls.roll_right]
				altered_state[cls.neighbor_idcs_pos[face], cls.adjacents] = ini_state[cls._Cube686_n3_n13, cls.rolled_adjecents]
			else:
				altered_state[face] = state[face, cls.roll_left]
				altered_state[cls.neighbor_idcs_neg[face], cls.rolled_adjecents] = ini_state[cls._Cube686_n3_03, cls.adjacents]

		return altered_states

	@staticmethod
	def as_oh(states: np.ndarray) -> torch.tensor:
		# This representation is already one-hot encoded, so only ravelling is done
		if len(states.shape) == 3:
			states = np.expand_dims(states, 0)
		states = torch.from_numpy(states.reshape(len(states), 288)).to(gpu).float()
		return states

	@classmethod
	def as_correct(cls, t: torch.tensor) -> torch.tensor:
		"""
		oh is a one-hot encoded tensor of shape n x 288 as produced by _Cube686.as_oh
		This methods creates a correctness representation of the tensor of shape n x 6 x 8
		"""
		oh = t.reshape(len(t), 6, 8, 6).to(gpu)
		correct_repr = torch.all(oh[:] == cls.solved_cuda, dim=3).long()
		correct_repr[correct_repr==0] = -1
		return correct_repr.float()

	@classmethod
	def _as633(cls, state: np.ndarray):
		state68 = np.where(state == 1)[2].reshape((6, 8))
		state69 = (np.ones((9, 6)) * np.arange(6)).astype(int).T  # Nice
		for i in range(6):
			state69[i, cls.map633] = np.roll(state68[i], -cls.shifts[i], axis=0)
		return state69.reshape((6, 3, 3))

	@classmethod
	def stringify(cls, state: np.ndarray):
		return _stringify_cube(cls._as633(state))


_environments = {
	"cube2024": _Cube2024,
	"cube686": _Cube686
}

def get_env(env: str) -> _Environment:
	assert env in _environments, f"Environment ('{env}' was specified) must be one of " + ", ".join(_environments.keys())
	return _environments[env]

