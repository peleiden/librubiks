from os import cpu_count

import numpy as np
import torch
import torch.multiprocessing as mp

from src.rubiks import get_repr, set_repr
from src.rubiks.cube.maps import SimpleState, get_corner_pos, get_side_pos, get_tensor_map, get_633maps


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

# Implemented toplevel for multithreading
def _sequence_scrambler(n: int):
	"""
	A non-inplace scrambler that returns the state to each of the scrambles useful for ADI
	"""
	env = _Cube2024 if get_repr() else _Cube686
	scrambled_states = np.empty((n, *env.get_solved_instance().shape), dtype=env.dtype)
	scrambled_states[0] = env.get_solved()

	faces = np.random.randint(6, size = (n, ))
	dirs = np.random.randint(2, size = (n, )).astype(bool)
	for i, face, d in zip(range(n-1), faces, dirs):
		scrambled_states[i+1] = env.rotate(scrambled_states[i], face, d)
	return scrambled_states, env.as_oh(scrambled_states)

class Cube:
	# If the six sides are represented by an array, the order should be F, B, T, D, L, R
	# For niceness
	F, B, T, D, L, R = 0, 1, 2, 3, 4, 5
	# Corresponding colours
	colours = ["red", "orange", "white", "yellow", "green", "blue"]
	rgba = [
		(1, 0, 0, 1),
		(1, .6, 0, 1),
		(1, 1, 1, 1),
		(1, 1, 0, 1),
		(0, 1, 0, 1),
		(0, 0, 1, 1),
	]

	dtype = np.int8  # Data type used for internal representation
	_solved2024 = _get_2024solved(dtype)
	_solved686 = _get_686solved(dtype)

	action_space = list()
	for i in range(6): action_space.extend( [(i, True), (i, False)] )
	action_dim = len(action_space)

	@classmethod
	def rotate(cls, current_state: np.ndarray, face: int, pos_rev: bool) -> np.ndarray:
		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""
		method = _Cube2024.rotate if get_repr() else _Cube686.rotate
		return method(current_state, face, pos_rev)

	@classmethod
	def scramble(cls, n: int, force_not_solved=False):
		faces = np.random.randint(6, size=(n,))
		dirs = np.random.randint(2, size=(n,)).astype(bool)
		state = cls.get_solved()
		for face, d in zip(faces, dirs):
			state = cls.rotate(state, face, d)  # Uses rotate instead of move as checking for victory is not needed here
		
		if force_not_solved and cls.is_solved(state):
			return cls.scramble(n, True)
		
		return state, faces, dirs

	@classmethod
	def sequence_scrambler(cls, games: int, n: int):
		"""
		A non-inplace scrambler which returns the state to each of the scrambles useful for ADI
		Returns a games x n x 20 tensor with states as well as their one-hot representations (games * n) x 480
		"""
		# Multithreads if over 1000 games
		# Experimentally, this seems to be around the point at which multithreading is worth it
		if games > 1000:
			with mp.Pool(cpu_count()) as p:
				res = p.map(_sequence_scrambler, [n]*games)
		else:
			res = [_sequence_scrambler(n) for _ in range(games)]
		states = np.array([x[0] for x in res])
		oh_states = torch.stack([x[1] for x in res]).view(-1, cls.get_oh_shape())
		return states, oh_states

	@classmethod
	def get_solved_instance(cls):
		# Careful - this method returns the instance - not a copy - so the output is readonly
		# If speed is not critical, use get_solved()
		return cls._solved2024 if get_repr() else cls._solved686

	@classmethod
	def get_solved(cls):
		return cls.get_solved_instance().copy()

	@classmethod
	def is_solved(cls, state: np.ndarray):
		return (state == cls.get_solved_instance()).all()

	@classmethod
	def as_oh(cls, states: np.ndarray) -> torch.tensor:
		# Takes in n states and returns an n x 480 one-hot tensor
		method = _Cube2024.as_oh if get_repr() else _Cube686.as_oh
		return method(states)

	@staticmethod
	def get_oh_shape():
		return 480 if get_repr() else 288

	@staticmethod
	def rev_action(action: int):
		return action + 1 if action % 2 == 0 else action - 1

	@classmethod
	def as633(cls, state: np.ndarray) -> np.ndarray:
		"""
		Order: F, B, T, D, L, R
		"""
		method = _Cube2024.as633 if get_repr() else _Cube686.as633
		return method(state)

	@classmethod
	def stringify(cls, state: np.ndarray) -> str:
		state633 = cls.as633(state)
		stringarr = np.empty((9, 12), dtype=str)
		stringarr[...] = " "
		simple = np.array([
			[-1, cls.T, -1, -1],
			[cls.L, cls.F, cls.R, cls.B],
			[-1, cls.D, -1, -1],
		])
		for i in range(6):
			pos = tuple(int(x) for x in np.where(simple==i))
			stringarr[pos[0]*3 : pos[0]*3+3, pos[1]*3 : pos[1]*3+3] = state633[i].astype(str)
		string = "\n".join([" ".join([x for x in y]) for y in stringarr])
		return string


class _Cube2024(Cube):

	map_pos, map_neg = get_tensor_map(Cube.dtype)
	corner_633map, side_633map = get_633maps(Cube.F, Cube.B, Cube.T, Cube.D, Cube.L, Cube.R)
	
	@classmethod
	def rotate(cls, current_state: np.ndarray, face: int, pos_rev: bool):
		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""
		altered_state = current_state.copy()
		map_ = cls.map_pos[face] if pos_rev else cls.map_neg[face]
		altered_state[:8] += map_[0, altered_state[:8]]
		altered_state[8:] += map_[1, altered_state[8:]]
		
		return altered_state
	
	@classmethod
	def as_oh(cls, states: np.ndarray):
		# Takes in n states and returns an n x 480 one-hot tensor
		if len(states.shape) == 1:
			oh = torch.zeros(1, 480)
			idcs = np.arange(20) * 24 + states
			oh[0, idcs] = 1
		else:
			oh = torch.zeros(states.shape[0], 480)
			idcs = np.array([np.arange(20) * 24 + state for state in states])
			all_idcs = np.arange(len(idcs))
			for i in idcs.T:
				oh[all_idcs, i] = 1
		return oh
	
	@classmethod
	def as633(cls, state: np.ndarray):
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


class _Cube686(Cube):

	# The i'th index contain the neighbors of the i'th side in positive direction
	neighbours = np.array([
		[4, 3, 5, 2],  # Front
		[3, 4, 2, 5],  # Back
		[0, 5, 1, 4],  # Top
		[5, 0, 4, 1],  # Down
		[2, 1, 3, 0],  # Left
		[1, 2, 0, 3],  # Right
	])
	adjacents = np.array([  # TODO
		[6, 7, 0],
		[2, 3, 4],
		[4, 5, 6],
		[0, 1, 2],
	])
	# Maps an 8 long vector starting at (0, 0) in 3x3 onto a 9 long vector which can be reshaped to 3x3
	map633 = np.array([0, 3, 6, 7, 8, 5, 2, 1])
	# Number of times the 8 long vector has to be shifted to the left to start at (0, 0) in 3x3
	shifts = np.array([0, 6, 6, 4, 2, 4])

	@staticmethod
	def _shift_left(a: np.ndarray, num_elems: int):
		return np.roll(a, -num_elems, axis=0)

	@staticmethod
	def _shift_right(a: np.ndarray, num_elems: int):
		return np.roll(a, num_elems, axis=0)

	@classmethod
	def rotate(cls, current_state: np.ndarray, face: int, pos_rev: bool):
		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		# if not 0 <= face <= 5:
		# 	raise IndexError("Face should be 0-5, not %i" % face)
		altered_state = current_state.copy()
		altered_state[face] = cls._shift_right(current_state[face], 2)\
			if pos_rev else cls._shift_left(current_state[face], 2)

		ini_state = current_state[cls.neighbours[face]]

		if pos_rev:
			for i in range(4):
				altered_state[cls.neighbours[face, i], cls.adjacents[i]] = ini_state[i-1, cls.adjacents[i-1]]
		else:
			for i in range(4):
				altered_state[cls.neighbours[face, i-1], cls.adjacents[i-1]] = ini_state[i, cls.adjacents[i]]

		return altered_state

	@classmethod
	def as_oh(cls, states: np.ndarray):
		# This representation is already one-hot encoded, so only ravelling is done
		if len(states.shape) == 1:
			states = np.expand_dims(states, 0)
		states = torch.from_numpy(states.reshape(-1, 288)).float()
		return states

	@classmethod
	def as633(cls, state: np.ndarray):
		state68 = np.where(state == 1)[2].reshape((6, 8))
		state69 = (np.ones((9, 6)) * np.arange(6)).astype(int).T  # Nice
		for i in range(6):
			print(cls._shift_right(state68[i], cls.shifts[i]))
			state69[i, cls.map633] = cls._shift_left(state68[i], cls.shifts[i])
			print(state69[i])
		return state69.reshape((6, 3, 3))


if __name__ == "__main__":
	
	set_repr(False)
	state = Cube.get_solved()
	# print(Cube.as633(state))
	# print(Cube.stringify(state))
	# print()
	state = Cube.rotate(state, 0, True)
	state68 = np.where(state == 1)[2].reshape((6, 8))
	print(state68)
	print(Cube.stringify(state))
	# print()
	# state = Cube.rotate(state, 0, False)
	# print(Cube.as633(state))
	# print()
	
	
	# Benchmarking example
	# from src.rubiks.utils.benchmark import Benchmark
	# def test_scramble(games):
	# 	# Function is weird, as it is designed to work for both single and multithreaded benchmarks
	# 	if hasattr(games, "__iter__"):
	# 		for _ in games:
	# 			Cube.as_oh(Cube.scramble(n)[0])
	# 	else:
	# 		Cube.as_oh(Cube.scramble(n)[0])
	#
	# n = int(1e5)
	# nt = range(1, 7)
	# games = [None] * 24
	#
	# title = f"Scramble bench: {len(games)} cubes each with {n} scrambles"
	# bm = Benchmark(test_scramble, "local_benchmarks/scramble_2024", title)
	# bm.singlethreaded("Using 20 x 24 representation", games)
	# threads, times = bm.multithreaded(nt, games, "Using 20 x 24 representation")
	# bm.plot_mt_results(threads, times, f"{title} using 20 x 24")

	
	

