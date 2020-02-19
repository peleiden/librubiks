import numpy as np


class RubiksCube:

	assembled = np.zeros((6, 8, 6), dtype = np.uint8)
	for i in range(6):
		assembled[i, ..., i] = 1

	#Scrambling procudere saved as dict for reproducability 
	scrambling_procedure = {
		'N_scrambles':	(5, 10), #Tuple for scrambling random # moves in uniform(low, high)
	}

	def __init__(self):

		"""
		Shape: 6 x 8 uint8, see method three here: https://stackoverflow.com/a/55505784 
		"""

		self.state = self.assembled.copy()
		
		# The i'th index contain the neighbors of the i'th side in positive direction
		self.neighbors = np.array([
			[1, 5, 4, 2],  # Front
			[2, 3, 5, 0],  # Left
			[0, 4, 3, 1],  # Top
			[5, 1, 2, 4],  # Back
			[3, 2, 0, 5],  # Right
			[4, 0, 1, 3],  # Bottom
		])
		self.adjacents = np.array([
			[6, 7, 0],
			[2, 3, 4],
			[4, 5, 6],
			[0, 1, 2],
		])

	def move(self, face: int, pos_rev: bool):
		'''
		Performs rotation, mutates state and returns whether cube is completed
		'''
		self.state = self.rotate(self.state, face, pos_rev)
		return self.is_assembled()
		 
	def reset(self):
		'''
		Resets cube by random scramblings in accordance with self.scrambling_procedure
		'''
		self.state = self.assembled.copy()		
		self.scramble( self.scrambling_procedure["N_scrambles"] )
		
		while self.is_assembled(): self.scramble(1) # Avoid randomly solving the cube

	
	def rotate(self, current_state: np.array, face: int, pos_rev: bool):

		"""
		Performs one move on the cube, specified by the side (0-5) and whether the revolution is positive (boolean)
		"""

		# if not 0 <= face <= 5:
		# 	raise IndexError("Face should be 0-5, not %i" % face)
		altered_state = current_state.copy()

		altered_state[face] = self._shift_right(self.state[face], 2)\
			if pos_rev else self._shift_left(self.state[face], 2)
		
		ini_state = current_state[self.neighbors[face]]
		
		if pos_rev:
			for i in range(4):
				altered_state[self.neighbors[face, i], self.adjacents[i]]\
					= ini_state[i-1, self.adjacents[i-1]]
		else:
			for i in range(4):
				altered_state[self.neighbors[face, i-1], self.adjacents[i-1]]\
					= ini_state[i, self.adjacents[i]]
		
		return altered_state
		
	@staticmethod
	def _shift_left(a: np.ndarray, num_elems: int):

		return np.roll(a, -num_elems, axis = 0)
	
	@staticmethod
	def _shift_right(a: np.ndarray, num_elems: int):

		return np.roll(a, num_elems, axis = 0)

	def scramble(self, n: int):

		faces = np.random.randint(6, size = (n, ))
		dirs = np.random.randint(2, size = (n, )).astype(bool)

		for face, d in zip(faces, dirs):
			self.state = self.rotate(self.state, face, d) #Uses rotate instead of move as checking for victory is not needed here
		
		return faces, dirs
	
	def is_assembled(self):
		
		return (self.state == self.assembled).all()
		
	def __str__(self):
		
		return str(self.as68())
	
	def as68(self):

		"""
		Un-encodes one-hot and returns self.state as 6x8 matrix
		"""

		state68 = np.where(self.state == 1)[2].reshape((6, 8))
		return state68

if __name__ == "__main__":
	
	# Benchmarking example
	from utils.benchmark import Benchmark
	def test_scramble(games):
		# Function is weird, as it is designed to work for both single and multithreaded benchmarks
		if hasattr(games, "__iter__"):
			for _ in games:
				rube = RubiksCube()
				rube.scramble(n)
		else:
			rube = RubiksCube()
			rube.scramble(n)
	
	n = int(1e5)
	nt = range(1, 7)
	games = np.empty(24)

	title = f"Scramble bench: {games.size} cubes each with {n} scrambles"
	bm = Benchmark(test_scramble, "local_benchmarks/scramble_example", title)
	bm.singlethreaded("", games)
	threads, times = bm.multithreaded(nt, games)
	bm.plot_mt_results(threads, times, title)

	
	

