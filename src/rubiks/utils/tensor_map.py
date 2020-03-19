import numpy as np


class State:
	# Initialized in solved state
	corners = np.arange(8)
	corner_orientations = np.zeros(8)
	sides = np.arange(12)
	side_orientations = np.zeros(12)
	def __str__(self):
		return f"Corners:             {[int(x) for x in self.corners]}\n" + \
			   f"Corner orientations: {[int(x) for x in self.corner_orientations]}\n" + \
			   f"Sides:               {[int(x) for x in self.sides]}\n" + \
			   f"Side orientations:   {[int(x) for x in self.side_orientations]}"

class Actions:
	F: (
		(0, 1, 2, 3, 0),  # Corner mapping
		(0, 1, 2, 3, 0),  # Side mapping
		0,  # Corner orientation static - other two switch
		False,  # Side orientation switch
	)
	B: (
		(4, 7, 6, 5, 4),
		(8, 11, 10, 9, 8),
		0,
		False,
	)
	T: (
		(0, 3, 7, 4, 0),
		(0, 7, 8, 4, 0),
		1,
		True,
	)
	D: (
		(1, 5, 6, 2, 1),
		(2, 5, 10, 6, 2),
		1,
		True,
	)
	L: (
		(0, 4, 5, 1, 0),
		(1, 4, 9, 5, 1),
		2,
		False,
	)
	R: (
		(7, 3, 2, 6, 7),
		(3, 6, 11, 7, 3),
		2,
		False,
	)


# Midlertidig dokumentation
# Hver cubie har en position, 0-7 for hjørner og 0-11 for sider
# Hver rotation mapper hver relevant position til en anden
# Derudover mappes orientering
# Den sticker på cubien, der holdes øje med, har en orientering 0-2 for hjørner og 0-1 for sider
# Tallet angiver en "prioritet". Den side, der har flest stickers i løst tilstand, har højst prioritet
# F/B starter med 8 hver, T_D med 2 hver og L/R med 0 hver
# På grund af symmetrien vil der aldrig være overlap
# Alle maps er angivet for positiv omløbsretning
# Set forfra
# Forreste lag | Midterste lag | Bagerste lag
# h0 s0 h3     | s4    s7      | h4 s8  h7
# s1    s3     |               | s9     s11
# h1 s2 h2     | s5    s6      | h5 s10 h6
# TODO: Make direct tensor mappings
# Should be 2x24 mapping tensor and probably generated by script
# TODO: Human readable version

def _get_cornes_pos(pos: int, orientation: int):
	return pos * 3 + orientation

def _get_side_pos(pos: int, orientation: int):
	return pos * 2 + orientation

def _get_tensor_map():
	# Returns two maps
	# The first is positive revolution, second is negative
	# Each is a six long list containg 2x24 mapping tensors
	# Order is F, B, T, D, L, R
	# Row one for corners [:8] and row two for sides [8:]
	# Value at index i should be added to i in state representation
	actions = [Actions.F, Actions.B, Actions.T, Actions.D, Actions.L, Actions.R]
	map_pos = list()
	map_neg = list()
	for i in range(6):
		action = actions[i]
		pos = np.zeros((2, 24))
		neg = np.zeros((2, 24))
		
		
		
	
	return map_pos, map_neg


if __name__ == "__main__":
	print(_get_tensor_map())


