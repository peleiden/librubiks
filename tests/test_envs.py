import numpy as np
import torch

from tests import MainTest

from librubiks import gpu, envs
from librubiks.envs.cube_maps import SimpleState, get_corner_pos, get_side_pos

env2024 = envs.get_env("cube2024")
env686 = envs.get_env("cube686")

class TestRubiksCube(MainTest):

	def test_init(self):
		state = env2024.get_solved()
		assert env2024.is_solved(state)
		assert env2024.get_solved_instance().shape == (20,)
		state = env686.get_solved()
		assert env686.is_solved(state)
		assert env686.get_solved_instance().shape == (6, 8, 6)

	def test_cube(self):
		self._rotation_tests(env2024)
		self._multi_act_test(env2024)
		self._rotation_tests(env686)
		self._multi_act_test(env686)

	def _rotation_tests(self, env: envs.Environment):
		state = env.get_solved()
		for action in env.action_space:
			state = env.act(state, action)
		# Test that stringify and by extension _as633 works on solved state
		state = env.get_solved()
		assert env.stringify(state) == "\n".join([
			"      2 2 2            ",
			"      2 2 2            ",
			"      2 2 2            ",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"      3 3 3            ",
			"      3 3 3            ",
			"      3 3 3            ",
		])
		# Perform moves and check if states are assembled/not assembled as expected
		moves = (6, 0, 6, 7, 2, 3, 9, 8, 1, 0)#((0, 1), (0, 0), (0, 1), (1, 1), (2, 0), (3, 0))
		assembled = (False, True, False, False, False, False, False, False, False, True)
		for m, a in zip(moves, assembled):
			state = env.act(state, m)
			assert a == env.is_solved(state)

		# Perform move and check if it fits with how the string representation would look
		state = env.get_solved()
		state = env.act(state, 6)
		assert env.stringify(state) == "\n".join([
			"      2 2 2            ",
			"      2 2 2            ",
			"      5 5 5            ",
			"4 4 2 0 0 0 3 5 5 1 1 1",
			"4 4 2 0 0 0 3 5 5 1 1 1",
			"4 4 2 0 0 0 3 5 5 1 1 1",
			"      4 4 4            ",
			"      3 3 3            ",
			"      3 3 3            ",
		])

		# Performs all moves and checks if result fits with how it theoretically should look
		state = env.get_solved()
		moves = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
		# moves = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
		# 		 (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1))
		assembled = (False, False, False, False, False, False,
					 False, False, False, False, False, False)
		for m, a in zip(moves, assembled):
			state = env.act(state, m)
			assert a == env.is_solved(state)
		assert env.stringify(state) == "\n".join([
			"      2 0 2            ",
			"      5 2 4            ",
			"      2 1 2            ",
			"4 2 4 0 2 0 5 2 5 1 2 1",
			"4 4 4 0 0 0 5 5 5 1 1 1",
			"4 3 4 0 3 0 5 3 5 1 3 1",
			"      3 1 3            ",
			"      5 3 4            ",
			"      3 0 3            ",
		])

	def _multi_act_test(self, env: envs.Environment):
		states = np.array([env.get_solved()]*5)
		for _ in range(10):
			actions = env.sample_actions(5)
			states_classic = np.array([env.act(state, action) for state, action in zip(states, actions)])
			states = env.multi_act(states, actions)
			assert (states_classic == states).all()
	
	def test_rev_actions(self):
		state = env2024.get_solved()
		for action in env2024.action_space:
			state = env2024.act(state, action)
			state = env2024.act(state, env2024.rev_action(action))
			assert env2024.is_solved(state)
		actions = env2024.sample_actions(5)
		for action in actions:
			state = env2024.act(state, action)
		for action in env2024.multi_rev_action(actions)[::-1]:
			state = env2024.act(state, action)
		assert env2024.is_solved(state)

	def test_scramble(self):
		state = env2024.get_solved()
		state, actions = env2024.scramble(1)
		assert not env2024.is_solved(state)

		state = env2024.get_solved()
		state, actions = env2024.scramble(20)
		assert not env2024.is_solved(state)
		for action in env2024.multi_rev_action(actions)[::-1]:
			state = env2024.act(state, action)
		assert env2024.is_solved(state)

	def test_iter_actions(self):
		actions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] * 2)
		assert np.all(actions == env2024.iter_actions(2))

	def test_as_oh(self):
		state = env2024.get_solved()
		oh = env2024.as_oh(state)
		supposed_state = torch.zeros(20, 24, device=gpu)
		corners = [get_corner_pos(c, o) for c, o
				   in zip(SimpleState.corners.tolist(), SimpleState.corner_orientations.tolist())]
		supposed_state[torch.arange(8), corners] = 1
		sides = [get_side_pos(s, o) for s, o
				 in zip(SimpleState.sides.tolist(), SimpleState.side_orientations.tolist())]
		supposed_state[torch.arange(8, 20), sides] = 1
		assert (supposed_state.flatten() == oh).all()

	def test_as633(self):
		state = env2024._as633(env2024.get_solved())
		target633 = list()
		for i in range(6):
			target633.append(np.ones((3, 3)) * i)
		target633 = np.array(target633)
		assert (state == target633).all()

