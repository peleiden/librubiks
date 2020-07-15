import numpy as np
import torch

from tests import MainTest

from librubiks import gpu, envs
from librubiks.model import Model, ModelConfig, create_net, load_net, save_net

from librubiks.solving.agents import Agent, RandomSearch, BFS, ValueSearch, AStar

env = envs.get_env("cube2024")


def _action_queue_test(state, agent, sol_found):
	assert all([0 <= x < env.action_dim for x in agent.action_queue])
	for action in agent.action_queue:
		state = env.act(state, env.action_space[action])
	assert env.is_solved(state) == sol_found


class TestAgents(MainTest):
	def test_agents(self):
		net = create_net(ModelConfig(env.key))
		agents = [
			RandomSearch(),
			BFS(),
			ValueSearch(net),
		]
		for s in agents: self._test_agents(s)

	def _test_agents(self, agent: Agent):
		state, _ = env.scramble(4)
		solution_found  = agent.search(state, .05)
		for action in agent.action_queue:
			state = env.act(state, action)
		assert solution_found == env.is_solved(state)


class TestAStar(MainTest):

	#TODO: More indepth testing: Especially of updating of parents

	def test_agent(self):
		test_params = {
			(0, 10),
			(0.5, 2),
			(1, 1),
		}
		net = create_net(ModelConfig(env)).eval()
		for params in test_params:
			agent = AStar(net, *params)
			self._can_win_all_easy_games(agent)
			agent.reset("Tue", "Herlau")
			assert not len(agent.indices)
			assert not len(agent.open_queue)

	def _can_win_all_easy_games(self, agent):
		state, _ = env.scramble(2, force_not_solved=True)
		is_solved = agent.search(state, time_limit=1)
		if is_solved:
			for action in agent.action_queue:
				state = env.act(state, env.action_space[action])
			assert env.is_solved(state)

	def test_expansion(self):
		net = create_net(ModelConfig(env.key)).eval()
		init_state, _ = env.scramble(3)
		agent = AStar(net, lambda_=0.1, expansions=5)
		agent.search(init_state, time_limit=1)
		init_idx = agent.indices[init_state.tostring()]
		assert init_idx == 1
		assert agent.G[init_idx]  == 0
		for action in env.action_space:
			substate = env.act(init_state, action)
			idx = agent.indices[substate.tostring()]
			assert agent.G[idx] == 1
			assert agent.parents[idx] == init_idx

	def test_cost(self):
		net = create_net(ModelConfig(env)).eval()
		games = 5
		states, _ = env.sequence_scrambler(games, 1, True)
		agent = AStar(net, lambda_=1, expansions=2)
		agent.reset(1, 1)
		i = []
		for i, _ in enumerate(states): agent.G[i] = 1
		cost = agent.cost(states, i)
		assert cost.shape == (games,)



