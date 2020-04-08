# import os, sys
# from flask import Flask, request, jsonify
# from flask_restful import Resource, Api
# from flask_cors import CORS

# from ast import literal_eval
# import numpy as np
# import torch

# from src.rubiks.solving.agents import Agent, DeepAgent
# from src.rubiks.solving.search import RandomDFS, BFS, PolicySearch, MCTS
# from src.rubiks.cube.cube import Cube

# app = Flask(__name__)
# api = Api(app)
# CORS(app)
# print(os.getcwd(), sys.path[0])
# net_loc = "trained_model"
# agents = [
# 	{ "name": "Tilfældige træk", "agent": Agent(RandomDFS()) },
# 	{ "name": "BFS", "agent": Agent(BFS()) },
# 	{ "name": "Deterministisk politik", "agent": DeepAgent(PolicySearch.from_saved(net_loc, False)) },
# 	{ "name": "Stokastisk politik", "agent": DeepAgent(PolicySearch.from_saved(net_loc, True)) },
# 	{ "name": "Dybkube", "agent": DeepAgent(MCTS.from_saved(net_loc)) },
# ]

# def as69(state: np.ndarray):
# 	# Nice
# 	return Cube.as633(state).reshape((6, 9))


# def get_state_dict(state: np.ndarray or list):
# 	state = np.array(state)
# 	return jsonify({
# 		"state": as69(state).tolist(),
# 		"state20": state.tolist(),
# 	})

# @app.route("/")
# def index():
#     return "<a href='https://asgerius.github.io/rl-rubiks'>Gå til hovedside</a>"

# @app.route("/info")
# def get_info():
# 	return jsonify({
# 		"cuda": torch.cuda.is_available(),
# 		"agents": [x["name"] for x in agents],
# 	})

# @app.route("/solved")
# def get_solved():
# 	return get_state_dict(Cube.get_solved())

# @app.route("/action", methods=["POST"])
# def act():
# 	data = literal_eval(request.data.decode("utf-8"))
# 	action = data["action"]
# 	state = data["state20"]
# 	new_state = Cube.rotate(state, *Cube.action_space[action])
# 	return get_state_dict(new_state)

# @app.route("/scramble", methods=["POST"])
# def scramble():
# 	data = literal_eval(request.data.decode("utf-8"))
# 	depth = data["depth"]
# 	state = np.array(data["state20"])
# 	states = []
# 	for _ in range(depth):
# 		action = np.random.randint(Cube.action_dim)
# 		state = Cube.rotate(state, *Cube.action_space[action])
# 		states.append(state)
# 	finalOh = states[-1]
# 	states = np.array([as69(state) for state in states])
# 	return jsonify({
# 		"states": states.tolist(),
# 		"finalState20": finalOh.tolist(),
# 	})

# @app.route("/solve", methods=["POST"])
# def solve():
# 	data = literal_eval(request.data.decode("utf-8"))
# 	time_limit = data["timeLimit"]
# 	agent = agents[data["agentIdx"]]["agent"]
# 	state = np.array(data["state20"])
# 	solution_found, steps = agent.generate_action_queue(state, time_limit)
# 	actions = [agent.action() for _ in range(steps)]
# 	states = []
# 	if solution_found:
# 		for action in actions:
# 			state = Cube.rotate(state, *action)
# 			states.append(state)
# 	finalOh = states[-1] if states else state
# 	states = np.array([as69(state) for state in states])
# 	return jsonify({
# 		"solution": solution_found,
# 		"actions": actions,
# 		"states": states.tolist(),
# 		"finalState20": finalOh.tolist(),
# 	})


# if __name__ == "__main__":
# 	app.run()

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_restful import Api
from flask_cors import CORS
from ast import literal_eval
print(os.listdir())

app = Flask(__name__)
api = Api(app)
CORS(app)

# with open("src/test.txt") as f:
#     d = f.read()

data = {
	"array": np.arange(5).tolist()
}

@app.route("/")
def index():
	return "<h1>Hello there</h1>"

@app.route("/array")
def get_array():
	return jsonify(data)

if __name__ == "__main__":
	app.run()

