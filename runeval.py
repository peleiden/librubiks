from glob import glob as glob  # glob
from ast import literal_eval

from pelutils import Parser, set_seeds, log

from librubiks.jobs import EvalJob

train_folders = sorted(glob('data/local_train2*'))  # Stops working in the next millenium

###
# Should correspond to arguments in EvalJob
###
options = {
	'location': {
		'default':  train_folders[-1] if train_folders else '.',
		'help':     'Location to search for model and save results.\nMust use location/<run_name>/model.pt structure.',
		'type':     str,
	},
	'agents': {
		'default':  'AStar',
		'help':     'One or more space seperated agents corresponding to agent classes in librubiks.solving.agents.',
		'type':     lambda arg: arg.split(),
	},
	'scrambling': {
		'default':  'deep',
		'help':     'The scrambling depths at which to test the model. Can be given as a singel integer for one depth or\n'
					'Two space-seperated integers (given in string delimeters such as --scrambling "10 25")',
		# Ugly way to define list of two numbers or single number input or 'deep'
		'type':     lambda args: ([int(args.split()[0]), int(args.split()[1])] if len(args.split()) > 1 else [int(args), int(args)+1]) if args != 'deep' else [0],
	},
	'games': {
		'default':  500,
		'help':     'Number of games to play in evaluation for each depth, for each agent.',
		'type':     int,
	},
	'max_time': {
		'default':  0,
		'help':     'Max searching time for agent. 0 for unlimited. Evaluation is terminated when either max_time or max_states is reached.',
		'type':     float,
	},
	'max_states': {
		'default':  200_000,
		'help':     'Max number of searched states for agent per configuration. 0 for unlimited. '
					'Evaluation is terminated when either max_time or max_states is reached.',
		'type':     lambda arg: int(float(arg)),
	},
	'use_best': {
		'default':  True,
		'help':     "Set to True to use model-best.pt instead of model.pt.",
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'optimized_params': {
		'default':  False,
		'help':     'Set to True to overwrite agent params with the ones in corresponding JSON created by hyper_optim, if it exists. '
					'If True, there must only be one agent given.',
		'type':     literal_eval,
		'choices':  [True, False],
	},
	'astar_lambdas': {
		'default':  '0.2',
		'help':     'The A* search lambda parameter: How much to weight the distance from start to nodes in cost calculation. '
					'There must be as many space seperated values as there are AStar agents given.',
		'type':     lambda args: [float(x) for x in args.split()],
	},
	'astar_expansions': {
		'default':  '100',
		'help':     'The A* expansions parameter: How many nodes to expand to at a time. Can be thought of as a batch size: '
					'Higher is much faster but lower should be a bit more precise. '
					'There must be as many space seperated values as there are AStar agents given.',
		'type':     lambda args: [int(x) for x in args.split()],
	},
}

if __name__ == "__main__":
	description = r"""
___________________________________________________________________
  /_/_/_/\  ______ _      ______ _   _______ _____ _   __ _____
 /_/_/_/\/\ | ___ \ |     | ___ \ | | | ___ \_   _| | / //  ___|
/_/_/_/\/\/\| |_/ / |     | |_/ / | | | |_/ / | | | |/ / \ `--.
\_\_\_\/\/\/|    /| |     |    /| | | | ___ \ | | |    \  `--. \
 \_\_\_\/\/ | |\ \| |____ | |\ \| |_| | |_/ /_| |_| |\  \/\__/ /
  \_\_\_\/  \_| \_\_____/ \_| \_|\___/\____/ \___/\_| \_/\____/
__________________________________________________________________
Evaluate Rubiks agents using config or CLI arguments. If no location
is given, data/local_train with newest name is used. If the location
contains multiple neural networks, the deep agents are evalued for
each of them.
"""
	# TODO: Instead of this mess combining plots of different jobs, keep them seperate and allow multiple agents per job
	with log.log_errors:
		set_seeds()

		parser = Parser(options, description=description, name='eval', description_last='Tue')
		run_settings = parser.parse()
		# TODO: log.configure(...)
		jobs = [EvalJob(**settings, in_subfolder=len(run_settings)>1) for settings in run_settings]

		for job in jobs:
			job.execute()

