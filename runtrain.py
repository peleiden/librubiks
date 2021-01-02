from pelutils import Parser, set_seeds, log

from librubiks.envs import environments
from librubiks.jobs import TrainJob

options = {
    'rollouts': {
        'default':  500,
        'help':     'Number of complete rollouts each consisting of simulating play through the Auto Didactic method AND performing minibatch learning on the resulting ',
        'type':     int,
    },
    'rollout-games': {
        'default':  1000,
        'help':     'Number of simulated games, using the Auto Didactic method, in each rollout',
        'type':     int,
    },
    'rollout-depth': {
        'default':  100,
        'help':     'Number of random rotations applied to each game in the Auto Didactic simulation',
        "type":     int,
    },
    'arch': {
        'default':  'fc',
        'help':     'Network architecture. \'fc\' for fully-connected or \'res\' for fully-connected with residual blocks',
        'type':     str,
        'choices':  ['fc', 'res'],
    },
    'alpha-update': {
        'default':  0,
        'help':     'alpha is set to max{alpha + alpha_update, 1} update_interval times during training. 0 for weighted and 1 for unweighted.\n'+
                    'alpha is a parameter that interpolates between no weighting of training examples (alpha=1) and weighting training examples by 1/depth (alpha=0).',
        'type':     float,
    },
    'update-interval': {
        'default':  50,
        'help':     'How often alpha and lr are updated. First update is performed when rollout == update_interval. Set to 0 for never',
        'type':     int,
    },
    'reward-method' : {
        'default':  'lapanfix',
        'help':     'Which way to set target values near goal state. "paper" forces nothing and does not train on goal state. ' +
                    '"lapanfix" trains on goalstate and forces it = 0. "schultzfix" forces substates for goal to 0 and does not train on goal state. ' +
                    '"reward0" changes reward for transitioning to goal state to 0 and does not train on goal state.',
        'type':     str,
        'choices':  ['paper', 'lapanfix', 'schultzfix', 'reward0'],
    },
    'batch-size': {
        'default':  1000,
        'help':     'Number of training examples to be used in each parameter update, e.g. minibatch size for gradient descent' +
                    'Note: Training is done on rollout_games*rollout_depth examples, so batch_size must be <= this',
        'type':     int
    },
    'optim-fn': {
        'default':  'Adam',
        'help':     'Name of optimization function corresponding to class in torch.optim',
        'type':     str,
    },
    'lr': {
        'default':  1e-5,
        'help':     'Learning rate of parameter update',
        'type':     float,
    },
    'gamma': {
        'default':  1,
        'help':     'Learning rate reduction parameter. Learning rate is set updated as lr <- gamma * lr 100 times during training',
        'type':     float,
    },
    'evaluation-interval': {
        'default':  50,
        'help':     'An evaluation is performed every evaluation_interval rollouts. Set to 0 for never',
        'type':     int,
    },
    'tau': {
        'default':  1,
        'help':     'Network change parameter for generating training data. If tau=1, use newest network for ADI',
        'type':     float,
    },
    'nn-init': {
        'default':  'glorot',
        'help':     'Initialialization strategy for the NN. Choose either "glorot", "he" or write a number. If a number is given, the network is initialized to this constant.',
        'type':     str,
    },
    'env-key': {
        'default':  'cube2024',
        'help':     'The environment that should be trained on',
        'type':     str,
        'choices':  list(environments.keys()),
    },
    'analysis': {
        'action':   'store_true',
        'help':     'If true, analysis of model changes, value and loss behaviour is done in each rollout and ADI pass',
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

Start one or more Reinforcement Learning training session(s)
on the Rubik's Cube using config or CLI arguments.
"""
    with log.log_errors:
        set_seeds()

        parser = Parser(options, description=description, name='train', description_last=True)
        parsley = parser.parse()
        parser.document_settings()
        jobs = [TrainJob(**settings) for settings in parsley]
        for job in jobs:
            job.execute()
