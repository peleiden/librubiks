from dataclasses import dataclass

import numpy as np
import torch

from librubiks import gpu, no_grad, reset_cuda
from librubiks.utils import Logger, NullLogger, unverbose, TickTock, TimeUnit

from librubiks import envs, DataStorage
from librubiks.analysis import TrainAnalysis, AnalysisData
from librubiks.model import Model

from librubiks.solving.agents import DeepAgent
from librubiks.solving.evaluation import Evaluator


@dataclass
class TrainData(DataStorage):
	name: str
	states_per_rollout: int  # Number of states backpropped each rollout
	rollouts: int  # Number of rollouts (or epochs)
	rollout_games: int  # Number of games per rollout
	rollout_depth: int  # TODO: Consider if this should be changed to fit arbitrary training method
	losses: np.ndarray  # Loss for each rollout
	evaluation_rollouts: np.ndarray  # Array containing every rollout on which an evaluation was performed
	evaluation_results: np.ndarray  # Solve shares
	eval_games: int  # Number of games each evaluation
	eval_time: float  # Time given to solve each game during evaluation

	subfolder = "train-data"
	json_name = "train-data.json"


class BatchFeedForward:
	"""
	This class handles feedforwarding large batches that would otherwise cause memory overflow
	It works by splitting it into smaller batches, if it encounters a memory error
	Only works when gradient should be tracked
	"""

	def __init__(self, net: Model, data_points: int, logger=NullLogger(), initial_batches=1, increase_factor=2):
		self.net = net
		self.data_points = data_points
		self.log = logger
		self.batches = initial_batches
		self.increase_factor = increase_factor

	@no_grad
	def forward(self, x: torch.tensor) -> torch.tensor:
		while True:
			try:
				output_parts = [self.net(x[slice_]) for slice_ in self._get_slices()]
				output = torch.cat(output_parts)
				break
			except RuntimeError as e:  # Usually caused by running out of vram. If not, the error is still raised, else batch size is reduced
				if "alloc" not in str(e):
					raise e
				self.log.verbose(f"Intercepted RuntimeError {e}\nIncreasing number of ADI feed forward batches from "
								 f"{self.batches} to {self.batches*self.increase_factor}")
				self._more_batches()
		return output

	def update_net(self, net: Model):
		self.net = net

	def __call__(self, x: torch.tensor) -> torch.tensor:
		return self.forward(x)

	def _more_batches(self):
		self.batches *= self.increase_factor

	def _get_slices(self):
		slice_size = self.data_points // self.batches + 1
		# Final slice may have overflow, however this is simply ignored when indexing
		slices = [slice(i*slice_size, (i+1)*slice_size) for i in range(self.batches)]
		return slices


class Train:

	def __init__(
		self,
		env: envs.Environment,
		rollouts: int,
		rollout_games: int,
		rollout_depth: int,
		batch_size: int,  # Required to be > 1 when training with batchnorm
		optim_fn,
		lr: float,
		gamma: float,
		alpha_update: float,
		tau: float,
		update_interval: int,
		reward_method: str,
		agent: DeepAgent,
		evaluator: Evaluator,
		evaluation_interval: int,
		name: str,
		with_analysis: bool,
		value_criterion = torch.nn.MSELoss,
		logger: Logger = NullLogger(),
	):
		"""
		Sets up evaluation array, instantiates critera and stores and documents settings

		:param bool with_analysis: If true, a number of statistics relating to loss behaviour and model output are stored.
		:param float alpha_update: alpha <- alpha + alpha_update every update_interval rollouts (excl. rollout 0)
		:param float gamma: lr <- lr * gamma every update_interval rollouts (excl. rollout 0)
		:param float tau: How much of the new network to use to generate ADI data
		"""
		self.env = env

		self.rollouts = rollouts
		self.states_per_rollout = rollout_games * rollout_depth
		self.batch_size = self.states_per_rollout if not batch_size else batch_size
		self.rollout_games = rollout_games
		self.rollout_depth = rollout_depth
		self.adi_ff_batches = 1  # Number of batches used for feedforward in ADI_traindata. Used to limit vram usage
		self.reward_method = reward_method

		self.agent = agent

		self.tau = tau
		self.alpha_update = alpha_update
		self.lr	= lr
		self.gamma = gamma
		self.update_interval = update_interval  # How often alpha and lr are updated

		self.optim = optim_fn
		self.value_criterion = value_criterion(reduction='none')

		self.evaluator = evaluator
		self.evaluation_interval = evaluation_interval
		self.log = logger
		self.log("\n".join([
			"Created trainer",
			f"Alpha update: {self.alpha_update:.2f}",
			f"Learning rate and gamma: {self.lr} and {self.gamma}",
			f"Learning rate and alpha will update every {self.update_interval} rollouts: lr <- {self.gamma:.4f} * lr and alpha += {self.alpha_update:.4f}"\
				if self.update_interval else "Learning rate and alpha will not be updated during training",
			f"Optimizer:       {self.optim}",
			f"Value criterion: {self.value_criterion}",
			f"Rollouts:        {self.rollouts}",
			f"Batch size:      {self.batch_size}",
			f"Rollout games:   {self.rollout_games}",
			f"Rollout depth:   {self.rollout_depth}",
			f"alpha update:    {self.alpha_update}",
		]))

		self.name = name
		self.with_analysis = with_analysis

		self.tt = TickTock()

	def train(self, net: Model) -> (Model, Model, TrainData, AnalysisData):
		""" Training loop: generates data, optimizes parameters, evaluates (sometimes) and repeats.

		Trains `net` for `self.rollouts` rollouts each consisting of `self.rollout_games` games and scrambled  `self.rollout_depth`.
		The network is evaluated for each rollout number in `self.evaluations` according to `self.evaluator`.
		Stores multiple performance and training results.

		:param torch.nn.Model net: The network to be trained. Must accept input consistent with self.env.get_oh_size()
		:return: The network after all evaluations and the network with the best evaluation score (win fraction)
		:rtype: (torch.nn.Model, torch.nn.Model, TrainData, AnalysisData)
		"""

		self.tt.reset()
		self.tt.tick()
		self.states_per_rollout = self.rollout_depth * self.rollout_games
		losses = np.zeros(self.rollouts)
		evaluation_rollouts = self._get_evaluation_rollouts()
		evaluation_results = list()
		self.log(f"Beginning training. Optimization is performed in batches of {self.batch_size}")
		self.log("\n".join([
			f"Rollouts: {self.rollouts}",
			f"Each consisting of {self.rollout_games} games with a depth of {self.rollout_depth}",
			f"Evaluations: {len(evaluation_rollouts)}",
			f"Rough upper bound on total evaluation time during training: "
			f"{TickTock.stringify_time(len(evaluation_rollouts)*self.evaluator.approximate_time(), TimeUnit.minute)}",
		]))
		best_solve = 0
		best_net = net.clone()
		self.agent.update_net(net)
		if self.with_analysis:
			analysis = TrainAnalysis(self.env,
									 evaluation_rollouts,
									 self.rollout_games,
									 self.rollout_depth,
									 extra_evals=100,
									 reward_method=self.reward_method,
									 logger=self.log)
			analysis.orig_params = net.get_params()
		else:
			analysis = None

		generator_net = net.clone()
		ff = BatchFeedForward(generator_net, self.rollout_games * self.rollout_depth * self.env.action_dim, self.log)

		alpha = 1 if self.alpha_update == 1 else 0
		optimizer = self.optim(net.parameters(), lr=self.lr)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.gamma)

		for rollout in range(self.rollouts):
			reset_cuda()

			generator_net = self._update_gen_net(generator_net, net) if self.tau != 1 else net
			generator_net.eval()
			ff.update_net(generator_net)

			self.tt.profile("ADI training data")
			training_data, value_targets, loss_weights = ADI_traindata(self, ff, alpha, analysis)
			self.tt.profile("To cuda")
			training_data = training_data.to(gpu)
			value_targets = value_targets.to(gpu)
			loss_weights = loss_weights.to(gpu)
			self.tt.end_profile("To cuda")
			self.tt.end_profile("ADI training data")

			reset_cuda()

			self.tt.profile("Training loop")
			net.train()
			batches = self._get_batches(self.states_per_rollout, self.batch_size)
			for batch in batches:
				optimizer.zero_grad()
				value_pred = net(training_data[batch])

				loss = torch.mean(self.value_criterion(value_pred.squeeze(), value_targets[batch]) * loss_weights[batch])
				loss.backward()
				optimizer.step()
				losses[rollout] += loss.detach().cpu().numpy().mean() / len(batches)

			self.tt.end_profile("Training loop")

			# Updates learning rate and alpha
			if rollout and self.update_interval and rollout % self.update_interval == 0:
				if self.gamma != 1:
					lr_scheduler.step()
					lr = optimizer.param_groups[0]["lr"]
					self.log(f"Updated learning rate from {lr/self.gamma:.2e} to {lr:.2e}")
				if (alpha + self.alpha_update <= 1 or np.isclose(alpha + self.alpha_update, 1)) and self.alpha_update:
					alpha += self.alpha_update
					self.log(f"Updated alpha from {alpha-self.alpha_update:.2f} to {alpha:.2f}")
				elif alpha < 1 and alpha + self.alpha_update > 1 and self.alpha_update:
					self.log(f"Updated alpha from {alpha:.2f} to 1")
					alpha = 1

			if self.log.is_verbose() or rollout in (np.linspace(0, 1, 20)*self.rollouts).astype(int):
				self.log(f"Rollout {rollout} completed with mean loss {losses[rollout]}")

			if self.with_analysis:
				self.tt.profile("Analysis of rollout")
				analysis.rollout(net, rollout, value_targets)
				self.tt.end_profile("Analysis of rollout")

			if rollout in evaluation_rollouts:
				net.eval()

				self.agent.update_net(net)
				self.tt.profile(f"Evaluating using agent {self.agent}")
				with unverbose:
					eval_results = self.evaluator.eval([self.agent])
				eval_solved = (eval_results.sol_lengths != -1).mean()
				evaluation_results.append(eval_solved)
				self.tt.end_profile(f"Evaluating using agent {self.agent}")

				if eval_solved > best_solve:
					best_solve = eval_solved
					best_net = net.clone()
					self.log(f"Updated best net with solve rate {eval_solved*100:.2f} % at depth {self.evaluator.scrambling_depths}")

		self.log.section("Finished training")
		if evaluation_rollouts.size:
			self.log(f"Best net solves {best_solve*100:.2f} % of games at depth {self.evaluator.scrambling_depths}")
		self.log.verbose("Training time distribution")
		self.log.verbose(self.tt)
		total_time = self.tt.tock()
		eval_time = self.tt.profiles[f'Evaluating using agent {self.agent}'].sum() if evaluation_rollouts.size else 0
		train_time = self.tt.profiles["Training loop"].sum()
		adi_time = self.tt.profiles["ADI training data"].sum()
		nstates = self.rollouts * self.rollout_games * self.rollout_depth * self.env.action_dim
		states_per_sec = int(nstates / (adi_time+train_time))
		self.log("\n".join([
			f"Total running time:               {self.tt.stringify_time(total_time, TimeUnit.second)}",
			f"- Training data for ADI:          {self.tt.stringify_time(adi_time, TimeUnit.second)} or {adi_time/total_time*100:.2f} %",
			f"- Training time:                  {self.tt.stringify_time(train_time, TimeUnit.second)} or {train_time/total_time*100:.2f} %",
			f"- Evaluation time:                {self.tt.stringify_time(eval_time, TimeUnit.second)} or {eval_time/total_time*100:.2f} %",
			f"States witnessed incl. substates: {TickTock.thousand_seps(nstates)}",
			f"- Per training second:            {TickTock.thousand_seps(states_per_sec)}",
		]))

		traindata = TrainData(
			name                = self.name,
			states_per_rollout  = self.states_per_rollout,
			rollouts            = self.rollouts,
			rollout_games       = self.rollout_games,
			rollout_depth       = self.rollout_depth,
			losses              = losses,
			evaluation_rollouts = evaluation_rollouts,
			evaluation_results  = np.array(evaluation_results),
			eval_games          = self.evaluator.n_games,
			eval_time           = self.evaluator.max_time,
		)

		return net, best_net, traindata, analysis.data if analysis is not None else None

	def _update_gen_net(self, generator_net: Model, net: Model):
		"""Create a network with parameters weighted by self.tau"""
		self.tt.profile("Creating generator network")
		genparams, netparams = generator_net.state_dict(), net.state_dict()
		new_genparams = dict(genparams)
		for pname, param in netparams.items():
			new_genparams[pname].data.copy_(
				self.tau * param.data.to(gpu) + (1-self.tau) * new_genparams[pname].data.to(gpu)
			)
		generator_net.load_state_dict(new_genparams)
		self.tt.end_profile("Creating generator network")
		return generator_net.to(gpu)

	@staticmethod
	def _get_batches(size: int, bsize: int):
		"""
		Generates indices for batch
		"""
		nbatches = int(np.ceil(size/bsize))
		idcs = np.arange(size)
		np.random.shuffle(idcs)
		batches = [slice(batch*bsize, (batch+1)*bsize) for batch in range(nbatches)]
		batches[-1] = slice(batches[-1].start, size)
		return batches

	def _get_evaluation_rollouts(self) -> np.ndarray:

		# Perform evaluation every evaluation_interval and after last rollout
		if self.evaluation_interval:
			evaluation_rollouts = np.arange(0, self.rollouts, self.evaluation_interval) - 1
			if self.evaluation_interval == 1:
				evaluation_rollouts = evaluation_rollouts[1:]
			else:
				evaluation_rollouts[0] = 0
			if self.rollouts-1 != evaluation_rollouts[-1]:
				evaluation_rollouts = np.append(evaluation_rollouts, self.rollouts-1)
		else:
			evaluation_rollouts = np.array([])

		return evaluation_rollouts


@no_grad
def ADI_traindata(train: Train, ff: BatchFeedForward, alpha: float, analysis: TrainAnalysis) -> (torch.tensor, torch.tensor, torch.tensor):
	""" Training data generation

	Implements Autodidactic Iteration as per McAleer, Agostinelli, Shmakov and Baldi, "Solving the Rubik's Cube Without Human Knowledge" section 4.1
	Loss weighting is dependant on `self.loss_weighting`.


	:return: Games * sequence_length number of observations divided in four arrays
		- states contains the rubiks state for each data point
		- policy_targets and value_targets contains optimal value and policy targets for each training point
		- loss_weights contains the weight for each training point (see weighted samples subsection of McAleer et al paper)
	"""
	#TODO: Convert `train` use to using the arguments with relevant data

	train.tt.profile("Scrambling")
	# Only include solved state in training if using Max Lapan convergence fix
	states, oh_states = train.env.sequence_scrambler(
		train.rollout_games,
		train.rollout_depth,
		with_solved = train.reward_method == 'lapanfix'
	)
	train.tt.end_profile()

	# Keeps track of solved states - Max Lapan's convergence fix
	solved_scrambled_states = train.env.multi_is_solved(states)

	# Generates possible substates for all scrambled states. Shape: n_states*action_dim x *Cube_shape
	train.tt.profile("ADI substates")
	substates = train.env.multi_act(np.repeat(states, train.env.action_dim, axis=0), train.env.iter_actions(len(states)))
	train.tt.end_profile()
	train.tt.profile("One-hot encoding")
	substates_oh = train.env.multi_as_oh(substates)
	train.tt.end_profile()

	train.tt.profile("Reward")
	solved_substates = train.env.multi_is_solved(substates)
	# Reward for won state is 1 normally but 0 if running with reward0
	rewards = (torch.zeros if train.reward_method == 'reward0' else torch.ones)\
		(*solved_substates.shape)
	rewards[~solved_substates] = -1
	rewards = rewards.unsqueeze(1)
	train.tt.end_profile()

	# Generates policy and value targets
	train.tt.profile("ADI feedforward")
	values = ff(substates_oh).cpu()
	train.tt.end_profile()

	train.tt.profile("Calculating targets")
	values += rewards
	values = values.reshape(-1, 12)
	value_targets = values[np.arange(len(values)), torch.argmax(values, dim=1)]
	if train.reward_method == 'lapanfix':
		# Trains on goal state, sets goalstate to 0
		value_targets[solved_scrambled_states] = 0
	elif train.reward_method == 'schultzfix':
		# Does not train on goal state, but sets first 12 substates to 0
		first_substates = np.zeros(len(states), dtype=bool)
		first_substates[np.arange(0, len(states), train.rollout_depth)] = True
		value_targets[first_substates] = 0
	train.tt.end_profile()

	# Weighting examples according to alpha
	weighted = np.tile(1 / np.arange(1, train.rollout_depth+1), train.rollout_games)
	unweighted = np.ones_like(weighted)
	ws, us = weighted.sum(), len(unweighted)
	loss_weights = ((1-alpha) * weighted / ws + alpha * unweighted / us) * (ws + us)

	if analysis is not None:
		train.tt.profile("ADI analysis")
		analysis.ADI(values)
		train.tt.end_profile()
	return oh_states, value_targets, torch.from_numpy(loss_weights).float()

