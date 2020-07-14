import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolour

from librubiks import envs, DataStorage
from librubiks.model import Model
from librubiks.utils import NullLogger, Logger


@dataclass
class AnalysisData(DataStorage):
	avg_value_targets: np.ndarray
	substate_val_stds: np.ndarray

class TrainAnalysis:
	"""Performs analysis of the training procedure to understand loss and training behaviour"""
	def __init__(self,
				 env: envs.Environment,
				 evaluations: np.ndarray,
				 games: int,
				 depth: int,
				 extra_evals: int,
				 reward_method: str,
				 logger: Logger = NullLogger()):
		"""Initialize containers mostly

		:param np.ndarray evaluations:  array of the evaluations performed on the model. Used for the more intensive analysis
		:param int depth: Rollout depth
		:param extra_evals: If != 0, extra evaluations are added for the first `exta_evals` rollouts
		"""

		self.env = env
		self.games = games
		self.depth = depth
		self.depths = np.arange(depth)
		self.extra_evals = min(evaluations[-1] if len(evaluations) else 0, extra_evals) #Wont add evals in the future (or if no evals are needed)
		self.evaluations = np.unique( np.append(evaluations, range( self.extra_evals )) )
		self.reward_method = reward_method

		self.orig_params = None
		self.params = None

		self.first_states = np.stack((
			self.env.get_solved(),
			*self.env.multi_rotate(self.env.repeat_state(self.env.get_solved(), self.env.action_dim), self.env.iter_actions()),
		))
		self.first_states = self.env.as_oh( self.first_states )
		self.first_state_values = list()

		self.data = AnalysisData(list(), list())
		self.param_changes = list()
		self.param_total_changes = list()

		self.log = logger
		self.log.verbose(f"Analysis of this training was enabled. Extra analysis is done for evaluations and for first {extra_evals} rollouts")

	def rollout(self, net: Model, rollout: int, value_targets: torch.Tensor):
		"""Saves statistics after a rollout has been performed for understanding the loss development

		:param torch.nn.Model net: The current net, used for saving values and policies of first 12 states
		:param rollout int: The rollout number. Used to determine whether it is evaluation time => check targets
		:param torch.Tensor value_targets: Used for visualizing value change
		"""
		# First time
		if self.params is None: self.params = net.get_params()


		if rollout in self.evaluations:
			net.eval()

			# Calculating value targets
			targets = value_targets.cpu().numpy().reshape((-1, self.depth))
			self.data.avg_value_targets.append(targets.mean(axis=0))

			# Calculating model change
			model_change = torch.sqrt((net.get_params()-self.params)**2).mean().cpu()
			model_total_change = torch.sqrt((net.get_params()-self.orig_params)**2).mean().cpu()
			self.params = net.get_params()
			self.param_changes.append(float(model_change))
			self.param_total_changes.append(model_total_change)

			#In the beginning: Calculate value given to first 12 substates
			if rollout <= self.extra_evals:
				self.first_state_values.append( net(self.first_states, policy=False, value=True).cpu().detach().numpy() )

			net.train()

	def ADI(self, values: torch.Tensor):
		"""Saves statistics after a run of ADI. """
		self.data.substate_val_stds.append(
			float(values.std(dim=1).mean())
		)

	def _get_evaluations_for_value(self):
		"""
		Returns a boolean vector of length len(self.evaluations) containing whether or not the curve should be in focus
		"""
		focus_rollouts = np.zeros(len(self.evaluations), dtype=bool)
		if len(self.evaluations) > 15:
			early_rollouts = 5
			late_rollouts = 10
			early_indices = [0, *np.unique(np.round(np.logspace(0, np.log10(self.extra_evals*2/3), early_rollouts-1)).astype(int))]
			late_indices = np.unique(np.linspace(self.extra_evals, len(self.evaluations)-1, late_rollouts, dtype=int))
			focus_rollouts[early_indices] = True
			focus_rollouts[late_indices] = True
		else:
			focus_rollouts[...] = True
		return focus_rollouts

