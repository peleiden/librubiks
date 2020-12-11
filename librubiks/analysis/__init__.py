import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolour

from pelutils import DataStorage

from librubiks import envs, no_grad
from librubiks.model import Model
from pelutils import log


@dataclass
class AnalysisData(DataStorage):
    env_key: str
    reward_method: str
    extra_evals: int
    depths: np.ndarray
    first_state_values: np.ndarray
    avg_value_targets: np.ndarray
    substate_val_stds: np.ndarray
    evaluations: np.ndarray

    subfolder = "analysis-data"
    json_name = "analysis-data.json"


class TrainAnalysis:
    """Performs analysis of the training procedure to understand loss and training behaviour"""
    def __init__(
        self,
        env: envs.Environment,
        evaluations: np.ndarray,
        games: int,
        depth: int,
        extra_evals: int,
        reward_method: str,
    ):
        """
        Initialize containers mostly

        :param np.ndarray evaluations:  array of the evaluations performed on the model. Used for the more intensive analysis
        :param int depth: Rollout depth
        :param extra_evals: If != 0, extra evaluations are added for the first `exta_evals` rollouts
        """
        self.env = env
        self.games = games
        self.depth = depth

        self.orig_params = None
        self.params = None

        self.first_states = np.stack((
            self.env.get_solved(),
            *self.env.multi_act(self.env.repeat_state(self.env.get_solved()), self.env.iter_actions()),
        ))
        self.first_states = self.env.multi_as_oh(self.first_states)

        if evaluations.size:
            extra_evals = min(int(evaluations[-1]), extra_evals)  # All rollouts are analyzed up until extra_evals
        self.possible_evaluation_rollouts = np.unique(np.append(evaluations, range(extra_evals)))

        self.data = AnalysisData(
            env_key = self.env.key,
            reward_method = reward_method,
            extra_evals = extra_evals,
            depths = np.arange(self.depth),
            first_state_values = np.array([]),
            avg_value_targets = np.array([]),
            substate_val_stds = np.array([]),
            evaluations = np.array([]),
        )
        self.param_changes = list()
        self.param_total_changes = list()

        log.verbose(f"Analysis of this training was enabled. Extra analysis is done for evaluations and for first {extra_evals} rollouts")

    @no_grad
    def rollout(self, net: Model, rollout: int, value_targets: torch.Tensor):
        """
        Saves statistics after a rollout has been performed for understanding the loss development

        :param torch.nn.Model net: The current net, used for saving values and policies of first 12 states
        :param rollout int: The rollout number. Used to determine whether it is evaluation time => check targets
        :param torch.Tensor value_targets: Used for visualizing value change
        """
        # First time
        if self.params is None:
            self.params = net.get_params()

        if rollout in self.possible_evaluation_rollouts:
            net.eval()
            self.data.evaluations = np.append(self.data.evaluations, rollout)

            # Calculating value targets
            targets = value_targets.cpu().numpy().reshape((-1, self.depth))
            if self.data.avg_value_targets.size:
                self.data.avg_value_targets = np.append(
                    self.data.avg_value_targets,
                    [targets.mean(axis=0)],
                    axis=0,
                )
            else:
                self.data.avg_value_targets = np.array([targets.mean(axis=0)])

            # Calculating model change
            model_change = torch.sqrt((net.get_params()-self.params)**2).mean().cpu()
            model_total_change = torch.sqrt((net.get_params()-self.orig_params)**2).mean().cpu()
            self.params = net.get_params()
            self.param_changes.append(float(model_change))
            self.param_total_changes.append(model_total_change)

            # In the beginning: Calculate value given to solved state and substates
            if rollout <= self.data.extra_evals:
                try:
                    self.data.first_state_values = np.vstack([
                        self.data.first_state_values,
                        np.expand_dims(net(self.first_states).cpu().detach().numpy(), 0)
                    ])
                except ValueError:
                    self.data.first_state_values = np.expand_dims(net(self.first_states).cpu().detach().numpy(), 0)

            net.train()

    def ADI(self, values: torch.Tensor):
        """Saves statistics after a run of ADI."""
        self.data.substate_val_stds = np.append(
            self.data.substate_val_stds,
            float(values.std(dim=1).mean()),
        )

