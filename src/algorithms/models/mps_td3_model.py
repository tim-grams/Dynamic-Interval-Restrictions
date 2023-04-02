from abc import ABC
from typing import Optional, List

import gym
import numpy as np
from gym.spaces import Box
from ray.rllib.algorithms.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from src.algorithms.models.models import BatchScaling

torch, nn = try_import_torch()


class MPSTD3Model(DDPGTorchModel, ABC):

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            actor_hiddens: Optional[List[int]] = None,
            actor_hidden_activation: str = "relu",
            critic_hiddens: Optional[List[int]] = None,
            critic_hidden_activation: str = "relu",
            twin_q: bool = False,
            add_layer_norm: bool = False,
            **kwargs
    ):
        assert isinstance(action_space, Box)
        DDPGTorchModel.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            actor_hiddens,
            actor_hidden_activation,
            critic_hiddens,
            critic_hidden_activation,
            twin_q,
            add_layer_norm
        )

        self.scaling_layer = BatchScaling(self.action_space)

    def max_q_action(self, observation: TensorType, intervals: TensorType) -> TensorType:
        max_action = torch.zeros(observation.size(0), 1)
        max_interval = torch.zeros(intervals.size(0), 2)
        max_q = torch.full((observation.size(0), 1), -np.inf)

        for i in range(0, int(intervals.size(1) / 2), 2):
            new_observation = torch.cat([intervals[:, i:i + 2], observation], 1)
            action = self.policy_model(new_observation)
            q_value = self.q_model(torch.cat([new_observation, action], -1))

            empty = intervals[:, i] != intervals[:, i + 1]
            improved = q_value > max_q
            update = torch.logical_and(empty.reshape(-1, 1), improved)

            max_action[update] = action[update]
            max_interval[torch.cat([update, update], 1)] = intervals[:, i:i+2][torch.cat([update, update], 1)]
            max_q[update] = q_value[update]

        scaled_max_action = self.scaling_layer(max_action, max_interval.reshape(max_interval.size(0), -1, 2))

        return scaled_max_action, max_interval,
