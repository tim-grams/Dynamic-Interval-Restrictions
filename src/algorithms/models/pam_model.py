from abc import ABC
from typing import Sequence, Union, List, Dict

import gym
import numpy as np
from gym.spaces import Discrete, Box
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from src.algorithms.models.models import BatchScaling

torch, nn = try_import_torch()


class PAMModel(DQNTorchModel, ABC):

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 num_outputs_parameter_module: int,
                 internal_action_space: gym.spaces.Space,
                 q_hiddens: Sequence[int] = (256,),
                 p_hiddens: Sequence[int] = (256,),
                 dueling: bool = False,
                 dueling_activation: str = 'relu',
                 parameter_module_activation: str = 'relu',
                 num_atoms: int = 1,
                 use_noisy: bool = False,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 sigma0: float = 0.5,
                 add_layer_norm: bool = False,
                 **kwargs
                 ):
        DQNTorchModel.__init__(self,
                               obs_space=obs_space,
                               action_space=action_space,
                               num_outputs=num_outputs,
                               model_config=model_config,
                               name=name,
                               q_hiddens=q_hiddens,
                               dueling=dueling,
                               dueling_activation=dueling_activation,
                               num_atoms=num_atoms,
                               use_noisy=use_noisy,
                               v_min=v_min,
                               v_max=v_max,
                               sigma0=sigma0,
                               add_layer_norm=add_layer_norm)
        assert isinstance(action_space, Discrete)

        self.internal_action_space = internal_action_space
        self.dueling = dueling

        ins = num_outputs_parameter_module

        parameter_module = nn.Sequential()

        for i, n in enumerate(p_hiddens):
            if use_noisy:
                parameter_module.add_module(
                    f'parameter_A_{i}',
                    NoisyLayer(
                        ins, n, sigma0=sigma0, activation=parameter_module_activation
                    )
                )
            else:
                parameter_module.add_module(
                    f'parameter_A_{i}',
                    SlimFC(ins, n, activation_fn=parameter_module_activation)
                )
                if add_layer_norm:
                    parameter_module.add_module(
                        f'LayerNorm_A_{i}',
                        nn.LayerNorm(n)
                    )
            ins = n

        if use_noisy:
            parameter_module.add_module(
                'A',
                NoisyLayer(
                    ins, action_space.n * self.num_atoms, sigma0, activation=None
                )
            )
        elif p_hiddens:
            parameter_module.add_module(
                'A',
                SlimFC(
                    ins, action_space.n * self.num_atoms, activation_fn=None
                )
            )

        self.parameter_module = parameter_module

        self.scaling_layer = BatchScaling(Box(low=-np.inf, high=np.inf, shape=self.internal_action_space.shape))

    def get_parameters(self, observation, available_intervals):
        parameters = self.parameter_module(observation)

        available_intervals = available_intervals.reshape(1, -1, 2).repeat(parameters.size(0), 1, 1)

        return self.scaling_layer(parameters, available_intervals)

    def q_variables(
            self, as_dict=False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        if as_dict:
            return {
                **self.advantage_module.state_dict(),
                **self.value_module.state_dict(),
            } if self.dueling else self.advantage_module.state_dict()
        return list(self.advantage_module.parameters()) + (
            list(self.value_module.parameters())
        ) if self.dueling else list(self.advantage_module.parameters())

    def p_variables(
            self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        if as_dict:
            return self.parameter_module.state_dict()
        return list(self.parameter_module.parameters())
