from abc import ABC

from gym.spaces import Dict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()


class DiscreteMaskModel(TorchModelV2, nn.Module, ABC):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name
                 ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "allowed_actions" in orig_space.spaces
                and "observation" in orig_space.spaces
        )
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.internal_model = ComplexInputNetwork(
            orig_space['observation'],
            action_space,
            num_outputs,
            model_config,
            name + '_internal',
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict['obs']['allowed_actions']

        logits, _ = self.internal_model({'obs': input_dict['obs']['observation']}, state, seq_lens)

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
