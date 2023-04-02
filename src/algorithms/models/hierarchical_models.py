from abc import ABC

import numpy as np
from gym.spaces import Dict, Box
from ray.rllib import SampleBatch
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

torch, nn = try_import_torch()


class IntervalLevelModel(TorchModelV2, nn.Module, ABC):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs
                 ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "allowed_actions" in orig_space.spaces
                and "observation" in orig_space.spaces
        )
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        attention_dim = kwargs['attention_dim']
        attention_heads = kwargs['attention_heads']
        postprocessing_hiddens = kwargs['postprocessing_hiddens']
        postprocessing_activation = kwargs['postprocessing_activation']

        self.QKV_projection = nn.Linear(2, 3 * attention_dim)
        self.interval_attention = nn.MultiheadAttention(attention_dim, attention_heads,
                                                        batch_first=True)

        self.observation_encoder = ComplexInputNetwork(
                orig_space['observation']['observation'],
                Box(low=-np.inf, high=np.inf, shape=(attention_dim,)),
                attention_dim,
                model_config,
                'observation_encoder'
            )

        self.postprocessing = nn.Sequential()
        ins = attention_dim
        for i, n in enumerate(postprocessing_hiddens):
            self.postprocessing.add_module(
                'postprocessing',
                SlimFC(
                    ins,
                    n,
                    activation_fn=postprocessing_activation
                )
            )
            ins = n
        self.postprocessing.add_module(
            'postprocessing',
            SlimFC(
                attention_dim,
                1,
                activation_fn='relu'
            )
        )

    def forward(self, input_dict, state, seq_lens):
        allowed_intervals = input_dict[SampleBatch.OBS]['observation']['allowed_actions']
        mask = allowed_intervals[:, :, 0] == allowed_intervals[:, :, 1]
        mask[torch.all(mask, dim=1)] = ~mask[torch.all(mask, dim=1)]

        q, k, v = self.QKV_projection(allowed_intervals).chunk(3, -1)
        attention_out = self.interval_attention(q, k, v, mask, need_weights=False)[0]

        encoder_out = self.observation_encoder(SampleBatch(
            obs=input_dict[SampleBatch.OBS]['observation']['observation'], _is_training=True
        ))[0]

        hidden = torch.mul(attention_out, encoder_out.reshape(encoder_out.size(0), 1, encoder_out.size(1)))

        logits = self.postprocessing(hidden)
        logits = logits.reshape(logits.size(0), logits.size(1))

        inf_mask = torch.clamp(torch.log(~mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        return self.observation_encoder.value_function()
