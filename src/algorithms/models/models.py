import numpy as np
from gym.spaces import Box
from ray.rllib import Policy
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.torch.noop import TorchNoopModel
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.exploration import ParameterNoise
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class BatchScaling(nn.Module):
    """ Parameterizable scaling layer

    Args:
        action_space (gym.Space): action space defining the boundaries for the scaling function
    """

    def __init__(self, action_space):
        super().__init__()
        assert isinstance(action_space, Box)

        self.bounded = np.logical_and(
            action_space.bounded_above, action_space.bounded_below
        ).any()

        if self.bounded:
            self.low = action_space.low[0]
            self.high = action_space.high[0]
        else:
            self.low = 0.0
            self.high = 1.0

    def forward(self, action, interval_action_space):
        if not self.bounded:
            action = nn.Sigmoid()(action)

        scaled_actions = torch.add(
            torch.mul(
                (action - self.low).reshape(action.size(0), -1), (torch.subtract(
                    interval_action_space[:, :, 1], interval_action_space[:, :, 0]) / (self.high - self.low))),
            interval_action_space[:, :, 0])

        if self.bounded:
            return torch.min(
                torch.max(scaled_actions, interval_action_space[:, :, 0]),
                interval_action_space[:, :, 1])
        else:
            return scaled_actions


def make_models(policy: Policy, template) -> ModelV2:
    """ Builds model object for a policy

    Args:
        policy (Policy): Policy to configure the model for
        template: Template of the model to build

    Returns:
        model
    """
    from src.algorithms.models.pam_model import PAMModel

    if policy.config["use_state_preprocessor"]:
        raise NotImplementedError('Preprocessor with this model is not supported yet.')

    if template is PAMModel:
        num_outputs = int(np.product(policy.observation_space.shape))
        internal_action_space = policy.action_space
        num_outputs_parameter_module = int(np.product(policy.internal_observation_space.shape))
    else:
        internal_action_space = None
        num_outputs = int(np.product(policy.internal_observation_space.shape))
        num_outputs_parameter_module = None

    add_layer_norm = (
            isinstance(getattr(policy, "exploration", None), ParameterNoise)
            or policy.config["exploration_config"]["type"] == "ParameterNoise"
    )

    config = {
        'obs_space': policy.observation_space,
        'action_space': policy.internal_action_space if template is PAMModel else policy.action_space,
        'num_outputs': num_outputs,
        'model_config': policy.config["model"],
        'framework': policy.config["framework"],
        'model_interface': template,
        'default_model': TorchNoopModel,
        'name': 'actor_model',
        'num_outputs_parameter_module': num_outputs_parameter_module,
        'internal_observation_space': policy.internal_observation_space if hasattr(policy, 'internal_observation_space') else None,
        'internal_action_space': internal_action_space,
        'actor_hidden_activation': policy.config["actor_hidden_activation"] if 'actor_hidden_activation' in policy.config else None,
        'actor_hiddens': policy.config["actor_hiddens"] if 'actor_hiddens' in policy.config else None,
        'critic_hidden_activation': policy.config["critic_hidden_activation"] if 'critic_hidden_activation' in policy.config else None,
        'critic_hiddens': policy.config["critic_hiddens"] if 'critic_hiddens' in policy.config else None,
        'interval_hiddens': policy.config['interval_hiddens'] if 'interval_hiddens' in policy.config else None,
        'interval_hidden_activation': policy.config['interval_hidden_activation'] if 'interval_hidden_activation' in policy.config else None,
        'attention_hidden_dimension': policy.config['attention_hidden_dimension'] if 'attention_hidden_dimension' in policy.config else None,
        'q_hiddens': policy.config['q_hiddens'] if 'q_hiddens' in policy.config else None,
        'p_hiddens': policy.config['p_hiddens'] if 'p_hiddens' in policy.config else None,
        'dueling': policy.config["dueling"] if 'dueling' in policy.config else None,
        'num_atoms': policy.config["num_atoms"] if 'num_atoms' in policy.config else None,
        'use_noisy': policy.config["noisy"] if 'noisy' in policy.config else None,
        'v_min': policy.config["v_min"] if 'v_min' in policy.config else None,
        'v_max': policy.config["v_max"] if 'v_max' in policy.config else None,
        'sigma0': policy.config["sigma0"] if 'sigma0' in policy.config else None,
        'add_layer_norm': add_layer_norm,
        'twin_q': policy.config["twin_q"] if 'twin_q' in policy.config else None
    }

    model = ModelCatalog.get_model_v2(**config)

    config['name'] = 'target_model'
    policy.target_model = ModelCatalog.get_model_v2(**config)

    return model


def interval(out: TensorType, intervals: TensorType):
    """ Returns the interval in which a value lies

    Args:
        out (TensorType): Value to search the interval for
        intervals (TensorType): Tensor of action space intervals

    Returns:
        interval (TensorType): Interval in which out is contained
    """
    results = torch.ones(intervals.size(dim=0), 2)

    for i in range(0, int(intervals.size(1) / 2)):
        is_in_interval = torch.logical_and(intervals[:, i * 2] <= out[:, 0],
                                           out[:, 0] <= intervals[:, i * 2 + 1])
        results[is_in_interval] = intervals[:, i * 2:i * 2 + 2][is_in_interval]

    return results


def interval_index(out: TensorType, intervals: TensorType):
    """ Returns the index of the interval in which a value lies

    Args:
        out (TensorType): Value to search the interval for
        intervals (TensorType): Tensor of action space intervals

    Returns:
        interval (TensorType): index of the interval in which out is contained
    """
    results = torch.zeros(out.size(dim=0))

    for i in range(0, int(intervals.size(1) / 2)):
        is_in_interval = torch.logical_and(intervals[:, i * 2 + 1] <= out[:, 0],
                                           intervals[:, i * 2] != intervals[:, i * 2 + 1])
        results[is_in_interval] += 1

    return results.long()
