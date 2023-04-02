from abc import ABC
from typing import Type, List, Tuple

import ray
from ray.rllib import SampleBatch, Policy
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.algorithms.dqn.dqn_torch_policy import build_q_losses, build_q_model_and_distribution, \
    build_q_stats, adam_optimizer, grad_process_and_td_error_fn, extra_action_out_fn, \
    setup_early_mixins, before_loss_init, ComputeTDErrorMixin, compute_q_values
from ray.rllib.models import ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy import build_policy_class
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.utils import override, try_import_torch
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.torch_utils import concat_multi_gpu_td_errors, FLOAT_MIN
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType

torch, nn = try_import_torch()


def get_distribution_inputs_and_class(
    policy: Policy,
    model: ModelV2,
    input_dict: SampleBatch,
    *,
    explore: bool = True,
    is_training: bool = False,
    **kwargs
) -> Tuple[TensorType, type, List[TensorType]]:
    q_vals = compute_q_values(
        policy, model, input_dict, explore=explore, is_training=is_training
    )
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals

    original = restore_original_dimensions(input_dict[SampleBatch.OBS], policy.observation_space, tensorlib="torch")
    if not torch.all(original['allowed_actions'] == 0.0):
        mask = torch.clamp(torch.log(original['allowed_actions']), min=FLOAT_MIN)
        q_vals = torch.add(q_vals, mask)

    model.tower_stats["q_values"] = q_vals

    return (
        q_vals,
        TorchCategorical,
        []
    )


MaskedDQNTorchPolicy = build_policy_class(
    name="DQNTorchPolicy",
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DEFAULT_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)


class MaskedDQN(DQN, ABC):

    @classmethod
    @override(DQN)
    def get_default_policy_class(
            cls, config: AlgorithmConfigDict
    ) -> Type[MaskedDQNTorchPolicy]:
        if config["framework"] == "torch":
            return MaskedDQNTorchPolicy
        else:
            raise NotImplementedError('Tensorflow is not supported yet.')
