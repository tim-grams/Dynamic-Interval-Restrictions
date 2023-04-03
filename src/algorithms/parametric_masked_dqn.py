from abc import ABC
from typing import Optional, Any, Dict, Tuple, Type, List

import gym
import numpy as np
import ray
from gym.spaces import Box, Discrete
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import ComputeTDErrorMixin
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import postprocess_nstep_and_prio, PRIO_WEIGHTS
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import override, try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import apply_grad_clipping, concat_multi_gpu_td_errors, FLOAT_MIN, \
    reduce_mean_ignore_inf, huber_loss, softmax_cross_entropy_with_logits
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType, ModelGradients

from src.algorithms.distributions.dynamic_interval import TorchDynamicIntervals
from src.algorithms.models.parametric_masking_model import ParametricMaskingModel
from src.algorithms.models.models import make_models

torch, nn = try_import_torch()


class ParametricMaskedDQNTorchPolicy(TargetNetworkMixin, ComputeTDErrorMixin, TorchPolicyV2, ABC):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: AlgorithmConfigDict,
    ):
        assert isinstance(action_space, Box)

        config = dict(ray.rllib.algorithms.ddpg.ddpg.DDPGConfig().to_dict(), **config)

        self.global_step = 0

        # Define Parameter observation space of shape [observation] out of original [allowed_actions + observation]
        orig_space = getattr(observation_space, "original_space", observation_space)
        new_shape = (observation_space.shape[0] - orig_space['allowed_actions'].shape[0],)
        self.internal_observation_space = Box(low=-np.inf, high=np.inf, shape=new_shape)

        # Define DQN action space of shape [number_of_bins]
        self.internal_action_space = Discrete(orig_space['allowed_actions'].shape[0])

        # Define continuous allowed actions
        self.bin_width = ((action_space.high - action_space.low) / orig_space['allowed_actions'].shape[0])[0]
        self.allowed_actions = torch.cat(
            [torch.arange(action_space.low[0], action_space.high[0], self.bin_width).reshape(-1, 1),
             torch.arange(action_space.low[0] + self.bin_width, action_space.high[0] + self.bin_width,
                          self.bin_width).reshape(-1, 1)], 1)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ComputeTDErrorMixin.__init__(self)

        self._initialize_loss_from_dummy_batch()

        TargetNetworkMixin.__init__(self)

    @override(TorchPolicyV2)
    def make_model_and_action_dist(
            self,
    ) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        model = make_models(self, ParametricMaskingModel)
        return model, TorchDynamicIntervals

    @override(TorchPolicyV2)
    def apply_gradients(self, gradients: ModelGradients) -> None:
        self._q_optimizer.step()
        self._p_optimizer.step()

        self.global_step += 1

    @override(TorchPolicyV2)
    def optimizer(
            self,
    ) -> List["torch.optim.Optimizer"]:

        self._q_optimizer = torch.optim.Adam(
            params=self.model.q_variables(), lr=self.config["q_lr"], eps=1e-7
        )

        self._p_optimizer = torch.optim.Adam(
            params=self.model.p_variables(), lr=self.config["p_lr"], eps=1e-7
        )

        return [self._q_optimizer, self._p_optimizer]

    @override(TorchPolicyV2)
    def action_distribution_fn(
            self,
            model: ModelV2,
            *,
            obs_batch: TensorType,
            state_batches: TensorType,
            is_training: bool = False,
            **kwargs
    ) -> Tuple[TensorType, type, List[TensorType]]:
        model_out, _ = model(
            SampleBatch(obs=obs_batch[SampleBatch.CUR_OBS], _is_training=is_training)
        )

        # Split in [observation] and [available_intervals]
        observation = model_out[:, self.internal_action_space.n:]
        mask = model_out[:, :self.internal_action_space.n]

        # Predict parameterization and q values
        parameters = model.get_parameters(observation, self.allowed_actions)
        q_vals = self.compute_q_values(self.model, torch.cat([observation, parameters], 1))[0]

        # Apply mask and gather action and interval
        action, selected_interval = self.action_and_interval(parameters, q_vals, mask)
        available_intervals = self.available_intervals(mask)

        dist_inputs = torch.cat([action, selected_interval, available_intervals], 1)

        return dist_inputs, TorchDynamicIntervals, []

    @override(TorchPolicyV2)
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[TorchDistributionWrapper],
            train_batch: SampleBatch,
    ) -> List[TensorType]:
        target_model = self.target_models[model]

        input_dict = SampleBatch(
            obs=train_batch[SampleBatch.OBS], _is_training=True
        )
        input_dict_next = SampleBatch(
            obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True
        )

        model_out_t, _ = model(input_dict, [], None)
        model_out_tp1, _ = target_model(input_dict_next, [], None)

        # Model observation and parameters
        obs_t = model_out_t[:, self.internal_action_space.n:]
        p_t = model.get_parameters(obs_t, self.allowed_actions)

        # Target model observation and parameters
        obs_tp1 = model_out_tp1[:, self.internal_action_space.n:]
        p_tp1 = target_model.get_parameters(obs_tp1, self.allowed_actions)

        # Q values
        q_t, q_logits_t, q_probs_t = self.compute_q_values(model, torch.cat([obs_t, p_t], 1))
        q_tp1, q_logits_tp1, q_probs_tp1 = self.compute_q_values(target_model, torch.cat([obs_tp1, p_tp1], 1))

        # Mask all actions which were not evaluated
        one_hot_selection = torch.nn.functional.one_hot(
            torch.abs((train_batch[SampleBatch.ACTIONS].reshape(1, -1) / self.bin_width)[0].long()), self.internal_action_space.n
        )

        # Q loss computation
        q_t_selected = torch.sum(
            torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
            * one_hot_selection,
            1,
        )

        if self.config["double_q"]:
            p_tp1 = model.get_parameters(obs_tp1, self.allowed_actions)
            q_tp1_using_online_net, q_logits_tp1_using_online_net, q_dist_tp1_using_online_net = self.compute_q_values(
                model,
                torch.cat([obs_tp1, p_tp1], 1)
            )
            q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
            q_tp1_best_one_hot_selection = torch.nn.functional.one_hot(
                q_tp1_best_using_online_net, self.internal_action_space.n
            )
            q_tp1_best = torch.sum(
                torch.where(
                    q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                )
                * q_tp1_best_one_hot_selection,
                1,
            )
            q_probs_tp1_best = torch.sum(
                q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
            )
        else:
            q_tp1_best_one_hot_selection = torch.nn.functional.one_hot(
                torch.argmax(q_tp1, 1), self.internal_action_space.n
            )
            q_tp1_best = torch.sum(
                torch.where(
                    q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                )
                * q_tp1_best_one_hot_selection,
                1,
            )
            q_probs_tp1_best = torch.sum(
                q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
            )

        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

        q_t_selected_target = train_batch[SampleBatch.REWARDS] + self.config['gamma'] ** self.config[
            'n_step'] * q_tp1_best_masked

        td_error = q_t_selected - q_t_selected_target.detach()
        q_loss = torch.mean(
            train_batch[PRIO_WEIGHTS].float() * huber_loss(td_error)
        )

        # Parameter loss as sum of negative q values
        q_t_p_loss = torch.sum(
            torch.where(
                q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device)
            ),
            1,
        )

        p_loss = torch.mul(torch.mean(
            train_batch[PRIO_WEIGHTS].float() * q_t_p_loss
        ), -1)

        model.tower_stats["q_t"] = q_t
        model.tower_stats["td_error"] = td_error
        model.tower_stats["actor_loss"] = p_loss
        model.tower_stats["critic_loss"] = q_loss

        return [q_loss, p_loss]

    @override(TorchPolicyV2)
    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[Any, SampleBatch]] = None,
            episode: Optional[Episode] = None,
    ) -> SampleBatch:
        return postprocess_nstep_and_prio(
            self, sample_batch, other_agent_batches, episode
        )

    @override(TorchPolicyV2)
    def extra_grad_process(
            self, optimizer: torch.optim.Optimizer, loss: TensorType
    ) -> Dict[str, TensorType]:
        return apply_grad_clipping(self, optimizer, loss)

    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
        return dict({LEARNER_STATS_KEY: {}}, **fetches)

    def action_and_interval(self, parameters, q_vals, mask):
        if not torch.all(mask == 0):
            mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
            q_vals = torch.add(q_vals, mask)

        q_max = torch.argmax(q_vals, dim=1).reshape(-1, 1)

        action = torch.gather(parameters, 1, q_max)

        lower_bounds = torch.mul(q_max, self.bin_width)
        upper_bounds = torch.add(torch.mul(q_max, self.bin_width), self.bin_width)

        return action, torch.cat([lower_bounds, upper_bounds], 1)

    def available_intervals(self, mask):
        return torch.mul(torch.cat([mask.reshape(mask.size(0), -1, 1), mask.reshape(mask.size(0), -1, 1)], 2),
                         self.allowed_actions).reshape(mask.size(0), -1)

    def compute_q_values(self, model, model_out):
        if self.config["num_atoms"] > 1:
            action_scores, z, support_logits_per_action, logits, probs_or_logits = model.get_q_value_distributions(
                model_out)
        else:
            action_scores, logits, probs_or_logits = model.get_q_value_distributions(model_out)

        if self.config["dueling"]:
            state_score = model.get_state_value(model_out)
            if self.config["num_atoms"] > 1:
                support_logits_per_action_mean = torch.mean(support_logits_per_action, dim=1)
                support_logits_per_action_centered = support_logits_per_action - torch.unsqueeze(
                    support_logits_per_action_mean, dim=1)
                support_logits_per_action = torch.unsqueeze(state_score, dim=1) + support_logits_per_action_centered
                support_prob_per_action = nn.functional.softmax(support_logits_per_action, dim=-1)
                value = torch.sum(z * support_prob_per_action, dim=-1)
                logits = support_logits_per_action
                probs_or_logits = support_prob_per_action
            else:
                advantages_mean = reduce_mean_ignore_inf(action_scores, 1)
                advantages_centered = action_scores - torch.unsqueeze(advantages_mean, 1)
                value = state_score + advantages_centered
        else:
            value = action_scores

        return value, logits, probs_or_logits

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        q_t = torch.stack(self.get_tower_stats("q_t"))
        stats = {
            "actor_loss": torch.mean(torch.stack(self.get_tower_stats("actor_loss"))),
            "critic_loss": torch.mean(torch.stack(self.get_tower_stats("critic_loss"))),
            "mean_q": torch.mean(q_t),
            "max_q": torch.max(q_t),
            "min_q": torch.min(q_t),
            "td_error": torch.mean(torch.stack(self.get_tower_stats("td_error")))
        }
        return convert_to_numpy(stats)


class ParametricMaskedDQNConfig(DQNConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or DQN)
        self.p_hiddens = [256, 256]
        self.q_hiddens = [256, 256]
        self.q_lr = 5e-4
        self.p_lr = 5e-4


class ParametricMaskedDQN(DQN, ABC):

    @classmethod
    @override(DQN)
    def get_default_config(cls):
        return ParametricMaskedDQNConfig()

    @classmethod
    @override(DQN)
    def get_default_policy_class(
            cls, config: AlgorithmConfigDict
    ) -> Type[ParametricMaskedDQNTorchPolicy]:
        if config["framework"] == "torch":
            return ParametricMaskedDQNTorchPolicy
        else:
            raise NotImplementedError('Tensorflow is not supported yet.')
