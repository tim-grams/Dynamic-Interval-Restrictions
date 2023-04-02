from abc import ABC
from typing import Type, Optional, Dict, Any

import gym
from ray.rllib import SampleBatch, Policy
from ray.rllib.algorithms.ddpg import DDPG
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.algorithms.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.evaluation import Episode
from ray.rllib.utils import override
from ray.rllib.utils.typing import AlgorithmConfigDict


class ReplacementDDPGPolicy(DDPGTorchPolicy, ABC):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: AlgorithmConfigDict
    ):
        DDPGTorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config
        )

    @override(DDPGTorchPolicy)
    def postprocess_trajectory(
            self,
            sample_batch: SampleBatch,
            other_agent_batches: Optional[Dict[Any, SampleBatch]] = None,
            episode: Optional[Episode] = None,
    ) -> SampleBatch:
        infos = sample_batch[SampleBatch.INFOS]
        if isinstance(infos[0], Dict):
            sample_batch[SampleBatch.ACTIONS] = [list(info['Executed']) for info in infos]
        return postprocess_nstep_and_prio(
            self, sample_batch, other_agent_batches, episode
        )


class ReplacementDDPG(DDPG, ABC):

    @override(DDPG)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return ReplacementDDPGPolicy
        else:
            raise NotImplementedError('Tensorflow is not supported yet.')
