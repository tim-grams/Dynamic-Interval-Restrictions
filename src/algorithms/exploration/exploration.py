from typing import Union, Optional

from gym import Space
from ray.rllib.models import ModelV2
from ray.rllib.utils import PublicAPI, override, try_import_torch, PiecewiseSchedule
from ray.rllib.utils.exploration import GaussianNoise, EpsilonGreedy
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.schedules import Schedule
from ray.rllib.utils.typing import TensorType

from src.algorithms.distributions.dynamic_interval import TorchDynamicIntervals

torch, nn = try_import_torch()


@PublicAPI
class DynamicIntervalGaussianNoise(GaussianNoise):
    """ mixture of gaussian noise clipped into an interval and epsilon greedy
    """
    def __init__(
            self,
            action_space: Space,
            *,
            framework: str,
            model: ModelV2,
            random_timesteps: int = 1000,
            stddev: float = 0.1,
            initial_scale: float = 1.0,
            final_scale: float = 0.02,
            scale_timesteps: int = 10000,
            scale_schedule: Optional[Schedule] = None,
            initial_epsilon: float = 1.0,
            final_epsilon: float = 0.05,
            warmup_timesteps: int = 0,
            epsilon_timesteps: int = int(1e5),
            epsilon_schedule: Optional[Schedule] = None,
            **kwargs
    ):
        GaussianNoise.__init__(
            self,
            action_space=action_space,
            framework=framework,
            model=model,
            random_timesteps=random_timesteps,
            stddev=stddev,
            initial_scale=initial_scale,
            final_scale=final_scale,
            scale_timesteps=scale_timesteps,
            scale_schedule=scale_schedule,
            **kwargs
        )

        self.epsilon_schedule = from_config(
            Schedule, epsilon_schedule, framework=framework
        ) or PiecewiseSchedule(
            endpoints=[
                (0, initial_epsilon),
                (warmup_timesteps, initial_epsilon),
                (warmup_timesteps + epsilon_timesteps, final_epsilon),
            ],
            outside_value=final_epsilon,
            framework=self.framework,
        )

    @override(GaussianNoise)
    def get_exploration_action(
        self,
        *,
        action_distribution: TorchDynamicIntervals,
        timestep: Union[int, TensorType],
        explore: bool = True
    ):
        assert self.framework == 'torch'

        self.last_timestep = (
            timestep if timestep is not None else self.last_timestep + 1
        )

        if explore:
            if self.last_timestep < self.random_timesteps:
                action = action_distribution.random_sample()
            else:
                epsilon = self.epsilon_schedule(self.last_timestep)
                random_actions = action_distribution.random_sample()

                det_actions = action_distribution.deterministic_sample()
                batch_size = det_actions.size(0)
                scale = self.scale_schedule(self.last_timestep)
                gaussian_sample = scale * torch.normal(
                    mean=torch.zeros(det_actions.size()), std=self.stddev
                ).to(self.device)
                exploit_action = det_actions + gaussian_sample
                exploit_action[:, 0] = torch.where(exploit_action[:, 0] > action_distribution.chosen_interval[:, 1],
                                                   action_distribution.chosen_interval[:, 1], exploit_action)
                exploit_action[:, 0] = torch.where(exploit_action[:, 0] < action_distribution.chosen_interval[:, 0],
                                                   action_distribution.chosen_interval[:, 0], exploit_action)

                action = torch.where(
                    torch.empty((batch_size,)).uniform_().to(self.device) < epsilon,
                    random_actions,
                    exploit_action,
                )
        else:
            action = action_distribution.deterministic_sample()

        logp = torch.zeros((action.size()[0],), dtype=torch.float32, device=self.device)
        return action, logp


@PublicAPI
class DynamicIntervalEpsilonGreedy(EpsilonGreedy):
    """ Epsilon greedy implementation that selects only randomly from allowed intervals
    """
    @override(EpsilonGreedy)
    def get_exploration_action(
        self,
        *,
        action_distribution: TorchDynamicIntervals,
        timestep: Union[int, TensorType],
        explore: Optional[Union[bool, TensorType]] = True,
    ):
        assert self.framework == 'torch'

        self.last_timestep = timestep

        exploit_action = action_distribution.deterministic_sample()
        batch_size = exploit_action.size(0)
        logp = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)

        if explore:
            epsilon = self.epsilon_schedule(self.last_timestep)

            random_actions = action_distribution.random_sample()

            action = torch.where(
                torch.empty((batch_size,)).uniform_().to(self.device) < epsilon,
                random_actions,
                exploit_action,
            )

            return action, logp

        return exploit_action, logp
