from typing import Union

import typing
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID


class CustomMetricsCallback(DefaultCallbacks):

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: typing.Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2],
            **kwargs,
    ):
        episode.user_data['invalid_actions'] = 0.0

        episode.user_data['solved'] = 0.0

        episode.user_data['level'] = 0.0

        episode.user_data['fraction_allowed_actions'] = []
        episode.user_data['allowed_interval_length'] = []
        episode.user_data['number_intervals'] = []

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: typing.Optional[typing.Dict[PolicyID, Policy]] = None,
            episode: Union[Episode, EpisodeV2],
            **kwargs,
    ):
        info = episode.last_info_for()
        if info is None:
            info = episode.last_info_for('interval_level_agent')
        if info is None:
            info = episode.last_info_for('action_level_agent')

        if 'invalid' in info and info['invalid']:
            episode.user_data['invalid_actions'] += 1.0
        if 'solved' in info and info['solved']:
            episode.user_data['solved'] += 1.0
        if 'level' in info:
            episode.user_data['level'] = info['level']
        if 'fraction_allowed_actions' in info:
            episode.user_data['fraction_allowed_actions'].append(info['fraction_allowed_actions'])
        if 'allowed_interval_length' in info:
            episode.user_data['allowed_interval_length'].append(info['allowed_interval_length'])
        if 'number_intervals' in info:
            episode.user_data['number_intervals'].append(info['number_intervals'])

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: typing.Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            **kwargs,
    ):
        episode.custom_metrics['count_invalid_actions'] = episode.user_data['invalid_actions']
        episode.custom_metrics['solved'] = episode.user_data['solved']
        episode.custom_metrics['level'] = episode.user_data['level']

        episode.custom_metrics['fraction_allowed_actions'] = episode.user_data['fraction_allowed_actions']
        episode.custom_metrics['allowed_interval_length'] = episode.user_data['allowed_interval_length']
        episode.custom_metrics['number_intervals'] = episode.user_data['number_intervals']

        episode.custom_metrics['fraction_allowed_actions_avg'] = sum(episode.user_data['fraction_allowed_actions']) / episode.length
        episode.custom_metrics['fraction_allowed_actions_max'] = max(episode.user_data['fraction_allowed_actions'])
        episode.custom_metrics['fraction_allowed_actions_min'] = min(episode.user_data['fraction_allowed_actions'])

        episode.custom_metrics['number_intervals_avg'] = sum(episode.user_data['number_intervals']) / episode.length
        episode.custom_metrics['number_intervals_max'] = max(episode.user_data['number_intervals'])
        episode.custom_metrics['number_intervals_min'] = min(episode.user_data['number_intervals'])

        episode.custom_metrics['allowed_interval_length_avg'] = sum(episode.user_data['allowed_interval_length']) / episode.length
        episode.custom_metrics['allowed_interval_length_max'] = max(episode.user_data['allowed_interval_length'])
        episode.custom_metrics['allowed_interval_length_min'] = min(episode.user_data['allowed_interval_length'])
