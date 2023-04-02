from ray.rllib.utils.exploration import GaussianNoise

td3_config = {
    'twin_q': True, 'policy_delay': 1, 'smooth_target_policy': True, 'target_noise': 0.20259988679187632,
    'target_noise_clip': 0.19479747379650628, 'actor_hiddens': [256], 'actor_hidden_activation': 'relu',
    'critic_hiddens': [512], 'critic_hidden_activation': 'relu', 'critic_lr': 0.00023393886918355352,
    'actor_lr': 6.704166165213775e-05, 'tau': 0.004380116483022898, 'l2_reg': 0.0005592053899614199,
    'exploration_config': {'type': GaussianNoise,
                           'stddev': 1.0, 'random_timesteps': 268, 'initial_scale': 9.591442106885852,
                           'final_scale': 0.40296313388905725, 'scale_timesteps': 3073},
    'replay_buffer_config': {'capacity': 66258,
                             'prioritized_replay_alpha': 0.5537428127469798,
                             'prioritized_replay_beta': 0.37455999591887745},
    'train_batch_size': 438, 'target_network_update_freq': 1, 'normalize_actions': False
}
