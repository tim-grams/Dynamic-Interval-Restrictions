from ray.rllib.utils.exploration import GaussianNoise

td3_masked_config = {
    'twin_q': True,
    'policy_delay': 1,
    'smooth_target_policy': True,
    'target_noise':  0.08217987949477068,
    'target_noise_clip': 0.3573921885030278,
    'actor_hiddens': [256],
    'actor_hidden_activation': 'relu',
    'critic_hiddens': [256, 512],
    'critic_hidden_activation': 'relu',
    'critic_lr': 0.0002153316529926007,
    'actor_lr': 3.310914836311123e-05,
    'tau': 0.0031909192085114455,
    'l2_reg': 0.000743340446634739,
    'exploration_config': {'type': GaussianNoise,
                           'stddev': 1.0, 'random_timesteps': 293, 'initial_scale': 5.467054285597614,
                           'final_scale': 2.397651870154254, 'scale_timesteps': 17759},
    'replay_buffer_config': {
        'capacity': 50433,
        'prioritized_replay_alpha': 0.4642422889773992,
        'prioritized_replay_beta':  0.4237843845705956
    },
    'train_batch_size': 384,
    'target_network_update_freq': 2,
    'normalize_actions': False
}
