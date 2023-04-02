td3_config_test = {
    'twin_q': False,
    'policy_delay': 1,
    'smooth_target_policy': True,
    'target_noise': 0.1,
    'target_noise_clip': 0.1,
    'actor_hiddens': [500, 300],
    'actor_hidden_activation': 'relu',
    'critic_hiddens': [250, 600],
    'critic_hidden_activation': 'relu',
    'critic_lr': 2e-4,
    'actor_lr': 2e-5,
    'tau': 0.0024,
    'l2_reg': 1.2e-4,
    'exploration_config': {
        'random_timesteps': 3200,
        'ou_theta': 0.15,
        'ou_sigma': 0.13,
        'initial_scale': 0.5,
        'final_scale': 0.046,
        'scale_timesteps': 25000
    },
    'replay_buffer_config': {
        'capacity': 50000,
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta': 0.5
    },
    'train_batch_size': 750,
    'target_network_update_freq': 1,
    'normalize_actions': False
}

td3_config_oilfield = {
    'twin_q': True,
    'policy_delay': 1,
    'smooth_target_policy': False,
    'target_noise': 0.2,
    'target_noise_clip': 0.5,
    'actor_hiddens': [300, 400],
    'actor_hidden_activation': 'relu',
    'critic_hiddens': [300, 500],
    'critic_hidden_activation': 'relu',
    'critic_lr': 5e-4,
    'actor_lr': 8e-5,
    'tau': 0.001,
    'l2_reg': 1e-4,
    'exploration_config': {
        'random_timesteps': 3000,
        'ou_theta': 0.15,
        'ou_sigma': 0.2,
        'initial_scale': 0.8,
        'final_scale': 0.09,
        'scale_timesteps': 20000
    },
    'replay_buffer_config': {
        'capacity': 50000,
        'prioritized_replay_alpha': 0.6,
        'prioritized_replay_beta': 0.5
    },
    'train_batch_size': 512,
    'target_network_update_freq': 2,
    'normalize_actions': False
}
