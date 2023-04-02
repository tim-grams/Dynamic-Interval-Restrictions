from ray.rllib.utils.exploration import GaussianNoise

td3_masked_config = {
    'twin_q': True,
    'policy_delay': 1,
    'smooth_target_policy': True,
    'target_noise':  0.20960719529135902,
    'target_noise_clip': 0.2637370855724754,
    'actor_hiddens': [256, 512],
    'actor_hidden_activation': 'relu',
    'critic_hiddens': [512],
    'critic_hidden_activation': 'relu',
    'critic_lr': 0.0006811450226826312,
    'actor_lr': 1.031851452315129e-05,
    'tau': 0.002692941488459357,
    'l2_reg': 0.00026364978061225433,
    'exploration_config': {'type': GaussianNoise,
                           'stddev': 1.0, 'random_timesteps': 257, 'initial_scale': 17.258547900039574,
                           'final_scale': 1.7030821729902708, 'scale_timesteps': 16524},
    'replay_buffer_config': {
        'capacity': 32792,
        'prioritized_replay_alpha': 0.3430475321992233,
        'prioritized_replay_beta':  0.3717350195623043
    },
    'train_batch_size': 496,
    'target_network_update_freq': 2,
    'normalize_actions': False
}
