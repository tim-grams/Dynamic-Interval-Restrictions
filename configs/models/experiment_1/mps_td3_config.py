from src.algorithms.exploration.exploration import DynamicIntervalGaussianNoise

mps_td3_config = {
    'twin_q': True, 'policy_delay': 2, 'smooth_target_policy': True, 'target_noise': 0.04868657674975847,
    'target_noise_clip': 0.21023366145810846, 'actor_hiddens': [256, 128], 'actor_hidden_activation': 'relu',
    'critic_hiddens': [256], 'critic_hidden_activation': 'relu', 'critic_lr': 0.0003503883951955494,
    'actor_lr': 2.6951088952618366e-05, 'tau': 0.0008977489483604194, 'l2_reg': 2.4386862373498397e-05,
    'exploration_config': {'type': DynamicIntervalGaussianNoise,
                           'random_timesteps': 363, 'initial_scale': 8.745416833102059,
                           'final_scale': 0.11503916474901722, 'scale_timesteps': 15792,
                           'epsilon_timesteps': 15000,
                           'final_epsilon': 0.01,
                           'initial_epsilon': 0.2
                           },
    'replay_buffer_config': {'capacity': 32410, 'prioritized_replay_alpha': 0.6126233505604154,
                             'prioritized_replay_beta': 0.3217190947262794}, 'train_batch_size': 470,
    'target_network_update_freq': 2, 'normalize_actions': False
}
