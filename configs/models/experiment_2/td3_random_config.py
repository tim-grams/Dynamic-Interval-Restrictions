from ray.rllib.utils.exploration import GaussianNoise

td3_random_config = {
    'twin_q': True, 'policy_delay': 1, 'smooth_target_policy': True, 'target_noise': 0.38837567763779524,
    'target_noise_clip': 0.30479475964793473, 'actor_hiddens': [128, 128], 'actor_hidden_activation': 'relu',
    'critic_hiddens': [128], 'critic_hidden_activation': 'relu', 'critic_lr': 0.000888509615304484,
    'actor_lr': 6.694584441518567e-05, 'tau': 0.0019111991912836724, 'l2_reg': 0.0005834090698650431,
    'exploration_config': {'type': GaussianNoise,
                           'stddev': 1.0, 'random_timesteps': 219, 'initial_scale': 27.317866901559068,
                           'final_scale': 2.0013098006235372, 'scale_timesteps': 37168},
    'replay_buffer_config': {'capacity': 43248, 'prioritized_replay_alpha': 0.3663607766186519,
                             'prioritized_replay_beta': 0.41847997435699963}, 'train_batch_size': 478,
    'target_network_update_freq': 2, 'normalize_actions': False}
