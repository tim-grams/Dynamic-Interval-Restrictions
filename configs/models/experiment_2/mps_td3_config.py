from src.algorithms.exploration.exploration import DynamicIntervalGaussianNoise

mps_td3_config = {
    'twin_q': True, 'policy_delay': 1, 'smooth_target_policy': True, 'target_noise': 0.31912190444276295,
    'target_noise_clip': 0.244654190775938, 'actor_hiddens': [128, 128], 'actor_hidden_activation': 'relu',
    'critic_hiddens': [512, 256], 'critic_hidden_activation': 'relu', 'critic_lr': 0.000646567084486457,
    'actor_lr': 5.288905212010455e-05, 'tau': 0.002878519934755328, 'l2_reg': 0.00032072422792959435,
    'exploration_config': {'type': DynamicIntervalGaussianNoise,
                           'random_timesteps': 208, 'initial_scale': 18.554568064374195,
                           'final_scale': 1.5869383596609679, 'scale_timesteps': 27876,
                           'epsilon_timesteps': 26999,
                           'final_epsilon': 0.005292920751321664,
                           'initial_epsilon': 0.27014994056653024
                           },
    'replay_buffer_config': {'capacity': 45168, 'prioritized_replay_alpha': 0.3016319962732861,
                             'prioritized_replay_beta': 0.44320130539111274}, 'train_batch_size': 101,
    'target_network_update_freq': 1, 'normalize_actions': False
}
