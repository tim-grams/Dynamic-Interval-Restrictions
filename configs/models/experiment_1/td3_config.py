from ray.rllib.utils.exploration import GaussianNoise

td3_config = {
    'twin_q': True, 'policy_delay': 2, 'smooth_target_policy': True, 'target_noise': 0.3796816148541915,
    'target_noise_clip': 0.2778307415781265, 'actor_hiddens': [256], 'actor_hidden_activation': 'relu',
    'critic_hiddens': [512], 'critic_hidden_activation': 'relu', 'critic_lr': 0.0008493872830893942,
    'actor_lr': 6.303169615060298e-05, 'tau': 0.0021305295299441024, 'l2_reg': 5.989191900046081e-05,
    'exploration_config': {'type': GaussianNoise,
                           'stddev': 1.0, 'random_timesteps': 321, 'initial_scale': 16.25446048481061,
                           'final_scale': 0.5591310745907552, 'scale_timesteps': 37184},
    'replay_buffer_config': {'capacity': 21836,
                             'prioritized_replay_alpha': 0.6994764414963202,
                             'prioritized_replay_beta': 0.4737605561976087},
    'train_batch_size': 420, 'target_network_update_freq': 2, 'normalize_actions': False
}
