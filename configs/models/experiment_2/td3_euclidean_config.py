from ray.rllib.utils.exploration import GaussianNoise

td3_euclidean_config = {
    'twin_q': True,
    'policy_delay': 1,
    'smooth_target_policy': True,
    'target_noise': 0.3991591439908108,
    'target_noise_clip': 0.2530340633923967,
    'actor_hiddens': [512, 256],
    'actor_hidden_activation': 'relu',
    'critic_hiddens': [512],
    'critic_hidden_activation': 'relu',
    'critic_lr': 9.854074806889184e-05,
    'actor_lr': 2.5875004729359725e-06,
    'tau': 0.0015242731596507842,
    'l2_reg': 0.0008312172611146694,
    'exploration_config': {
        'type': GaussianNoise,
        'stddev': 1.0, 'random_timesteps': 385, 'initial_scale': 15.89006189835185,
        'final_scale': 1.6676665068577785, 'scale_timesteps': 7248},
    'replay_buffer_config': {
        'capacity': 29410,
        'prioritized_replay_alpha': 0.6047139966691818,
        'prioritized_replay_beta': 0.24715294621291123},
    'train_batch_size': 318,
    'target_network_update_freq': 1,
    'normalize_actions': False
}
