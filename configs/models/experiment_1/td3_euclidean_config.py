from ray.rllib.utils.exploration import GaussianNoise

td3_euclidean_config = {
    'twin_q': True,
    'policy_delay': 1,
    'smooth_target_policy': True,
    'target_noise': 0.0903452420229357,
    'target_noise_clip': 0.3203861911573977,
    'actor_hiddens': [256, 512],
    'actor_hidden_activation': 'relu',
    'critic_hiddens': [128, 128],
    'critic_hidden_activation': 'relu',
    'critic_lr': 0.00043855568727332113,
    'actor_lr': 3.5423425055416004e-06,
    'tau': 0.0007307248999778697,
    'l2_reg': 0.0004384737848914736,
    'exploration_config': {
        'type': GaussianNoise,
        'stddev': 1.0, 'random_timesteps': 350, 'initial_scale': 25.246187106806012,
        'final_scale': 1.3001427486898725, 'scale_timesteps': 36192},
    'replay_buffer_config': {
        'capacity': 9337,
        'prioritized_replay_alpha': 0.31006786814753234,
        'prioritized_replay_beta': 0.45345060123898884},
    'train_batch_size': 473,
    'target_network_update_freq': 3,
    'normalize_actions': False
}
