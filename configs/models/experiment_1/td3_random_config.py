from ray.rllib.utils.exploration import GaussianNoise

td3_random_config = {
    'twin_q': True, 'policy_delay': 3, 'smooth_target_policy': True, 'target_noise': 0.2704626647456578,
    'target_noise_clip': 0.2789702169236477, 'actor_hiddens': [256, 256], 'actor_hidden_activation': 'relu',
    'critic_hiddens': [256], 'critic_hidden_activation': 'relu', 'critic_lr': 0.0006455746461847786,
    'actor_lr': 8.182318443839905e-06, 'tau': 0.0009103921250191557, 'l2_reg': 1.8080659942180835e-05,
    'exploration_config': {'type': GaussianNoise,
                           'stddev': 1.0, 'random_timesteps': 305, 'initial_scale': 13.291647692099854,
                           'final_scale': 0.8190773366267439, 'scale_timesteps': 35129},
    'replay_buffer_config': {'capacity': 56739, 'prioritized_replay_alpha': 0.4917836118250767,
                             'prioritized_replay_beta': 0.30757371826366015}, 'train_batch_size': 499,
    'target_network_update_freq': 1, 'normalize_actions': False}
