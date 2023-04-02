from ray import tune
from ray.rllib.utils.exploration import GaussianNoise

td3_search_space = {
    'twin_q': True,
    'policy_delay': tune.randint(lower=1, upper=4),
    'smooth_target_policy': True,
    'target_noise': tune.uniform(lower=0.0, upper=0.5),
    'target_noise_clip': tune.uniform(lower=0.1, upper=0.5),
    'actor_hiddens': tune.choice(([128, 128], [256, 128], [256, 256], [512, 256],
                                  [256, 512], [256], [512], [128], [64])),
    'actor_hidden_activation': tune.choice(['relu']),
    'critic_hiddens': tune.choice(([128, 128], [256, 128], [256, 256], [512, 256],
                                  [256, 512], [256], [512], [128], [64])),
    'critic_hidden_activation': tune.choice(['relu']),
    'critic_lr': tune.uniform(lower=5e-08, upper=1e-3),
    'actor_lr': tune.uniform(lower=5e-08, upper=1e-3),
    'tau': tune.uniform(lower=1e-4, upper=0.005),
    'l2_reg': tune.uniform(lower=1e-8, upper=1e-3),
    'exploration_config': {
        'type': GaussianNoise,
        'random_timesteps': tune.randint(lower=100, upper=400),
        'initial_scale': tune.uniform(lower=0.2, upper=30.0),
        'final_scale': tune.uniform(lower=0.01, upper=3.0),
        'scale_timesteps': tune.randint(lower=1000, upper=50000)
    },
    'replay_buffer_config': {
        'capacity': tune.randint(lower=1500, upper=70000),
        'prioritized_replay_alpha': tune.uniform(lower=0.3, upper=0.7),
        'prioritized_replay_beta': tune.uniform(lower=0.2, upper=0.5)
    },
    'train_batch_size': tune.randint(lower=50, upper=512),
    'target_network_update_freq': tune.randint(lower=1, upper=4)
}
