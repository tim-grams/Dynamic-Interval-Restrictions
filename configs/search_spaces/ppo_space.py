from ray import tune

ppo_search_space = {
    'clip_param': tune.uniform(lower=0.1, upper=0.4),
    'vf_clip_param': tune.uniform(lower=5.0, upper=15.0),
    'vf_loss_coeff': tune.uniform(lower=0.3, upper=2.0),
    'entropy_coeff': tune.uniform(lower=0.0, upper=0.01),
    'lr': tune.uniform(lower=5e-8, upper=0.005),
    'lambda': tune.uniform(lower=0.8, upper=1.0),
    'sgd_minibatch_size': tune.randint(lower=4, upper=200),
    'num_sgd_iter': tune.randint(lower=3, upper=60),
    'train_batch_size': tune.randint(lower=200, upper=6000),
    'model': {
        'fcnet_hiddens': tune.choice(([128, 128], [256, 128], [256, 256], [512, 256],
                                      [256, 512], [256], [512], [128], [64]))
    }
}
