ppo_config_test = {
    'clip_param': 0.3,
    'vf_clip_param': 10.0,
    'vf_loss_coeff': 1.0,
    'entropy_coeff': 0.0,
    'lr': 5e-6,
    'lambda': 1.0,
    'sgd_minibatch_size': 128,
    'num_sgd_iter': 30,
    'train_batch_size': 2000
}

ppo_config_oilfield = {
    'clip_param': 0.3,
    'vf_clip_param': 10.0,
    'vf_loss_coeff': 1.0,
    'entropy_coeff': 0.0,
    'lr': 5e-5,
    'lambda': 1.0,
    'sgd_minibatch_size': 128,
    'num_sgd_iter': 30,
    'train_batch_size': 4000
}
