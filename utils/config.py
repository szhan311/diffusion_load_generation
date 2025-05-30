config_dataset = {
    'day_len':365,
    'train_ratio':0.8,
    'cond_dim':79,  # [4, 29, 41, 48] 4+25+12+7=48
    'num_class': 30
}

config_dataset_london = {
    'train_ratio':0.8,
    'cond_dim':81,  # [4, 29, 41, 48] 4+25+12+7=48
    'num_class': 30
}

config_ddpm = {
    'beta_scheduler': "linear", # ["linear", "quad", "cosine", "exponential", "sigmoid"]
    'ema_decay': 0.9,
    'beta_start': 1e-6,
    'beta_end': 2e-2,
    'loss_type': 'l2',
    'n_steps': 500
}

config_nn = {
    'input_dim': 96,
    'hidden_dim': 800,
    'nhead': 8,
}

config_nn_london = {
    'input_dim': 48,
    'hidden_dim': 1000,
    'nhead': 4,
}