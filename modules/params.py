prenet_configs = {
    "model_params": {
        "input_dim": 384,
        "output_dim": 768,
        "rank": 128,
        "prefix_len": 20,
    },
    "load_checkpoint": "saved_models/prenet_prefix_tuning_bookcorpus_multilingual.pth"
}

scm_configs = {
    "model_params": {
        "d_model": 512,
        "embed_dim": 384,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 4 * 512,
        "dropout": 0.1,
        "max_seq_len": 16,
    },
    "load_checkpoint": "saved_models/scm_v01.pth"
}