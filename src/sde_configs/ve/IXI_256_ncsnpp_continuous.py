from src.sde_configs.ve.AAPM_256_ncsnpp_continuous import (
    get_config as get_default_configs,
)


def get_config():
    config = get_default_configs()
    # data
    data = config.data
    data.image_size = 256
    data.dataset = "IXI"
    data.json = "./dataset_split_hcp_20260112.json"
    data.seq = "T1"
    data.orientation = "AX"  # AX, SAG, or COR

    # training = config.training
    # training.eval_freq = 0  # evaluate at very first step of each epoch
    return config
