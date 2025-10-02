import logging
import torch.distributed as dist
import os
def setup_logger(log_file='scan.log'):
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

def setup_distributed_inference_logging(model_key='_qwen2_4b'):
    rank = dist.get_rank()
    logger = logging.getLogger(f"distributed_rank_{rank}")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(f"./trace/inference{model_key}", exist_ok = True)
    log_file = f'./trace/inference{model_key}/rank_{rank}.log'
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def setup_distributed_train_logging():
    rank = dist.get_rank()
    logger = logging.getLogger(f"distributed_rank_{rank}")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs("./trace/train", exist_ok = True)
    log_file = f'./trace/train/rank_{rank}.log'
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger