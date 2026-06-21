
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Cố định các giá trị ngẫu nhiên để những lần chạy notebook
    cho kết quả gần giống nhau.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Cố định seed cho từng worker của DataLoader.

    worker là tiến trình phụ được PyTorch sử dụng
    để đọc ảnh nhanh hơn.
    """
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)
