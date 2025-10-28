import torch
import os
import pandas as pd
import pickle
from py3nvml.py3nvml import *
from filelock import FileLock


class XlsxDataWriter:
    """
    Utility class for writing dictionary data into Excel (.xlsx) files.
    Automatically appends new rows and creates folders if needed.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.folder_path = os.path.dirname(file_path)

    def write_dict(self, data_dict: dict):
        """Append a dictionary as a new row into an Excel file."""
        os.makedirs(self.folder_path, exist_ok=True)
        try:
            df = pd.read_excel(self.file_path, index_col=0)
        except FileNotFoundError:
            df = pd.DataFrame()
            df.to_excel(self.file_path, engine="openpyxl")

        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        df.to_excel(self.file_path, engine="openpyxl")


def get_available_device() -> int:
    """
    Automatically select the GPU with the most available memory.
    Returns the GPU index (int), or 'cpu' if no CUDA device is available.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    nvmlInit()
    available_mem = []

    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        available_mem.append(info.free)

    best_gpu = int(torch.argmax(torch.tensor(available_mem)))
    nvmlShutdown()
    return best_gpu