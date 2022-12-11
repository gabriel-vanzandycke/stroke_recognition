from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from utils import load_maps




@dataclass
class StrokeDetector:
    operations: List[Callable]
    display_progress: bool = False
    def __call__(self, folder):
        data = load_maps(folder, "depth")
        for operation in tqdm(self.operations, disable=not self.display_progress):
            data = operation(folder, data)
        return data

