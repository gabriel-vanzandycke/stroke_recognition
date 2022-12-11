from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from utils import load_maps




@dataclass
class StrokeDetector:
    operations: List[Callable]
    display_progress: bool = True
    debug: bool = True
    def __post_init__(self):
        self.display = Display()
    def __call__(self, folder):
        data = load_maps(folder, "depth")
        for operation in tqdm(self.operations, disable=not self.display_progress):
            data = operation(folder, data)
            if self.debug:
                self.display(data, name="after " + operation.__class__.__name__)
        return data



@dataclass
class Display:
    start: int = 100
    n: int = 4
    figsize: int = 2
    bins: int = 100
    def __call__(self, depth, name=None):
        figsize = (self.figsize*(self.n+3), self.figsize)
        fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[1, self.n])

        indices = np.linspace(100, len(depth)-1, self.n).astype(np.int32)
        axes[1].imshow(np.hstack([depth[i] for i in indices]))
        axes[1].set_xticks(np.arange(self.n)*depth.shape[-1])
        axes[1].set_xticklabels(indices)
        axes[1].set_yticks([])

        histogram, bins = np.histogram(depth, bins=self.bins)
        histogram[-1] = histogram[0] = 0
        axes[0].plot((bins[1:]+bins[:-1])/2, histogram)
        axes[0].set_yticks([])

        plt.tight_layout(pad=0)#, w_pad=0.5, h_pad=1.0)
        if name:
            axes[0].set_ylabel(name)
        plt.show()
        return depth
