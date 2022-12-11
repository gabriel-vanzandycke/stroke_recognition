from dataclasses import dataclass
from typing import Tuple, List
import contextlib

import skimage.filters
import skimage.exposure
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl

from utils import load_maps, Display

@dataclass
class Operation:
    debug = False
    def __post_init__(self):
        self.display = Display()
    def __call__(self, folder, data):
        with self.display if self.debug else contextlib.nullcontext() as cm:
            data = self.process(folder, data)
            if self.debug:
                cm(data, "after " + self.__class__.__name__)
            return data
    def display_obsolete(self, array):
        n = 4
        fig, axes = plt.subplots(1, 2, figsize=(2*(n+3), 2), width_ratios=[1, n])

        indices = np.linspace(100, len(array)-1, n).astype(np.int32)
        axes[1].imshow(np.hstack([array[i] for i in indices]))
        axes[1].set_xticks(np.arange(n)*array.shape[-1])
        axes[1].set_xticklabels(indices)
        axes[1].set_yticks([])

        histogram, bins = np.histogram(array, bins=100)
        histogram[-1] = histogram[0] = 0
        axes[0].plot((bins[1:]+bins[:-1])/2, histogram)
        axes[0].set_yticks([])

        plt.tight_layout(pad=0)
        axes[0].set_ylabel("after " + self.__class__.__name__)
        plt.show()

@dataclass
class Quantize(Operation):
    q: int
    cmap_debug: str = None
    debug: bool = False
    def process(self, folder, depth):
        if self.cmap_debug:
            plt.figure(figsize=(10,1)).gca().imshow(np.vstack([np.arange(self.q)]*2), aspect='auto', cmap=mpl.colormaps[self.cmap_debug])
        print(np.max(depth))
        print(np.min(depth))
        return np.digitize(depth, bins=np.arange(self.q))
        return ((depth*self.q/np.max(depth)).astype(np.uint8)*255/self.q).astype(np.uint8)


@dataclass
class Clipping(Operation):
    range: Tuple[int]
    debug: bool = False
    def process(self, folder, depth):
        return np.clip(depth, *self.range)

@dataclass
class Normalize(Operation):
    debug: bool = False
    def process(self, folder, depth):
        return (depth - np.min(depth))/np.ptp(depth)

@dataclass
class MorphoMath(Operation):
    name: str
    kwargs: dict
    debug: bool = False
    def process(self, folder, depth):
        func = lambda x: getattr(skimage.morphology, self.name)(x, **self.kwargs)
        return np.stack(list(map(func, depth)))

@dataclass
class ContrastStretching(Operation):
    percentiles: Tuple[int] = (2, 98)
    debug: bool = False
    def process(self, folder, depth):
        #in_range = np.percentile(depth, self.percentiles)
        func = lambda x: skimage.exposure.rescale_intensity(x, in_range='image')
        return np.stack(list(map(func, depth)))

@dataclass
class Equalization(Operation):
    debug: bool = False
    def process(self, folder, depth):
        return skimage.exposure.equalize_hist(depth)

@dataclass
class AdaptiveEqualization(Operation):
    clip_limit: float = 0.03
    debug: bool = False
    def process(self, folder, depth):
        func = lambda x: skimage.exposure.equalize_adapthist(x, clip_limit=self.clip_limit)
        return np.stack(list(map(func, depth)))

@dataclass
class Opposite(Operation):
    debug: bool = False
    def process(self, folder, depth):
        return np.max(depth)-depth

@dataclass
class ConfidenceThreshold(Operation):
    threshold: int = 100
    debug: bool = False
    def process(self, folder, depth):
        confidence = load_maps(folder, "confidence")
        depth[confidence < 100] = 0
        return depth

# Doesn't handle multiple persons correctly
@dataclass
class OtsuThresholding(Operation):
    range: Tuple[int] = (100, 1500)
    bins: int = 100
    debug: bool = False
    def threshold(self, array, **kwargs):
        histogram, bins = np.histogram(array, **kwargs)
        threshold = skimage.filters.threshold_otsu(hist=histogram)
        threshold = bins[threshold]
        if self.debug:
            plt.plot((bins[1:]+bins[:-1])/2, histogram)
            plt.plot([threshold, threshold], [0, np.max(histogram)])
            plt.show()
        return threshold
    def process(self, folder, depth):
        threshold = self.threshold(depth, bins=self.bins, range=self.range, normed=True)
        return np.clip(depth, self.range[0], threshold)

