from dataclasses import dataclass
from typing import Tuple, List

import skimage.filters
import skimage.exposure
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl

from utils import load_maps


@dataclass
class Quantize:
    q: int
    cmap_debug: str = None
    def __call__(self, folder, depth):
        if self.cmap_debug:
            plt.figure(figsize=(10,1)).gca().imshow(np.vstack([np.arange(self.q)]*2), aspect='auto', cmap=mpl.colormaps[self.cmap_debug])
        print(np.max(depth))
        print(np.min(depth))
        return np.digitize(depth, bins=np.arange(self.q))
        return ((depth*self.q/np.max(depth)).astype(np.uint8)*255/self.q).astype(np.uint8)


@dataclass
class Clipping:
    range: Tuple[int]
    def __call__(self, folder, depth):
        return np.clip(depth, *self.range)


class Normalize:
    def __call__(self, folder, depth):
        return (depth - np.min(depth))/np.ptp(depth)

class MorphoMath:
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs
    def __call__(self, folder, depth):
        func = lambda x: getattr(skimage.morphology, self.name)(x, *self.args, **self.kwargs)
        return np.stack(list(map(func, depth)))

class ContrastStretching:
    percentiles: Tuple[int] = (2, 98)
    def __call__(self, folder, depth):
        #in_range = np.percentile(depth, self.percentiles)
        func = lambda x: skimage.exposure.rescale_intensity(x, in_range='image')
        return np.stack(list(map(func, depth)))

class Equalization:
    def __call__(self, folder, depth):
        return skimage.exposure.equalize_hist(depth)

class AdaptiveEqualization:
    clip_limit: float = 0.03
    def __call__(self, folder, depth):
        func = lambda x: skimage.exposure.equalize_adapthist(x, clip_limit=self.clip_limit)
        return np.stack(list(map(func, depth)))


@dataclass
class Opposite:
    def __call__(self, folder, depth):
        return np.max(depth)-depth

@dataclass
class ConfidenceThreshold:
    threshold: int = 100
    def __call__(self, folder, depth):
        confidence = load_maps(folder, "confidence")
        depth[confidence < 100] = 0
        return depth

# Doesn't handle multiple persons correctly
@dataclass
class OtsuThresholding:
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
    def __call__(self, folder, depth):
        threshold = self.threshold(depth, bins=self.bins, range=self.range, normed=True)
        return np.clip(depth, self.range[0], threshold)
