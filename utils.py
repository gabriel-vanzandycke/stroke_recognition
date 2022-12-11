import configparser
from dataclasses import dataclass
import os
from typing import Tuple, List, Callable

import cv2
import glob
import imageio
import numpy as np
from tqdm.auto import tqdm

from calib3d import Calib, Point3D

from matplotlib import pyplot as plt


def colorify_heatmap(heatmap, colormap="jet"):
    return (plt.get_cmap(colormap)(heatmap)[...,0:3]*255).astype(np.uint8)


def build_calib(filename):
    parser = configparser.ConfigParser()
    filename = "dataset/ds325/camera_parameters.txt"
    parser.read_string("[section]\n" + open(filename).read())
    data = {k: float(v) for k,v in parser.items('section')}
    K = np.array([[data['focal_x'],        0       , data['center_x']],
                  [       0       , data['focal_y'], data['center_y']],
                  [       0       ,        0       ,        1        ]])
    kc = data['k1'], data['k2'], data['p1'], data['p2'], data['k3']
    width, height = int(data['width']), int(data['height'])
    return Calib(K=K, kc=kc, R=np.eye(3), T=Point3D(0,0,0), width=width, height=height)

def make_video(filename, folder, fun):
    with VideoMaker(filename) as vm:
        length = len(glob.glob(f"{folder}/*"))//2
        calib = build_calib(os.path.join(os.path.dirname(folder), 'camera_parameters.txt'))
        for i in tqdm(range(length), leave=False):
            filename = f"{folder}/{i:06d}_{{}}.{'tiff' if '325' in folder else 'tif'}"
            depth, confidence = list(map(lambda mod: imageio.imread(filename.format(mod)), ['depth', 'confidence']))
            image = fun(depth, confidence, calib)
            vm(image)


def load_maps(folder, name):
    return np.stack([imageio.imread(f) for f in sorted(glob.glob(os.path.join(folder, f"*_{name}.tif*")))])

def load_calib(folder):
    return build_calib(os.path.join(os.path.dirname(folder), 'camera_parameters.txt'))


@dataclass
class StrokeDetector:
    operations: List[Callable]
    display_progress: bool = True
    def __call__(self, folder):
        data = load_maps(folder, "depth")
        for operation in self.operations:
            data = operation(folder, data)
        return data




class VideoMaker():
    format_map = {
        ".mp4": 'mp4v',
        ".avi": 'XVID',
        ".mpeg4": 'H264'
    }
    writer = None
    def __init__(self, filename: str, framerate: int = 30):
        self.filename = filename
        self.framerate = framerate
        self.fourcc = cv2.VideoWriter_fourcc(*self.format_map[os.path.splitext(filename)[1]])
    def __enter__(self):
        return self
    def __call__(self, image):
        if self.writer is None:
            shape = (image.shape[1], image.shape[0])
            self.writer = cv2.VideoWriter(filename=self.filename, fourcc=self.fourcc, fps=self.framerate, frameSize=shape, apiPreference=cv2.CAP_FFMPEG)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.writer.write(image)
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()
            self.writer = None
            print("{} successfully written".format(self.filename))
    def __del__(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            print("{} successfully written".format(self.filename))
