import cv2
import numpy as np
from typing import Optional


class Postprocessor:
    def __init__(self, median_kernel=5, temporal_alpha=0.3):
        self.median_kernel = median_kernel
        self.temporal_alpha = temporal_alpha

    def filter(self, depth):
        """Median-Filter auf Depth-Map."""
        d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        f = cv2.medianBlur(d8, self.median_kernel).astype(np.float32)
        scale = np.max(depth) / max(np.max(f), 1e-6)
        return f * scale

    def temporal_smooth(self, depth, prev):
        if prev is None or prev.shape != depth.shape:
            return depth
        return self.temporal_alpha * depth + (1 - self.temporal_alpha) * prev

    @staticmethod
    def colorize(depth, cmap=cv2.COLORMAP_INFERNO):
        d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.applyColorMap(d8, cmap)

    @staticmethod
    def save_raw(depth, path):
        """
        Speichert die rohe Depth-Map (inverse Disparität) als .npz.
        Das ist das Original-Format aus Depth Anything V2.
        convert_depth_for_viewer.py rechnet dann in metrische Tiefe um.
        """
        if path.endswith('.npz'):
            np.savez_compressed(path, depth=depth)
        elif path.endswith('.npy'):
            np.save(path, depth)
        else:
            np.savez_compressed(path + '.npz', depth=depth)

    @staticmethod
    def save_vis(depth, path):
        cv2.imwrite(path, Postprocessor.colorize(depth))
