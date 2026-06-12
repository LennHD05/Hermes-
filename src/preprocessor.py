import cv2
import numpy as np
from typing import Tuple

class Preprocessor:
    def __init__(self, target_size=(640, 480), use_gpu=True):
        self.target_size = target_size
        self.use_gpu = use_gpu and self._check_cuda()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"[Preprocessor] {'GPU' if self.use_gpu else 'CPU'}")

    @staticmethod
    def _check_cuda():
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def process(self, left, right):
        if self.use_gpu:
            return self._process_gpu(left, right)
        return self._process_cpu(left, right)

    def _process_cpu(self, left, right):
        def prep(img):
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        return prep(left)[None, ...], prep(right)[None, ...]

    def _process_gpu(self, left, right):
        def prep(img):
            g = cv2.cuda_GpuMat()
            g.upload(img)
            g = cv2.cuda.resize(g, self.target_size, interpolation=cv2.INTER_LINEAR)
            r = g.download()
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            r = (r - self.mean) / self.std
            return np.ascontiguousarray(np.transpose(r, (2, 0, 1)))
        return prep(left)[None, ...], prep(right)[None, ...]
