import time
import numpy as np
from typing import Tuple
from src.calibrator import Calibrator
from src.preprocessor import Preprocessor
from src.inference import TensorRTInference
from src.postprocessor import Postprocessor

class DepthPipeline:
    def __init__(self, calib_path, engine_path, target_size=(640, 480),
                 use_gpu_preprocess=True, temporal_alpha=0.3):
        print("[Pipeline] Init...")
        self.calibrator = Calibrator(calib_path)
        info = self.calibrator.get_info()
        print(f"[Pipeline] Baseline={info['baseline_m']:.4f}m Focal={info['focal_length_px']:.1f}px")
        self.preprocessor = Preprocessor(target_size, use_gpu_preprocess)
        self.inference = TensorRTInference(engine_path)
        self.postprocessor = Postprocessor(temporal_alpha=temporal_alpha)
        self.prev_depth = None
        self.frame_count = 0

    def run(self, left, right):
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        lt, rt = self.preprocessor.process(left, right)
        t_pre = time.perf_counter() - t1
        t1 = time.perf_counter()
        disparity = self.inference.infer(lt, rt)
        t_inf = time.perf_counter() - t1
        t1 = time.perf_counter()
        depth = self.calibrator.disparity_to_depth(disparity)
        depth = self.postprocessor.filter(depth)
        depth = self.postprocessor.temporal_smooth(depth, self.prev_depth)
        self.prev_depth = depth.copy()
        t_post = time.perf_counter() - t1
        total = time.perf_counter() - t0
        self.frame_count += 1
        return depth, {
            'frame': self.frame_count,
            'total_ms': total * 1000,
            'preprocess_ms': t_pre * 1000,
            'inference_ms': t_inf * 1000,
            'postprocess_ms': t_post * 1000,
            'fps': 1.0 / max(total, 1e-6),
            'depth_mean_m': float(np.mean(depth[depth > 0])),
        }

    def benchmark(self, left, right, n_warmup=10, n_iters=100):
        for _ in range(n_warmup):
            self.run(left, right)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.run(left, right)
            times.append(time.perf_counter() - t0)
        t = np.array(times)
        r = {'mean_ms': float(np.mean(t)*1000), 'median_ms': float(np.median(t)*1000),
             'p95_ms': float(np.percentile(t,95)*1000), 'fps': float(1.0/np.mean(t))}
        print(f"Mean: {r['mean_ms']:.1f}ms ({r['fps']:.1f} FPS)  Median: {r['median_ms']:.1f}ms  P95: {r['p95_ms']:.1f}ms")
        return r
