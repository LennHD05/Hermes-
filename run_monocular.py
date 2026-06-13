#!/usr/bin/env python3
"""
Monocular Depth Estimation mit Depth Anything V2.
Nur ein Bild (links) als Input — kein Stereo-Paar nötig.

Usage:
    python3 run_monocular.py --engine models/depth_anything.engine --left test_left.png
    python3 run_monocular.py --engine models/depth_anything.engine --left 0  # Kamera
    python3 run_monocular.py --engine models/depth_anything.engine --left test_left.png --benchmark
"""
import cv2, argparse, sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.inference import TensorRTInference
from src.postprocessor import Postprocessor

def main():
    p = argparse.ArgumentParser(description='Monocular Depth Estimation')
    p.add_argument('--engine', required=True, help='TensorRT Engine Pfad')
    p.add_argument('--left', required=True, help='Linkes Bild oder Kamera-Index')
    p.add_argument('--size', default='640x480', help='Inference-Größe WxH')
    p.add_argument('--output', default='output', help='Output-Verzeichnis')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--save', action='store_true')
    p.add_argument('--trtexec', default=None, help='Pfad zu trtexec (für Engine-Build)')
    a = p.parse_args()

    w, h = map(int, a.size.split('x'))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Init] Loading engine: {a.engine}")
    inference = TensorRTInference(a.engine)
    post = Postprocessor()

    # Preprocessor (monocular — nur ein Bild)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(img):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return np.ascontiguousarray(img)[None, ...]  # 1x3xHxW

    def run_frame(img):
        t0 = time.perf_counter()
        tensor = preprocess(img)
        t1 = time.perf_counter()
        # Depth Anything: nur ein Input-Tensor
        depth = inference.infer_single(tensor)
        t2 = time.perf_counter()
        # Resize depth auf Input-Größe (DA internes Padding)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = post.filter(depth)
        t3 = time.perf_counter()
        meta = {
            'total_ms': (t3 - t0) * 1000,
            'preprocess_ms': (t1 - t0) * 1000,
            'inference_ms': (t2 - t1) * 1000,
            'postprocess_ms': (t3 - t2) * 1000,
            'fps': 1.0 / max(t3 - t0, 1e-6),
        }
        return depth, meta

    if a.benchmark:
        img = cv2.imread(a.left)
        if img is None:
            print("Bild nicht laden"); sys.exit(1)
        # Warmup
        for _ in range(10):
            run_frame(img)
        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            run_frame(img)
            times.append(time.perf_counter() - t0)
        t = np.array(times)
        print(f"\n{'='*50}")
        print(f"  Benchmark (100 iterations)")
        print(f"{'='*50}")
        print(f"  Mean:   {np.mean(t)*1000:.1f}ms ({1.0/np.mean(t):.1f} FPS)")
        print(f"  Median: {np.median(t)*1000:.1f}ms")
        print(f"  P95:    {np.percentile(t,95)*1000:.1f}ms")
        print(f"{'='*50}\n")
        return

    # Kamera oder Datei?
    if a.left.isdigit():
        cap = cv2.VideoCapture(int(a.left))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        use_cam = True
    else:
        use_cam = False

    # Auto-detect headless mode
    import os
    has_display = bool(os.environ.get('DISPLAY', ''))
    use_display = not a.no_display and has_display
    
    if not has_display and not a.no_display:
        print("[INFO] No display detected — running headless (use --save to save images)")

    idx = 0
    try:
        while True:
            if use_cam:
                ret, img = cap.read()
                if not ret: break
            else:
                img = cv2.imread(a.left)
                if img is None: break

            depth, meta = run_frame(img)

            if a.save:
                Postprocessor.save_raw(depth, str(out/f'depth_{idx:06d}.npy'))
                Postprocessor.save_vis(depth, str(out/f'depth_{idx:06d}.png'))
            elif not use_display:
                # Headless: save first frame as preview
                if idx == 0:
                    Postprocessor.save_vis(depth, str(out/f'depth_preview.png'))
                    print(f"[INFO] Preview saved: {out}/depth_preview.png")

            if use_display:
                vis = Postprocessor.colorize(depth)
                txt = f"FPS:{meta['fps']:.1f} Inf:{meta['inference_ms']:.0f}ms"
                cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow('Depth (Monocular)', vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): break
            else:
                if idx % 30 == 0:
                    print(f"Frame {idx}: {meta['fps']:.1f} FPS, {meta['inference_ms']:.0f}ms")

            idx += 1
            if not use_cam: break
    finally:
        if use_cam: cap.release()
        if use_display: cv2.destroyAllWindows()
        print(f"{idx} Frames verarbeitet")

if __name__ == '__main__':
    main()
