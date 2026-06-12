#!/usr/bin/env python3
import cv2, argparse, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import DepthPipeline
from src.postprocessor import Postprocessor

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--size', default='640x480')
    p.add_argument('--output', default='output')
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--save', action='store_true')
    a = p.parse_args()
    w, h = map(int, a.size.split('x'))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    pipe = DepthPipeline(a.calib, a.engine, target_size=(w, h))
    if a.benchmark:
        left, right = cv2.imread(a.left), cv2.imread(a.right)
        if left is None or right is None: print("Bilder nicht laden"); sys.exit(1)
        pipe.benchmark(left, right); return
    def open_src(s):
        if s.isdigit():
            c = cv2.VideoCapture(int(s)); c.set(cv2.CAP_PROP_FRAME_WIDTH, w); c.set(cv2.CAP_PROP_FRAME_HEIGHT, h); return c
        return None
    cl, cr = open_src(a.left), open_src(a.right)
    use_cam = cl is not None; idx = 0
    try:
        while True:
            if use_cam:
                rl, left = cl.read(); rr, right = cr.read()
                if not rl or not rr: break
            else:
                left, right = cv2.imread(a.left), cv2.imread(a.right)
                if left is None or right is None: break
            depth, meta = pipe.run(left, right)
            if a.save:
                Postprocessor.save_raw(depth, str(out/f'depth_{idx:06d}.npy'))
                Postprocessor.save_vis(depth, str(out/f'depth_{idx:06d}.png'))
            if not a.no_display:
                vis = Postprocessor.colorize(depth)
                txt = f"FPS:{meta['fps']:.1f} {meta['inference_ms']:.0f}ms D:{meta['depth_mean_m']:.2f}m"
                cv2.putText(vis, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow('Depth', vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): break
            else:
                if idx % 30 == 0: print(f"Frame {idx}: {meta['fps']:.1f} FPS")
            idx += 1
            if not use_cam: break
    finally:
        if use_cam: cl.release(); cr.release()
        if not a.no_display: cv2.destroyAllWindows()
        print(f"{idx} Frames")

if __name__ == '__main__':
    main()
