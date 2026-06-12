import cv2
import argparse
from src.pipeline import DepthPipeline

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--size', default='640x480')
    p.add_argument('--iters', type=int, default=100)
    a = p.parse_args()
    w, h = map(int, a.size.split('x'))
    left, right = cv2.imread(a.left), cv2.imread(a.right)
    if left is None or right is None:
        print("Bilder nicht laden"); exit(1)
    pipe = DepthPipeline(a.calib, a.engine, target_size=(w, h))
    pipe.benchmark(left, right, n_iters=a.iters)
