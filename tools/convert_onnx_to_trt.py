import subprocess
import argparse
from pathlib import Path

def convert(onnx_path, engine_path, fp16=True, workspace_mb=2048):
    cmd = ['trtexec', f'--onnx={onnx_path}', f'--saveEngine={engine_path}',
           f'--workspace={workspace_mb}', '--fp16',
           '--minShapes=left:1x3x256x384,right:1x3x256x384',
           '--optShapes=left:1x3x480x640,right:1x3x480x640',
           '--maxShapes=left:1x3x480x640,right:1x3x480x640']
    print(f"[TRT] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"[TRT] OK -> {engine_path} ({Path(engine_path).stat().st_size/1024/1024:.1f} MB)")
    else:
        print(f"[TRT] FAIL: {r.stderr}")
        raise RuntimeError("TRT failed")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--workspace', type=int, default=2048)
    a = p.parse_args()
    convert(a.onnx, a.engine, workspace_mb=a.workspace)
