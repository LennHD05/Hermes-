import subprocess
import argparse
from pathlib import Path
import shutil
import sys

def convert(onnx_path, engine_path, fp16=True, workspace_mb=2048, trtexec_path=None):
    trtexec = trtexec_path or shutil.which('trtexec')
    if trtexec is None:
        for p in ['/usr/bin/trtexec', '/usr/src/tensorrt/bin/trtexec', '/opt/nvidia/tensorrt/bin/trtexec']:
            if Path(p).exists():
                trtexec = p
                break
    if trtexec is None:
        raise FileNotFoundError("trtexec nicht gefunden. Ist TensorRT installiert?")

    cmd = [
        trtexec,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--memPoolSize=workspace:{workspace_mb}',
    ]
    
    if fp16:
        cmd.append('--fp16')

    # Check if ONNX has dynamic shapes by looking at the model
    # For static models, do NOT pass minShapes/optShapes/maxShapes
    import onnx
    model = onnx.load(onnx_path)
    is_dynamic = False
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:  # Has symbolic dimension like "height", "width"
                is_dynamic = True
                break
    
    if is_dynamic:
        print("[TRT] Dynamic shapes detected — adding profile flags")
        cmd += [
            '--minShapes=left:1x3x256x384,right:1x3x256x384',
            '--optShapes=left:1x3x480x640,right:1x3x480x640',
            '--maxShapes=left:1x3x480x640,right:1x3x480x640',
        ]
    else:
        print("[TRT] Static shapes — no profile flags needed")

    print(f"[TRT] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)

    if r.stdout:
        print(r.stdout)
    if r.stderr:
        print(r.stderr, file=sys.stderr)

    if r.returncode == 0:
        sz = Path(engine_path).stat().st_size / (1024*1024)
        print(f"[TRT] OK -> {engine_path} ({sz:.1f} MB)")
    else:
        raise RuntimeError(f"TRT conversion failed (exit {r.returncode})")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--int8', action='store_true')
    p.add_argument('--workspace', type=int, default=2048)
    p.add_argument('--trtexec', default=None)
    a = p.parse_args()
    
    # Check if onnx is available
    try:
        import onnx
    except ImportError:
        print("[WARN] onnx not installed — cannot detect dynamic shapes, assuming static")
    
    convert(a.onnx, a.engine, fp16=a.fp16 and not a.int8, workspace_mb=a.workspace, trtexec_path=a.trtexec)
