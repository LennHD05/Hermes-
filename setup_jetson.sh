#!/bin/bash
set -e
echo "=== Depth Estimation Pipeline Setup (Jetson Orin / JP6) ==="
PYTHON=python3
echo "[1/4] Python: $($PYTHON --version)"
if [ ! -d ".venv" ]; then
    echo "[2/4] Erstelle venv..."
    $PYTHON -m venv .venv --system-site-packages
else
    echo "[2/4] venv existiert"
fi
source .venv/bin/activate
echo "[3/4] Installiere Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "[4/4] Pruefe TensorRT + CUDA..."
python3 -c "import tensorrt; print(f'  TensorRT: {tensorrt.__version__}')" 2>/dev/null || echo "  WARN: TensorRT nicht im venv"
python3 -c "import pycuda.driver as cuda; cuda.init(); print(f'  GPU: {cuda.Device(0).name()}')" 2>/dev/null || echo "  WARN: PyCUDA nicht gefunden"
echo ""
echo "=== Setup fertig ==="
echo "Engine generieren:"
echo "  python3 tools/convert_onnx_to_trt.py --onnx models/raft_stereo_small.onnx --engine models/raft_stereo_small_fp16.engine"
echo "Testen:"
echo "  python3 run_depth.py --calib config/calibration.npz --engine models/raft_stereo_small_fp16.engine --left L.png --right R.png"
