#!/usr/bin/env python3
"""
Download-Hinweis für das RAFT-Stereo ONNX-Modell.

Das Modell ist bereits im Repo unter models/raft_stereo.onnx (64MB).
Es wurde mit PyTorch 2.6 + ONNX opset 16 exportiert.

Falls du es separat brauchst:
  - Direkt aus dem Repo: models/raft_stereo.onnx
  - Oder von GitHub: https://github.com/LennHD05/Hermes-/raw/main/models/raft_stereo.onnx

Für TensorRT Engine auf dem Jetson:
  python3 tools/convert_onnx_to_trt.py --onnx models/raft_stereo.onnx --engine models/raft_stereo.engine --fp16
"""
print(__doc__)
