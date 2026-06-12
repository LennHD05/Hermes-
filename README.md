# Depth Estimation Pipeline - Jetson Orin (JP6)

RAFT-Stereo + TensorRT fuer dichte Depth-Estimation.

## Quick Start
```bash
bash setup_jetson.sh
# Kalibrierung + ONNX-Modell reinlegen
python3 tools/convert_onnx_to_trt.py --onnx models/raft_stereo_small.onnx --engine models/raft_stereo_small_fp16.engine
python3 run_depth.py --calib config/calibration.npz --engine models/raft_stereo_small_fp16.engine --left L.png --right R.png
```

## Hinweise
- Input-Bilder muessen rectifiziert sein
- .npz Translation T ist in mm -> Pipeline rechnet in Meter
- Engine ist plattformspezifisch -> auf Jetson generieren
- Keine Admin-Rechte noetig
