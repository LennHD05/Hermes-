#!/usr/bin/env python3
"""
Konvertiert monocular Depth-Map .npz → InteractiveDistanceViewer Format.

Input .npz: disparity (inverse Tiefe, DA: nah=hoch, fern=niedrig)
             confidence (optional)
             calib (optional), sonst geladen aus stereo_params_Best.npz

Formel: depth_mm = (fx * baseline_mm) / disparity

Output .npz: depth (mm), disparity (px), confidence (0-1)

Usage:
    python3 convert_depth_for_viewer.py output/depth_000000.npz [stereo_params_Best.npz]
"""
import sys
import os
import numpy as np

CALIB_DEFAULT = "/media/data/LaborRobotikSS26/robotic-ss-2026/Sensorics/Stereo-Camera/stereo_params_Best.npz"


def convert(input_npz, calib_npz=None, output_path=None):
    # 1. Load input
    data = np.load(input_npz)

    # Unterstützt beide Formate:
    #   A) run_monocular.py Output: disparity + confidence
    #   B) Altes Format: depth (invertiert, relativ)
    if "disparity" in data:
        disparity_raw = data["disparity"].astype(np.float64)
        confidence = data["confidence"].astype(np.float32) if "confidence" in data else None
        print("[INFO] Input: disparity + confidence (neues Format)")
    elif "depth" in data:
        # Altes Format: invertierte relative Depth → zurück zu Disparity
        depth_inv = data["depth"].astype(np.float64)
        valid = depth_inv > 0
        d_min = depth_inv[valid].min() if valid.any() else 0
        d_max = depth_inv[valid].max() if valid.any() else 1
        # Rück-Invertierung: disparity = d_max - depth_inv + d_min
        disparity_raw = np.zeros_like(depth_inv)
        disparity_raw[valid] = d_max - depth_inv[valid] + d_min
        confidence = None
        print("[INFO] Input: depth (invertiert) → Disparity rückgerechnet")
    else:
        print(f"[ERROR] Weder 'disparity' noch 'depth' in {input_npz}")
        print(f"  Keys: {list(data.keys())}")
        sys.exit(1)

    # 2. Load calibration
    if calib_npz is None:
        calib_npz = CALIB_DEFAULT

    if not os.path.exists(calib_npz):
        print(f"[WARN] Calib nicht gefunden: {calib_npz}")
        print("[WARN] Nutze Fallback: fx=1515, baseline=54.5mm")
        fx = 1515.0
        baseline_mm = 54.5
    else:
        calib = np.load(calib_npz)
        mtx_l = calib["mtxL"]
        fx = float(mtx_l[0, 0])
        fy = float(mtx_l[1, 1])
        cx = float(mtx_l[0, 2])
        cy = float(mtx_l[1, 2])

        if "T" in calib:
            baseline_mm = float(abs(calib["T"][0]))
        else:
            baseline_mm = 54.5
            print(f"[WARN] Keine Baseline in Calib, nutze Fallback: {baseline_mm}mm")

    print(f"[CALIB] fx={fx:.1f}, baseline={baseline_mm:.1f}mm")

    # 3. Disparity → Depth (metrisch)
    #    depth_mm = (fx * baseline_mm) / disparity
    valid_disp = disparity_raw > 0
    depth_mm = np.zeros_like(disparity_raw)
    depth_mm[valid_disp] = (fx * baseline_mm) / disparity_raw[valid_disp]

    # Clamp: 10mm - 20000mm (0.01m - 20m)
    depth_mm = np.clip(depth_mm, 10.0, 20000.0)
    depth_mm[~valid_disp] = 0.0

    # 4. Confidence (falls nicht geladen)
    if confidence is None:
        confidence = np.ones_like(depth_mm, dtype=np.float32)
        confidence[~valid_disp] = 0.0
        # Heuristik: sehr nahe oder sehr weit → niedrigere Konfidenz
        confidence[depth_mm < 100] = 0.5
        confidence[depth_mm > 10000] = 0.5

    # 5. Disparity für Viewer: echte Disparity in Pixeln
    #    disparity_viewer = disparity_raw (die DA-Werte sind inverse Disparität,
    #    aber der Viewer nutzt sie nur für die Colorbar — die Metrik kommt aus depth_mm)
    disparity_viewer = disparity_raw.astype(np.float32)

    # 6. Save
    if output_path is None:
        output_path = input_npz.replace(".npz", "_viewer.npz")

    np.savez_compressed(
        output_path,
        depth=depth_mm.astype(np.float32),
        disparity=disparity_viewer,
        confidence=confidence,
    )

    valid = depth_mm > 0
    print(f"[OK] Gespeichert: {output_path}")
    print(f"  depth:     {depth_mm.shape}, range: {depth_mm[valid].min():.1f} - {depth_mm[valid].max():.1f} mm")
    print(f"  disparity: {disparity_viewer.shape}, range: {disparity_viewer[valid_disp].min():.2f} - {disparity_viewer[valid_disp].max():.2f}")
    print(f"  confidence: {confidence.shape}, range: {confidence[valid].min():.2f} - {confidence[valid].max():.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <depth.npz> [calibration.npz] [output.npz]")
        sys.exit(1)
    input_path = sys.argv[1]
    calib_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".npz") else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    convert(input_path, calib_path, output_path)
