#!/usr/bin/env python3
"""
Konvertiert monocular Depth-Map .npz in das Format für InteractiveDepthViewer.
Invertiert die Depth (DA: nah=hoch, fern=niedrig → Viewer: nah=niedrig, fern=hoch)
und erzeugt synthetische Disparity + Confidence.

Usage:
    python3 convert_depth_for_viewer.py output/depth_000000.npz /media/data/.../stereo_params_Best.npz
"""
import sys
import numpy as np

def convert(depth_npz_path, calib_npz_path, output_path=None):
    # Load depth (relative values from Depth Anything, 0-8.5 range)
    data = np.load(depth_npz_path)
    depth_raw = data['depth']
    
    # Load calibration
    calib = np.load(calib_npz_path)
    mtx_l = calib['mtxL']
    fx = float(mtx_l[0, 0])
    fy = float(mtx_l[1, 1])
    cx = float(mtx_l[0, 2])
    cy = float(mtx_l[1, 2])
    
    T = calib['T']
    baseline_mm = float(abs(T[0]))  # 54.5 mm
    
    # Invert depth: DA gibt hohe Werte für nahe Objekte
    # Viewer erwartet: niedrige Werte = nah, hohe Werte = fern
    # Also: depth_inverted = max - depth
    valid_mask = depth_raw > 0
    depth_min = depth_raw[valid_mask].min() if valid_mask.any() else 0
    depth_max = depth_raw[valid_mask].max() if valid_mask.any() else 1
    
    # Invertieren
    depth_inv = np.zeros_like(depth_raw)
    depth_inv[valid_mask] = depth_max - depth_raw[valid_mask] + depth_min
    
    # Metrisch skalieren: nutze Baseline + Focal Length
    # depth_mm = (fx * baseline_mm) / disparity
    # Wir erzeugen synthetische Disparity aus der invertierten Depth
    # disparity = (fx * baseline) / depth_mm
    # Aber wir haben keine echte metrische Depth — also skalieren wir
    # die invertierten Werte so, dass sie in einem sinnvollen mm-Bereich liegen
    
    # Skalierung: Map invertierte Depth auf 100mm - 10000mm Bereich
    depth_mm = np.zeros_like(depth_inv)
    if valid_mask.any():
        d_min = depth_inv[valid_mask].min()
        d_max = depth_inv[valid_mask].max()
        if d_max > d_min:
            # Linear skalieren: 100mm (nah) bis 10000mm (fern)
            depth_mm[valid_mask] = 100.0 + (depth_inv[valid_mask] - d_min) / (d_max - d_min) * 9900.0
        else:
            depth_mm[valid_mask] = 1000.0
    
    # Disparity aus metrischer Depth
    disparity = np.zeros_like(depth_mm)
    valid_depth = depth_mm > 0
    disparity[valid_depth] = (fx * baseline_mm) / depth_mm[valid_depth]
    
    # Confidence: basiert auf Depth-Qualität
    # Hohe Disparity = hohe Konfidenz (nahe Objekte)
    # Niedrige Disparity = niedrige Konfidenz (ferne Objekte)
    confidence = np.zeros_like(depth_mm, dtype=np.float32)
    if valid_depth.any():
        d_valid = disparity[valid_depth]
        d_max = d_valid.max()
        if d_max > 0:
            confidence[valid_depth] = np.clip(d_valid / d_max, 0.1, 1.0)
    
    if output_path is None:
        output_path = depth_npz_path.replace('.npz', '_viewer.npz')
    
    np.savez_compressed(output_path,
        depth=depth_mm,
        disparity=disparity,
        confidence=confidence,
    )
    
    print(f"[OK] Gespeichert: {output_path}")
    print(f"  depth: {depth_mm.shape}, range: {depth_mm[valid_depth].min():.1f} - {depth_mm.max():.1f} mm")
    print(f"  disparity: {disparity.shape}, range: {disparity[valid_depth].min():.2f} - {disparity.max():.2f} px")
    print(f"  confidence: {confidence.shape}, range: {confidence[valid_depth].min():.2f} - {confidence.max():.2f}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <depth.npz> <calibration.npz> [output.npz]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
