#!/usr/bin/env python3
"""
Konvertiert Depth-Map .npz (nur 'depth' Key) in das Format
das InteractiveDistanceViewer erwartet: depth + disparity + confidence

Usage:
    python3 convert_depth_for_viewer.py output/depth_000000.npz /media/data/.../stereo_params_Best.npz
"""
import sys
import numpy as np

def convert(depth_npz_path, calib_npz_path, output_path=None):
    # Load depth
    data = np.load(depth_npz_path)
    depth = data['depth']  # in Metern (von Depth Anything)
    
    # Load calibration for baseline + focal length
    calib = np.load(calib_npz_path)
    mtx_l = calib['mtxL']
    fx = float(mtx_l[0, 0])
    
    # Baseline aus Calib — falls T vorhanden
    if 'T' in calib:
        T = calib['T']
        baseline = float(abs(T[0]))  # in mm (aus Kalibrierung)
    else:
        # Fallback: typische Baseline für Stereo
        baseline = 54.0  # mm — anpassen an deine Kamera!
        print(f"[WARN] Keine Baseline in Calib, nutze Fallback: {baseline}mm")
    
    # Depth von Metern -> mm
    depth_mm = depth * 1000.0
    
    # Disparity berechnen: disparity = (fx * baseline) / depth
    # depth in mm, baseline in mm, fx in Pixel -> disparity in Pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        disparity = (fx * baseline) / depth_mm
        disparity[depth_mm <= 0] = 0
    
    # Confidence: einfache Heuristik — je größer die Depth, desto niedriger
    # 1.0 = hohe Konfidenz, 0.0 = niedrig
    confidence = np.ones_like(depth_mm, dtype=np.float32)
    # Sehr nahe oder sehr weit -> niedrigere Konfidenz
    confidence[depth_mm < 100] = 0.3   # < 10cm
    confidence[depth_mm > 5000] = 0.3  # > 5m
    confidence[depth_mm <= 0] = 0.0   # ungültig
    
    if output_path is None:
        output_path = depth_npz_path.replace('.npz', '_viewer.npz')
    
    np.savez_compressed(output_path,
        depth=depth_mm,
        disparity=disparity,
        confidence=confidence,
    )
    print(f"[OK] Gespeichert: {output_path}")
    print(f"  depth: {depth_mm.shape}, range: {depth_mm[depth_mm>0].min():.1f} - {depth_mm.max():.1f} mm")
    print(f"  disparity: {disparity.shape}, range: {disparity[disparity>0].min():.2f} - {disparity.max():.2f} px")
    print(f"  confidence: {confidence.shape}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <depth.npz> <calibration.npz> [output.npz]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
