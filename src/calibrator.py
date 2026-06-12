import numpy as np
from pathlib import Path
from typing import Dict, Any

class Calibrator:
    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Nicht gefunden: {path}")
        data = np.load(str(path))
        for key in ['K_left', 'K_right', 'T']:
            if key not in data:
                raise KeyError(f"Fehlender Key '{key}'. Vorhanden: {list(data.keys())}")
        self.K_left = data['K_left'].astype(np.float64)
        self.K_right = data['K_right'].astype(np.float64)
        self.T_raw = data['T'].astype(np.float64)
        self.T = self.T_raw / 1000.0
        self.baseline = float(abs(self.T[0]))
        self.focal_length = float(self.K_left[0, 0])

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        disparity = disparity.astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (self.focal_length * self.baseline) / disparity
            depth[disparity <= 0.01] = 0.0
            depth[depth > 50.0] = 0.0
        return depth.astype(np.float32)

    def get_info(self) -> Dict[str, Any]:
        return {
            'baseline_m': self.baseline,
            'focal_length_px': self.focal_length,
            'T_mm': self.T_raw.tolist(),
            'T_m': self.T.tolist(),
        }
