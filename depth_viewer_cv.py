#!/usr/bin/env python3
"""
Depth Map Viewer — OpenCV-basiert, kein Qt/matplotlib nötig.
Läuft auf headless Jetson ohne Display-Probleme.

Features:
    - Farbcodierte Depth-Map (Jet-Colormap)
    - Legende oben links: min/max/aktuelle Depth
    - Hover: Live-Depth in Metern (Maustasten-Highlight)
    - Klick links: Punkt setzen (grün)
    - Klick rechts: Letzten Punkt entfernen
    - Abstandsmessung zwischen 2 Punkten (3D wenn Calib verfügbar)
    - Tastatureingaben:
        [r]  Reset
        [s]  Screenshot
        [+]/[-]  Colormap-Helligkeit anpassen
        [q]/[Esc]  Beenden

Usage:
    python3 depth_viewer_cv.py output/depth_000000.npz
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_depth(val):
    """Depth-Wert als lesbarer String."""
    if val <= 0:
        return "ungueltig"
    if val < 1.0:
        return f"{val * 1000:.0f} mm"
    return f"{val:.3f} m"


def pixel_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ---------------------------------------------------------------------------
# Main Viewer
# ---------------------------------------------------------------------------

class DepthViewerCV:
    def __init__(self, npz_path):
        # Load depth map
        data = np.load(npz_path)
        self.depth = data["depth"].astype(np.float32)
        self.h, self.w = self.depth.shape
        self.npz_path = npz_path

        # Valid range
        valid = self.depth[self.depth > 0]
        self.d_min = float(valid.min()) if valid.size else 0.0
        self.d_max = float(valid.max()) if valid.size else 1.0

        # State
        self.points = []       # [(px, py, depth), ...]
        self.hover = None      # (px, py) or None
        self.brightness = 1.0

        # --- Load calibration ---
        self.fx = self.fy = self.cx = self.cy = None
        self._try_load_calibration()

        # Pre-render colored depth (updated on brightness change)
        self._render_colored()

        # Window
        cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Depth Viewer", self._mouse_cb)

        print("=" * 60)
        print("  Depth Map Viewer (OpenCV)")
        print("=" * 60)
        print(f"  File:  {os.path.basename(npz_path)}  ({self.w}x{self.h})")
        print(f"  Depth: {fmt_depth(self.d_min)} – {fmt_depth(self.d_max)}")
        print(f"  Calib: {'geladen' if self.fx else 'nicht gefunden'}")
        print()
        print("  Links-Klick  → Punkt setzen")
        print("  Rechts-Klick → Letzten Punkt entfernen")
        print("  [r] Reset  [s] Screenshot  [+/-] Helligkeit  [q] Beenden")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _try_load_calibration(self):
        candidates = [
            "/media/data/LaborRobotikSS26/robotic-ss-2026/Sensorics/Stereo-Camera/stereo_params_Best.npz",
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                c = np.load(path)
                if "mtxL" in c:
                    m = c["mtxL"]
                    self.fx = float(m[0, 0])
                    self.fy = float(m[1, 1])
                    self.cx = float(m[0, 2])
                    self.cy = float(m[1, 2])
                print(f"  [CALIB] {path}")
                return
            except Exception as e:
                print(f"  [CALIB] Fehler: {e}")

    def _estimate_3d(self, px, py, d):
        if self.fx is None:
            return None
        z = d
        x = (px - self.cx) * z / self.fx
        y = (py - self.cy) * z / self.fy
        return np.array([x, y, z])

    def _distance_3d(self, p1, p2):
        pt1 = self._estimate_3d(*p1)
        pt2 = self._estimate_3d(*p2)
        if pt1 is None or pt2 is None:
            return None
        return float(np.linalg.norm(pt1 - pt2))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_colored(self):
        """Depth → farbcodiertes Bild."""
        # Normalize
        normalized = (self.depth - self.d_min) / max(self.d_max - self.d_min, 1e-6)
        normalized = np.clip(normalized * self.brightness, 0, 1)
        uint8 = (normalized * 255).astype(np.uint8)

        # Jet colormap
        colored = cv2.applyColorMap(uint8, cv2.COLORMAP_JET)

        # Zero-depth → schwarz
        colored[self.depth <= 0] = [0, 0, 0]

        self.colored = colored

    def _render_overlay(self):
        """Overlay mit Legende, Punkten, Messungen."""
        img = self.colored.copy()
        h, w = img.shape[:2]

        # --- Legende oben links ---
        legend_h = 90
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (320, legend_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        cv2.putText(img, f"Min: {fmt_depth(self.d_min)}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Max: {fmt_depth(self.d_max)}", (10, 45),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Hover-Info
        if self.hover:
            px, py = self.hover
            d = self.depth[py, px] if 0 <= px < w and 0 <= py < h else 0
            cv2.putText(img, f"({px},{py}) {fmt_depth(d)}", (10, 65),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(img, "Hover fuer Depth", (10, 65),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(img, f"Brightness: {self.brightness:.1f}", (10, 82),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # --- Punkte zeichnen ---
        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for i, (px, py, d) in enumerate(self.points):
            c = colors[i % len(colors)]
            cv2.circle(img, (px, py), 6, c, 2)
            cv2.circle(img, (px, py), 2, c, -1)
            cv2.putText(img, f"P{i+1}", (px + 8, py - 8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            cv2.putText(img, fmt_depth(d), (px + 8, py + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        # --- Linien + Abstände ---
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                p1 = self.points[i]
                p2 = self.points[i + 1]
                c = colors[i % len(colors)]
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), c, 2)

                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2

                dist_3d = self._distance_3d(p1, p2)
                px_dist = pixel_distance((p1[0], p1[1]), (p2[0], p2[1]))

                if dist_3d is not None:
                    label = f"{fmt_depth(dist_3d)} ({px_dist:.0f}px)"
                else:
                    label = f"{px_dist:.0f} px"

                cv2.putText(img, label, (mid_x, mid_y - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # --- Punkte-Tabelle unten rechts ---
        if self.points:
            tw, th = 280, 25 + len(self.points) * 20 + (len(self.points) - 1) * 18
            if len(self.points) > 1:
                th += 10
            ox, oy = w - tw - 10, h - th - 10
            overlay = img.copy()
            cv2.rectangle(overlay, (ox, oy), (ox + tw, oy + th), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

            ty = oy + 18
            cv2.putText(img, "== Punkte ==", (ox + 8, ty),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            ty += 18
            for i, (px, py, d) in enumerate(self.points):
                c = colors[i % len(colors)]
                cv2.putText(img, f"P{i+1}: ({px},{py}) {fmt_depth(d)}",
                             (ox + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
                ty += 18

            if len(self.points) >= 2:
                ty += 5
                cv2.line(img, (ox + 5, ty - 3), (ox + tw - 5, ty - 3),
                         (100, 100, 100), 1)
                for i in range(len(self.points) - 1):
                    p1 = self.points[i]
                    p2 = self.points[i + 1]
                    dist_3d = self._distance_3d(p1, p2)
                    px_dist = pixel_distance((p1[0], p1[1]), (p2[0], p2[1]))
                    if dist_3d:
                        label = f"d(P{i+1}->P{i+2}): {fmt_depth(dist_3d)} ({px_dist:.0f}px)"
                    else:
                        label = f"d(P{i+1}->P{i+2}): {px_dist:.0f} px"
                    cv2.putText(img, label, (ox + 8, ty),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                    ty += 18

        return img

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------

    def _mouse_cb(self, event, x, y, flags, param):
        # Clamp
        x = max(0, min(x, self.w - 1))
        y = max(0, min(y, self.h - 1))
        self.hover = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            d = self.depth[y, x]
            if d > 0:
                self.points.append((x, y, float(d)))
                print(f"  P{len(self.points)}: ({x}, {y}) -> {fmt_depth(d)}")
            else:
                print(f"  ({x}, {y}): kein gueltiger Depth-Wert")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"  Entfernt: P{len(self.points)+1} ({removed[0]}, {removed[1]})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            img = self._render_overlay()
            cv2.imshow("Depth Viewer", img)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # q or Esc
                print("[EXIT]")
                break

            elif key == ord('r'):
                self.points.clear()
                print("[RESET] Alle Punkte geloescht")

            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = f"depth_viewer_{ts}.png"
                cv2.imwrite(out, img)
                print(f"[SCREENSHOT] {out}")

            elif key == ord('+') or key == ord('='):
                self.brightness = min(3.0, self.brightness + 0.1)
                self._render_colored()

            elif key == ord('-'):
                self.brightness = max(0.1, self.brightness - 0.1)
                self._render_colored()

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <depth_map.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        sys.exit(1)

    viewer = DepthViewerCV(npz_path)
    viewer.run()
