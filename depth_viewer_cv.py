#!/usr/bin/env python3
"""
Depth Map Viewer — OpenCV-basiert, mit Kalibrierung.

Features:
    - Farbcodierte Depth-Map (Jet-Colormap)
    - Legende oben links: min/max/aktuelle Depth
    - Hover: Live-Depth
    - Klick: Punkte setzen
    - Abstandsmessung (Pixel + 3D wenn kalibriert)
    - KALIBRIERUNG: [k] → 2 Punkte klicken + echte Entfernung eingeben
      → Viewer skaliert automatisch alle Werte in Metern

Usage:
    python3 depth_viewer_cv.py output/depth_000000.npz
"""

import sys, os, cv2, numpy as np
from datetime import datetime


def fmt_depth(val_m):
    if val_m <= 0:
        return "ungueltig"
    if val_m < 1.0:
        return f"{val_m * 1000:.0f} mm"
    return f"{val_m:.3f} m"


def pixel_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class DepthViewerCV:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.depth_raw = data["depth"].astype(np.float32)
        self.h, self.w = self.depth_raw.shape
        self.npz_path = npz_path

        # Skalierungsfaktor (roh → Meter)
        # None = nicht kalibriert
        self.scale_factor = None
        self.depth = self.depth_raw.copy()

        # Calib laden
        self.fx = self.fy = self.cx = self.cy = None
        self._try_load_calibration()

        # State
        self.points = []
        self.hover = None
        self.brightness = 1.0
        self.calib_mode = False
        self.calib_points = []

        valid = self.depth_raw[self.depth_raw > 0]
        self.d_min = float(valid.min()) if valid.size else 0.0
        self.d_max = float(valid.max()) if valid.size else 1.0

        self._render_colored()

        cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Depth Viewer", self._mouse_cb)

        print("=" * 60)
        print("  Depth Map Viewer (OpenCV)")
        print("=" * 60)
        print(f"  File:  {os.path.basename(npz_path)}  ({self.w}x{self.h})")
        print(f"  Raw Depth Range: {self.d_min:.2f} – {self.d_max:.2f}")
        print(f"  Kalibriert: {'Ja' if self.scale_factor else 'Nein'}")
        print(f"  Calib: {'geladen' if self.fx else 'nicht gefunden'}")
        print()
        print("  WICHTIG: Erst kalibrieren!")
        print("  [k] Kalibrierung starten:")
        print("      1. Klick auf Punkt 1 (z.B. vordere Kante)")
        print("      2. Klick auf Punkt 2 (z.B. hintere Kante)")
        print("      3. Echte Entfernung in Metern eingeben")
        print("      → Viewer skaliert automatisch")
        print()
        print("  Links-Klick  → Punkt setzen")
        print("  Rechts-Klick → Letzten Punkt entfernen")
        print("  [r] Reset  [s] Screenshot  [+/-] Helligkeit  [q] Beenden")
        print("=" * 60)

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

    def _render_colored(self):
        d = self.depth if self.scale_factor else self.depth_raw
        valid = d > 0
        d_min = d[valid].min() if valid.any() else 0
        d_max = d[valid].max() if valid.any() else 1
        normalized = (d - d_min) / max(d_max - d_min, 1e-6)
        normalized = np.clip(normalized * self.brightness, 0, 1)
        uint8 = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(uint8, cv2.COLORMAP_JET)
        colored[d <= 0] = [0, 0, 0]
        self.colored = colored

    def _render_overlay(self):
        img = self.colored.copy()
        h, w = img.shape[:2]
        d = self.depth if self.scale_factor else self.depth_raw

        # Legende oben links
        legend_h = 105
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (340, legend_h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        status = f"KALIBRIERT ({self.scale_factor:.4f})" if self.scale_factor else "NICHT KALIBRIERT"
        cv2.putText(img, f"Status: {status}", (10, 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if self.scale_factor else (0, 0, 255), 1)
        cv2.putText(img, f"Min: {fmt_depth(d[d>0].min()) if (d>0).any() else 'N/A'}", (10, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"Max: {fmt_depth(d.max())}", (10, 58),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if self.hover:
            px, py = self.hover
            val = d[py, px] if 0 <= px < w and 0 <= py < h else 0
            cv2.putText(img, f"({px},{py}) {fmt_depth(val)}", (10, 76),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(img, "Hover fuer Depth", (10, 76),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(img, f"[k] Kalibrieren  [+/-] Hell:{self.brightness:.1f}", (10, 94),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Kalibrierungsmodus
        if self.calib_mode:
            cv2.rectangle(img, (w//2 - 200, h//2 - 30), (w//2 + 200, h//2 + 30), (0, 0, 255), -1)
            cv2.putText(img, f"KALIBRIERUNG: {len(self.calib_points)}/2 Punkte", (w//2 - 180, h//2 + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Punkte
        colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for i, (px, py, depth_val) in enumerate(self.points):
            c = colors[i % len(colors)]
            cv2.circle(img, (px, py), 6, c, 2)
            cv2.circle(img, (px, py), 2, c, -1)
            cv2.putText(img, f"P{i+1}", (px + 8, py - 8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            cv2.putText(img, fmt_depth(depth_val), (px + 8, py + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        # Linien + Abstände
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

        # Tabelle unten rechts
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
            for i, (px, py, depth_val) in enumerate(self.points):
                c = colors[i % len(colors)]
                cv2.putText(img, f"P{i+1}: ({px},{py}) {fmt_depth(depth_val)}",
                             (ox + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
                ty += 18

            if len(self.points) >= 2:
                ty += 5
                cv2.line(img, (ox + 5, ty - 3), (ox + tw - 5, ty - 3), (100, 100, 100), 1)
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

    def _mouse_cb(self, event, x, y, flags, param):
        x = max(0, min(x, self.w - 1))
        y = max(0, min(y, self.h - 1))
        self.hover = (x, y)

        if self.calib_mode and event == cv2.EVENT_LBUTTONDOWN:
            val = self.depth_raw[y, x]
            if val > 0:
                self.calib_points.append((x, y, val))
                print(f"  Kalib-Punkt {len(self.calib_points)}: ({x}, {y}) raw={val:.4f}")
                if len(self.calib_points) == 2:
                    self._do_calibration()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            val = self.depth[y, x] if self.scale_factor else self.depth_raw[y, x]
            if val > 0:
                self.points.append((x, y, float(val)))
                print(f"  P{len(self.points)}: ({x}, {y}) -> {fmt_depth(val)}")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"  Entfernt: P{len(self.points)+1}")

    def _do_calibration(self):
        """Kalibrierung: 2 Punkte + echte Entfernung → Skalenfaktor."""
        p1 = self.calib_points[0]
        p2 = self.calib_points[1]

        # Pixel-Abstand
        px_dist = pixel_distance((p1[0], p1[1]), (p2[0], p2[1]))

        # Raw-Differenz
        raw_diff = abs(p1[2] - p2[2])

        print(f"\n  Kalibrierung:")
        print(f"    P1: ({p1[0]}, {p1[1]}) raw={p1[2]:.4f}")
        print(f"    P2: ({p2[0]}, {p2[1]}) raw={p2[2]:.4f}")
        print(f"    Pixel-Abstand: {px_dist:.1f}")
        print(f"    Raw-Differenz: {raw_diff:.4f}")

        # Echte Entfernung abfragen
        print(f"\n  Echte Entfernung zwischen P1 und P2 in Metern eingeben:")
        print(f"  (z.B. 0.25 fuer 25cm)")

        # Input über OpenCV-Text-Eingabe
        # Wir nutzen einen simplen Ansatz: Konsolen-Input
        try:
            real_dist = float(input("  Echte Entfernung (m): "))
        except (ValueError, EOFError):
            print("  Ungültige Eingabe — Kalibrierung abgebrochen")
            self.calib_mode = False
            self.calib_points.clear()
            return

        if real_dist <= 0 or raw_diff <= 0:
            print("  Ungültige Werte — Kalibrierung abgebrochen")
            self.calib_mode = False
            self.calib_points.clear()
            return

        # Skalenfaktor: echte Meter pro Raw-Einheit
        # raw_diff * scale_factor = real_dist
        self.scale_factor = real_dist / raw_diff

        # Depth-Map skalieren
        self.depth = self.depth_raw * self.scale_factor

        print(f"\n  ✓ KALIBRIERT!")
        print(f"    Skalenfaktor: {self.scale_factor:.6f} m/raw")
        print(f"    Neue Depth-Range: {fmt_depth(self.depth[self.depth>0].min())} – {fmt_depth(self.depth.max())}")
        print(f"    [k] erneut kalibrieren zum Feintuning\n")

        self._render_colored()
        self.calib_mode = False
        self.calib_points.clear()

    def run(self):
        while True:
            img = self._render_overlay()
            cv2.imshow("Depth Viewer", img)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:
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

            elif key == ord('k'):
                self.calib_mode = True
                self.calib_points.clear()
                print("\n  === KALIBRIERUNG ===")
                print("  Klick auf 2 Punkte mit bekannter Entfernung")
                print("  (z.B. vordere und hintere Kante einer Kiste)")

            elif key == ord('+') or key == ord('='):
                self.brightness = min(3.0, self.brightness + 0.1)
                self._render_colored()

            elif key == ord('-'):
                self.brightness = max(0.1, self.brightness - 0.1)
                self._render_colored()

        cv2.destroyAllWindows()


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
