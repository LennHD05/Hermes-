#!/usr/bin/env python3
"""
Depth Map Viewer — interaktiv, für monocular Depth Anything .npz Files.

Features:
    - Farbcodierte Depth-Map (Inferno Colormap)
    - Legende oben links mit min/max/aktueller Depth
    - Hover: Live-Depth-Anzeige in Metern + mm
    - Klick 1: Punkt setzen (grüner Marker, Koordinaten in Tabelle)
    - Klick 2: Zweiter Punkt → Abstand in Metern + Pixel-Abstand
    - Klick 3+: Neuer Durchgang (Punkt 1 wird zu Punkt 2, neuer Punkt 1)
    - Tastatureingaben:
        [r]  Reset alle Punkte
        [s]  Screenshot speichern
        [q]  / [Esc]  Beenden

Usage:
    python3 depth_viewer.py output/depth_000000.npz
"""

import sys
import os
import numpy as np
from datetime import datetime

# ---- Headless-safe matplotlib ----
import matplotlib
matplotlib.use('Qt5Agg')  # Qt5 — funktioniert auf Jetson mit Display
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt_depth(val_m):
    """Wert in Metern als lesbarer String."""
    if val_m <= 0:
        return "ungültig"
    if val_m < 1.0:
        return f"{val_m * 1000:.0f} mm"
    return f"{val_m:.3f} m"


def pixel_distance(p1, p2):
    """Euklidischer Abstand in Pixeln."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def depth_at(depth_map, px, py):
    """Sichere Depth-Abfrage."""
    h, w = depth_map.shape
    px, py = int(round(px)), int(round(py))
    if 0 <= px < w and 0 <= py < h:
        return float(depth_map[py, px])
    return None


# ---------------------------------------------------------------------------
# Main Viewer
# ---------------------------------------------------------------------------

class DepthViewer:
    def __init__(self, npz_path):
        # Load depth map
        data = np.load(npz_path)
        self.depth = data["depth"].astype(np.float32)
        self.h, self.w = self.depth.shape

        # --- Try to load calibration for 3D distance ---
        # (best-effort — viewer works without it)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.baseline = None
        self._try_load_calibration()

        # valid depth range
        valid = self.depth[self.depth > 0]
        self.d_min = float(valid.min()) if valid.size else 0.0
        self.d_max = float(valid.max()) if valid.size else 1.0

        # State
        self.points = []          # [(px, py, depth_m), ...]
        self.marker_artists = []
        self.line_artist = None
        self.text_artist = None

        # --- Figure ---
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(
            f"Depth Viewer — {os.path.basename(npz_path)}"
        )

        # Depth image (inferno, 0 = transparent)
        norm = Normalize(vmin=self.d_min, vmax=self.d_max)
        self.depth_rgba = plt.cm.inferno(norm(self.depth))
        # Make zero-depth pixels transparent
        self.depth_rgba[self.depth <= 0, 3] = 0.15
        self.ax.imshow(self.depth_rgba, aspect="auto")

        self.ax.set_title(
            f"{os.path.basename(npz_path)}  |  "
            f"Depth range: {fmt_depth(self.d_min)} – {fmt_depth(self.d_max)}  |  "
            f"Resolution: {self.w}×{self.h}"
        )
        self.ax.set_xlabel("X (px)")
        self.ax.set_ylabel("Y (px)")

        # --- Hover legend (top-left) ---
        self.hover_text = self.ax.text(
            0.01, 0.99, "Hover …",
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=9, family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
        )

        # --- Measurement text (bottom-left) ---
        self.measure_text = self.ax.text(
            0.01, 0.01, "",
            transform=self.ax.transAxes,
            va="bottom", ha="left",
            fontsize=9, family="monospace",
            color="lime",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
        )

        # --- Point table (top-right) ---
        self.table_text = self.ax.text(
            0.99, 0.99, "",
            transform=self.ax.transAxes,
            va="top", ha="right",
            fontsize=8, family="monospace",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
        )

        # Events
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Keyboard hint
        self.ax.text(
            0.99, 0.01, "[r] Reset  [s] Screenshot  [q/Esc] Quit",
            transform=self.ax.transAxes,
            va="bottom", ha="right",
            fontsize=7, color="gray",
        )

        plt.tight_layout()

    # ------------------------------------------------------------------
    # Calibration loader (best-effort)
    # ------------------------------------------------------------------

    def _try_load_calibration(self):
        """Try common calibration file paths."""
        candidates = [
            "/media/data/LaborRobotikSS26/robotic-ss-2026/Sensorics/Stereo-Camera/stereo_params_Best.npz",
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                calib = np.load(path)
                if "mtxL" in calib:
                    mtx = calib["mtxL"]
                    self.fx = float(mtx[0, 0])
                    self.fy = float(mtx[1, 1])
                    self.cx = float(mtx[0, 2])
                    self.cy = float(mtx[1, 2])
                if "T" in calib:
                    T = calib["T"]
                    self.baseline = float(abs(T[0])) / 1000.0  # mm → m
                print(f"[CALIB] Loaded: {path}")
                print(f"  fx={self.fx:.1f}  fy={self.fy:.1f}  "
                      f"cx={self.cx:.1f}  cy={self.cy:.1f}  "
                      f"baseline={self.baseline:.4f} m")
                return
            except Exception as e:
                print(f"[CALIB] Error loading {path}: {e}")
                continue
        print("[CALIB] No calibration found — 3D distance estimation unavailable")

    # ------------------------------------------------------------------
    # 3D helpers
    # ------------------------------------------------------------------

    def estimate_3d(self, px, py, depth_m):
        """Back-project pixel + depth to 3D (if calibration available)."""
        if self.fx is None or self.fy is None or self.cx is None:
            return None
        z = depth_m
        x = (px - self.cx) * z / self.fx
        y = (py - self.cy) * z / self.fy
        return np.array([x, y, z])

    def distance_3d(self, p1, p2):
        """3D Euclidean distance between two (px, py, depth_m)."""
        pt1 = self.estimate_3d(*p1)
        pt2 = self.estimate_3d(*p2)
        if pt1 is None or pt2 is None:
            return None
        return float(np.linalg.norm(pt1 - pt2))

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _draw_markers(self):
        """Redraw all point markers and connecting line."""
        # Clear old
        for a in self.marker_artists:
            a.remove()
        self.marker_artists.clear()
        if self.line_artist:
            self.line_artist.remove()
            self.line_artist = None
        if self.text_artist:
            self.text_artist.remove()
            self.text_artist = None

        colors = ["lime", "cyan", "yellow", "magenta"]
        for i, (px, py, d) in enumerate(self.points):
            c = colors[i % len(colors)]
            artist = self.ax.plot(
                px, py, "o", markersize=8,
                markerfacecolor=c, markeredgecolor="black",
                markeredgewidth=1.5, zorder=5,
            )[0]
            self.marker_artists.append(artist)

            # Label next to marker
            self.ax.annotate(
                f"P{i + 1}\n{fmt_depth(d)}",
                xy=(px, py), xytext=(10, -10),
                textcoords="offset points",
                fontsize=7, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

        # Line between points
        if len(self.points) >= 2:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.line_artist = self.ax.plot(
                xs, ys, "-", color="white", linewidth=1.5, alpha=0.7, zorder=4,
            )[0]

            # Distance labels between pairs
            for i in range(len(self.points) - 1):
                p1 = self.points[i]
                p2 = self.points[i + 1]
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2

                dist_3d = self.distance_3d(p1, p2)
                px_dist = pixel_distance(
                    (p1[0], p1[1]), (p2[0], p2[1])
                )

                if dist_3d is not None:
                    label = f"{fmt_depth(dist_3d)} ({px_dist:.0f} px)"
                else:
                    label = f"{px_dist:.0f} px"

                self.ax.text(
                    mid_x, mid_y, label,
                    fontsize=8, color="yellow", fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                )

    def _update_table(self):
        """Update the point table (top-right)."""
        if not self.points:
            self.table_text.set_text("")
            return

        lines = ["═══ Punkte ═══"]
        for i, (px, py, d) in enumerate(self.points):
            lines.append(f"P{i + 1}: ({px}, {py}) — {fmt_depth(d)}")

        if len(self.points) >= 2:
            lines.append("─" * 18)
            for i in range(len(self.points) - 1):
                p1 = self.points[i]
                p2 = self.points[i + 1]
                dist_3d = self.distance_3d(p1, p2)
                px_dist = pixel_distance(
                    (p1[0], p1[1]), (p2[0], p2[1])
                )
                if dist_3d is not None:
                    lines.append(
                        f"d(P{i + 1}→P{i + 2}): {fmt_depth(dist_3d)} "
                        f"({px_dist:.0f} px)"
                    )
                else:
                    lines.append(
                        f"d(P{i + 1}→P{i + 2}): {px_dist:.0f} px"
                    )

        # Total distance if > 2 points
        if len(self.points) > 2:
            total_px = sum(
                pixel_distance(
                    (self.points[i][0], self.points[i][1]),
                    (self.points[i + 1][0], self.points[i + 1][1]),
                )
                for i in range(len(self.points) - 1)
            )
            lines.append(f"Gesamt: {total_px:.0f} px")

        self.table_text.set_text("\n".join(lines))

    def _update_measure_text(self, px, py, d):
        """Bottom-left measurement info."""
        if d is None or d <= 0:
            self.measure_text.set_text("")
            return
        lines = [
            f"Cursor: ({int(px)}, {int(py)})",
            f"Depth:  {fmt_depth(d)}",
        ]
        # If we have one point, show distance to cursor
        if len(self.points) == 1:
            p1 = self.points[0]
            px_dist = pixel_distance((p1[0], p1[1]), (px, py))
            dist_3d = None
            if self.fx:
                dist_3d = self.distance_3d(p1, (px, py, d))
            if dist_3d:
                lines.append(f"→ P1: {fmt_depth(dist_3d)} ({px_dist:.0f} px)")
            else:
                lines.append(f"→ P1: {px_dist:.0f} px")
        self.measure_text.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_hover(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.hover_text.set_text("Hover …")
            self.fig.canvas.draw_idle()
            return

        px, py = int(event.xdata), int(event.ydata)
        d = depth_at(self.depth, px, py)

        if d is None:
            self.hover_text.set_text("Hover …")
        else:
            self.hover_text.set_text(
                f"x: {px}  y: {py}\n"
                f"depth: {fmt_depth(d)}"
            )
            self._update_measure_text(px, py, d)

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        px, py = int(event.xdata), int(event.ydata)
        d = depth_at(self.depth, px, py)

        if d is None or d <= 0:
            print(f"  ({px}, {py}): kein gültiger Depth-Wert")
            return

        # Add point (max 4 for clarity)
        if len(self.points) >= 4:
            self.points = [self.points[-1]]  # Keep last as P1

        self.points.append((px, py, d))

        print(f"  P{len(self.points)}: ({px}, {py}) → depth={fmt_depth(d)}")

        self._draw_markers()
        self._update_table()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = event.key.lower()

        if key == "r":
            # Reset
            self.points.clear()
            for a in self.marker_artists:
                a.remove()
            self.marker_artists.clear()
            if self.line_artist:
                self.line_artist.remove()
            if self.text_artist:
                self.text_artist.remove()
            self.table_text.set_text("")
            self.measure_text.set_text("")
            # Force redraw
            self.fig.canvas.draw_idle()
            print("[RESET] Alle Punkte gelöscht")

        elif key == "s":
            # Screenshot
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"depth_viewer_{ts}.png"
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[SCREENSHOT] {out}")

        elif key in ("q", "escape"):
            print("[EXIT]")
            plt.close(self.fig)

    # ------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------

    def show(self):
        print("=" * 65)
        print("  Depth Map Viewer")
        print("=" * 65)
        print(f"  File:    {self.w}×{self.h} px")
        print(f"  Depth:   {fmt_depth(self.d_min)} – {fmt_depth(self.d_max)}")
        print(f"  Calib:   {'loaded' if self.fx else 'not found'}")
        print()
        print("  Hover   → Live-Depth + Koordinaten (oben links)")
        print("  Click   1 → Punkt setzen")
        print("  Click   2 → Abstand P1→P2 (3D wenn Calib verfügbar)")
        print("  Click 3+ → Neuer Durchgang")
        print("  [r]     → Reset")
        print("  [s]     → Screenshot")
        print("  [q/Esc] → Beenden")
        print("=" * 65)
        plt.show()


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

    viewer = DepthViewer(npz_path)
    viewer.show()
