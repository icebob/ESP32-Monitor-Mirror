#!/usr/bin/env python3
"""
Pixel Update Screenshot Sender (Serial version)
Captures a monitor and streams run-length encoded pixel updates
to the ESP32 receiver over USB serial.
"""

import argparse
import ctypes
import struct
import sys
import threading
import time
from typing import Optional, Sequence

import cv2
import mss
import numpy as np
import serial
import win32con
import win32gui
import win32ui

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
RUN_HEADER_VERSION = 0x02


class ScreenshotSerialSender:
    def __init__(
        self,
        serial_port: str,
        baud_rate: int,
        monitor_index: Optional[int],
        prefer_largest: bool,
        target_fps: float,
        threshold: int,
        full_frame: bool,
        show_stats: bool,
        max_updates_per_frame: int,
        rotate_deg: int,
        crop: Optional[tuple[int, int, int, int]] = None,
        window_title: Optional[str] = None,
    ) -> None:
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.monitor_index = monitor_index
        self.prefer_largest = prefer_largest
        self.target_fps = target_fps
        self.threshold = threshold
        self.full_frame = full_frame
        self.show_stats = show_stats
        self.max_updates_per_frame = max_updates_per_frame
        self.rotate_deg = rotate_deg
        self.crop = crop
        self.window_title = window_title

        self.ser: Optional[serial.Serial] = None
        self.prev_rgb: Optional[np.ndarray] = None
        self.sent_initial_full: bool = False
        self.frame_id: int = 0
        self.monitor: Optional[dict] = None
        self.sct: Optional[mss.mss] = None
        self.hwnd: Optional[int] = None
        self._fit_offset: tuple[int, int] = (0, 0)  # letterbox offset (x, y)
        self._fit_size: tuple[int, int] = (DISPLAY_WIDTH, DISPLAY_HEIGHT)  # scaled content size
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._ack_event = threading.Event()

    # Connection helpers -------------------------------------------------
    def connect(self) -> bool:
        try:
            print(f"[SERIAL] Opening {self.serial_port} at {self.baud_rate} baud")
            self.ser = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=5,
                write_timeout=10,
            )
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            time.sleep(0.5)  # let ESP32 settle
            print("[SERIAL] OK Connected")
            return True
        except Exception as exc:
            print(f"[SERIAL] FAIL {type(exc).__name__}: {exc}")
            return False

    def disconnect(self) -> None:
        if self.ser:
            self.ser.close()
        self.ser = None
        self.prev_rgb = None
        self.sent_initial_full = False
        print("[SERIAL] Disconnected")

    # Touch input handling -----------------------------------------------
    def _mouse_move(self, x: int, y: int) -> None:
        ctypes.windll.user32.SetCursorPos(x, y)

    def _mouse_down(self) -> None:
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN

    def _mouse_up(self) -> None:
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP

    def _map_touch_to_screen(self, tx: int, ty: int) -> tuple[int, int]:
        """Map touch coords (800x480) to screen pixel coords, accounting for letterbox."""
        # Reverse the letterbox: map display coords to content coords
        x_off, y_off = self._fit_offset
        fit_w, fit_h = self._fit_size
        # Clamp to content area
        cx = max(0, min(tx - x_off, fit_w - 1))
        cy = max(0, min(ty - y_off, fit_h - 1))

        if self.hwnd:
            try:
                left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                w = right - left
                h = bottom - top
                sx = left + int(cx * w / fit_w)
                sy = top + int(cy * h / fit_h)
                return sx, sy
            except Exception:
                return tx, ty
        if not self.monitor:
            return tx, ty
        mon_left = self.monitor.get("left", 0)
        mon_top = self.monitor.get("top", 0)
        mon_w = self.monitor.get("width", DISPLAY_WIDTH)
        mon_h = self.monitor.get("height", DISPLAY_HEIGHT)
        sx = mon_left + int(cx * mon_w / fit_w)
        sy = mon_top + int(cy * mon_h / fit_h)
        return sx, sy

    def wait_for_ack(self, timeout: float = 5.0) -> bool:
        self._ack_event.clear()
        return self._ack_event.wait(timeout)

    def _serial_reader(self) -> None:
        """Background thread: reads ACK and touch events from serial."""
        buf = bytearray()
        while self._running and self.ser and self.ser.is_open:
            try:
                data = self.ser.read(self.ser.in_waiting or 1)
                if not data:
                    continue
                buf.extend(data)
                while buf:
                    # Check for ACK byte (0x06)
                    if buf[0] == 0x06:
                        buf = buf[1:]
                        self._ack_event.set()
                        continue
                    idx = buf.find(b'TCH')
                    if idx < 0:
                        buf = buf[-2:] if len(buf) > 2 else buf
                        break
                    if idx > 0:
                        buf = buf[idx:]
                    if len(buf) < 8:
                        break
                    tx = buf[3] | (buf[4] << 8)
                    ty = buf[5] | (buf[6] << 8)
                    ttype = buf[7]
                    buf = buf[8:]
                    sx, sy = self._map_touch_to_screen(tx, ty)
                    if ttype == 0:  # press
                        self._mouse_move(sx, sy)
                        self._mouse_down()
                    elif ttype == 1:  # move
                        self._mouse_move(sx, sy)
                    elif ttype == 2:  # release
                        self._mouse_up()
            except Exception:
                if self._running:
                    time.sleep(0.01)

    def _start_reader_thread(self) -> None:
        self._running = True
        self._reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self._reader_thread.start()

    def _stop_reader_thread(self) -> None:
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    # Window capture helpers ---------------------------------------------
    @staticmethod
    def list_windows() -> None:
        """List all visible windows with titles."""
        results = []

        def _enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                title = win32gui.GetWindowText(hwnd)
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                w, h = right - left, bottom - top
                if w > 0 and h > 0:
                    results.append((hwnd, title, w, h))

        win32gui.EnumWindows(_enum_cb, None)
        print(f"{'HWND':<12} {'Size':>10}  Title")
        print("-" * 60)
        for hwnd, title, w, h in sorted(results, key=lambda r: r[1].lower()):
            print(f"0x{hwnd:08X}  {w:>4}x{h:<4}  {title}")

    def find_window(self, title: str) -> bool:
        """Find window by partial title match (case-insensitive)."""
        title_lower = title.lower()
        matches = []

        def _enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                wt = win32gui.GetWindowText(hwnd)
                if wt and title_lower in wt.lower():
                    matches.append((hwnd, wt))

        win32gui.EnumWindows(_enum_cb, None)
        if not matches:
            print(f"[WIN] No window found matching '{title}'")
            return False
        self.hwnd = matches[0][0]
        print(f"[WIN] Matched: '{matches[0][1]}' (HWND=0x{self.hwnd:08X})")
        if len(matches) > 1:
            print(f"[WIN] Note: {len(matches)} windows matched, using first")
        return True

    def grab_window(self) -> Optional[np.ndarray]:
        """Capture window content using PrintWindow API."""
        if not self.hwnd:
            return None
        try:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            w = right - left
            h = bottom - top
            if w <= 0 or h <= 0:
                return None

            hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(bitmap)

            # PW_RENDERFULLCONTENT = 2 for better DirectX capture
            ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 2)

            bmp_info = bitmap.GetInfo()
            bmp_bits = bitmap.GetBitmapBits(True)
            frame = np.frombuffer(bmp_bits, dtype=np.uint8).reshape(
                bmp_info["bmHeight"], bmp_info["bmWidth"], 4
            )

            # Cleanup GDI objects
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)
            win32gui.DeleteObject(bitmap.GetHandle())

            # BGRA -> BGR
            return frame[:, :, :3].copy()
        except Exception as exc:
            print(f"[WIN] Capture failed: {exc}")
            return None

    # Monitor helpers ----------------------------------------------------
    @staticmethod
    def _select_monitor(
        monitors: Sequence[dict],
        monitor_index: Optional[int],
        prefer_largest: bool,
    ) -> Optional[dict]:
        if monitor_index is not None:
            if 1 <= monitor_index < len(monitors):
                return monitors[monitor_index]
            print(f"[MON] Invalid monitor index {monitor_index}; available 1..{len(monitors) - 1}")
            return None

        usable_monitors = monitors[1:]
        if not usable_monitors:
            return None

        if prefer_largest:
            return max(usable_monitors, key=lambda m: m.get("width", 0) * m.get("height", 0))

        return min(usable_monitors, key=lambda m: m.get("left", 0))

    def setup_capture(self) -> bool:
        # Window capture mode
        if self.window_title:
            if not self.find_window(self.window_title):
                return False
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            print(f"[WIN] Window size: {right - left}x{bottom - top}")
            return True

        # Monitor capture mode
        try:
            self.sct = mss.mss()
        except Exception as exc:
            print(f"[MON] Unable to start screen capture: {exc}")
            return False

        monitor = self._select_monitor(self.sct.monitors, self.monitor_index, self.prefer_largest)
        if not monitor:
            print("[MON] No monitor selected or available")
            return False

        # Apply crop region if specified
        if self.crop:
            cx, cy, cw, ch = self.crop
            monitor = {
                "left": monitor["left"] + cx,
                "top": monitor["top"] + cy,
                "width": cw,
                "height": ch,
            }

        self.monitor = monitor
        print(
            f"[MON] Using monitor at ({monitor['left']}, {monitor['top']}) "
            f"{monitor['width']}x{monitor['height']}"
        )
        return True

    def grab_frame(self) -> Optional[np.ndarray]:
        if self.hwnd:
            return self.grab_window()
        if not self.sct or not self.monitor:
            return None
        try:
            shot = self.sct.grab(self.monitor)
        except Exception as exc:
            print(f"[MON] Capture failed: {exc}")
            return None
        frame = np.array(shot)[:, :, :3]
        return frame

    # Conversion helpers -------------------------------------------------
    @staticmethod
    def rgb888_to_rgb565(rgb: np.ndarray) -> np.ndarray:
        r = (rgb[:, :, 0] >> 3).astype(np.uint16)
        g = (rgb[:, :, 1] >> 2).astype(np.uint16)
        b = (rgb[:, :, 2] >> 3).astype(np.uint16)
        return (r << 11) | (g << 5) | b

    def resize_and_convert(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = np.ascontiguousarray(frame)

        if self.rotate_deg == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_deg == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate_deg == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame.shape[:2]
        scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(frame, (new_w, new_h))
        resized = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        x_off = (DISPLAY_WIDTH - new_w) // 2
        y_off = (DISPLAY_HEIGHT - new_h) // 2
        resized[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        self._fit_offset = (x_off, y_off)
        self._fit_size = (new_w, new_h)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb565 = self.rgb888_to_rgb565(rgb)
        return rgb, rgb565

    # Packet creation ----------------------------------------------------
    def build_run_packets(self, rgb: np.ndarray, rgb565: np.ndarray) -> list[bytes]:
        if self.full_frame or not self.sent_initial_full or self.prev_rgb is None:
            mask = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH), dtype=bool)
        else:
            diff = np.abs(rgb.astype(np.int16) - self.prev_rgb.astype(np.int16))
            mask = diff.max(axis=2) > self.threshold

        # Vectorized RLE: process each row with numpy
        all_runs = []
        for y in range(DISPLAY_HEIGHT):
            row_mask = mask[y]
            if not row_mask.any():
                continue
            indices = np.where(row_mask)[0]
            colors = rgb565[y, indices]
            if len(indices) == 1:
                all_runs.append(np.array([[y, indices[0], 1, colors[0]]], dtype=np.uint16))
                continue
            # Break where index is non-consecutive or color changes
            breaks = np.where(
                (np.diff(indices) != 1) | (np.diff(colors) != 0)
            )[0] + 1
            starts = np.concatenate([[0], breaks])
            lengths = np.diff(np.concatenate([starts, [len(indices)]]))
            x_starts = indices[starts]
            run_colors = colors[starts]
            n = len(starts)
            row_runs = np.column_stack([
                np.full(n, y, dtype=np.uint16),
                x_starts.astype(np.uint16),
                lengths.astype(np.uint16),
                run_colors.astype(np.uint16),
            ])
            all_runs.append(row_runs)

        if not all_runs:
            return []

        runs_array = np.vstack(all_runs)
        total_runs = len(runs_array)

        packets: list[bytes] = []
        max_per = max(1, self.max_updates_per_frame)
        start = 0
        while start < total_runs:
            end = min(start + max_per, total_runs)
            count = end - start
            header = (
                b"PXUR"
                + bytes([RUN_HEADER_VERSION])
                + struct.pack("<I", self.frame_id)
                + struct.pack("<H", count)
            )
            payload = bytearray(header)
            payload.extend(runs_array[start:end].tobytes())
            packets.append(bytes(payload))
            start = end

        self.frame_id += len(packets)
        return packets

    # Main loop ----------------------------------------------------------
    def run(self) -> None:
        if not self.setup_capture():
            return
        if not self.connect():
            return

        frame_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        frame_count = 0
        sent_packets = 0
        sent_runs = 0
        start_t = time.time()

        self._start_reader_thread()
        print("[STREAM] Starting screenshot update loop (Ctrl+C to stop)")
        try:
            while True:
                frame_start = time.time()
                frame = self.grab_frame()
                if frame is None:
                    break

                rgb, rgb565 = self.resize_and_convert(frame)
                packets = self.build_run_packets(rgb, rgb565)
                self.prev_rgb = rgb

                if not packets and not self.sent_initial_full:
                    continue

                for pkt in packets:
                    runs_in_pkt = struct.unpack_from("<H", pkt, 9)[0]
                    try:
                        self.ser.write(pkt)
                        sent_packets += 1
                        sent_runs += runs_in_pkt
                        if runs_in_pkt > 0 and not self.wait_for_ack(10.0):
                            print("[ACK] Timeout - no ACK received")
                    except Exception as exc:
                        print(f"[SEND] Error: {type(exc).__name__}: {exc}")
                        break

                if not self.sent_initial_full:
                    self.sent_initial_full = True

                frame_count += 1
                now = time.time()
                elapsed_frame = now - frame_start
                if frame_delay > 0 and elapsed_frame < frame_delay:
                    time.sleep(frame_delay - elapsed_frame)

                if self.show_stats and now - start_t >= 2.0:
                    elapsed = now - start_t
                    fps_est = frame_count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[STATS] frames:{frame_count} packets:{sent_packets} "
                        f"runs:{sent_runs} fps~{fps_est:.1f}"
                    )
                    start_t = now
                    frame_count = 0
                    sent_packets = 0
                    sent_runs = 0

        except KeyboardInterrupt:
            print("\n[STREAM] Interrupted by user")
        finally:
            self._stop_reader_thread()
            self.disconnect()
            if self.sct:
                self.sct.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture a monitor or window and send pixel updates to ESP32 over serial"
    )
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=2000000, help="Baud rate (default 2000000)")
    parser.add_argument("--monitor-index", type=int, default=None, help="Monitor index (1-based)")
    parser.add_argument("--prefer-largest", action="store_true", help="Use largest monitor")
    parser.add_argument("--target-fps", type=float, default=10.0, help="Target FPS (default 10)")
    parser.add_argument("--threshold", type=int, default=5, help="Pixel change threshold (default 5)")
    parser.add_argument("--full-frame", action="store_true", help="Send every pixel every frame")
    parser.add_argument("--max-updates-per-frame", type=int, default=8000, help="Max runs per packet (default 8000)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotation degrees")
    parser.add_argument("--crop-x", type=int, default=None, help="Crop region X offset (pixels from monitor left)")
    parser.add_argument("--crop-y", type=int, default=None, help="Crop region Y offset (pixels from monitor top)")
    parser.add_argument("--crop-width", type=int, default=None, help="Crop region width")
    parser.add_argument("--crop-height", type=int, default=None, help="Crop region height")
    parser.add_argument("--stats", action="store_true", help="Show periodic frame/packet statistics")
    parser.add_argument("--window", type=str, default=None, help="Capture window by title (partial match, e.g. 'AS1000_PFD')")
    parser.add_argument("--list-windows", action="store_true", help="List all visible windows and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_windows:
        ScreenshotSerialSender.list_windows()
        sys.exit(0)

    if not args.port:
        print("Error: --port is required (unless using --list-windows)")
        sys.exit(1)

    crop = None
    if args.crop_width and args.crop_height:
        crop = (args.crop_x or 0, args.crop_y or 0, args.crop_width, args.crop_height)

    sender = ScreenshotSerialSender(
        serial_port=args.port,
        baud_rate=args.baud,
        monitor_index=args.monitor_index,
        prefer_largest=args.prefer_largest,
        target_fps=args.target_fps,
        threshold=args.threshold,
        full_frame=args.full_frame,
        show_stats=args.stats,
        max_updates_per_frame=args.max_updates_per_frame,
        rotate_deg=args.rotate,
        crop=crop,
        window_title=args.window,
    )
    sender.run()


if __name__ == "__main__":
    main()
