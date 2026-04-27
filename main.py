"""
Watchman — Real-time CCTV Human Detection
Stream source: IP Webcam (Android) MJPEG stream
Detection: YOLOv8n (person class only)
"""

import cv2
import time
import threading
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
STREAM_URL            = "http://192.168.137.170:8080/video"  # IP Webcam MJPEG
CONF_THRESHOLD        = 0.45           # detection confidence (0.4–0.5 sweet spot)
INFERENCE_WIDTH       = 640            # resize width for YOLO inference
DISPLAY_WIDTH         = 960            # resize width for display (lower = faster)
SKIP_FRAMES           = 2              # run YOLO every N frames (1 = every frame)
NOTIFICATION_COOLDOWN = 10             # seconds between desktop notifications
MAX_RECONNECT_TRIES   = 5             # reconnect attempts before giving up
RECONNECT_DELAY       = 3              # seconds between reconnect attempts
# ───────────────────────────────────────────────────────────────────────────────


# ─── NOTIFICATION ──────────────────────────────────────────────────────────────
# Try winotify (Windows 10/11 native), fall back to plyer, then to console-only
_notifier = None

try:
    from winotify import Notification as WinNotification
    _notifier = "winotify"
except ImportError:
    try:
        from plyer import notification as plyer_notification
        _notifier = "plyer"
    except ImportError:
        _notifier = None

if _notifier:
    print(f"[INIT] Desktop notifications via: {_notifier}")
else:
    print("[INIT] No notification library found — console alerts only")
    print("       Install one: pip install winotify  OR  pip install plyer")


def send_notification(count):
    """Send a desktop notification. Never raises."""
    try:
        if _notifier == "winotify":
            toast = WinNotification(
                app_id="Watchman",
                title="⚠️ Watchman Alert",
                msg=f"{count} person(s) detected on camera!",
                duration="short",
            )
            toast.show()
        elif _notifier == "plyer":
            plyer_notification.notify(
                title="⚠️ Watchman Alert",
                message=f"{count} person(s) detected on camera!",
                timeout=4,
                app_name="Watchman",
            )
    except Exception:
        pass  # never let notification errors crash the pipeline


# ─── LATEST-FRAME GRABBER ─────────────────────────────────────────────────────
# Why threading here: OpenCV's internal buffer queues MJPEG frames. While the
# main thread runs YOLO (~30-50ms), 1-2 frames pile up. Without draining, each
# cap.read() returns the NEXT buffered frame, not the LATEST — causing the
# display to fall seconds behind reality. This tiny thread continuously grabs
# (and discards) frames so the main thread always gets the freshest one.

class LatestFrame:
    """
    Continuously grabs frames in a background thread.
    The main thread always gets the most recent frame via read().
    """
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        """Return the latest frame. Non-blocking."""
        with self._lock:
            if self._frame is not None:
                return self._ret, self._frame.copy()
            return False, None

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# ─── STREAM CONNECTION ─────────────────────────────────────────────────────────
def open_stream(url):
    """
    Open an MJPEG stream via LatestFrame grabber.
    Returns (grabber, success).
    """
    grabber = LatestFrame(url)

    if not grabber.is_opened():
        grabber.release()
        return None, False

    # Wait up to 3s for the first valid frame
    deadline = time.time() + 3
    while time.time() < deadline:
        ret, frame = grabber.read()
        if ret and frame is not None:
            print(f"[STREAM] Connected. Frame: {frame.shape[1]}x{frame.shape[0]}")
            return grabber, True
        time.sleep(0.1)

    grabber.release()
    return None, False


def reconnect(url, max_tries, delay):
    """Try to reconnect to the stream with retries."""
    for attempt in range(1, max_tries + 1):
        print(f"[STREAM] Reconnect attempt {attempt}/{max_tries}...")
        grabber, ok = open_stream(url)
        if ok:
            print("[STREAM] Reconnected successfully.")
            return grabber
        time.sleep(delay)
    return None


# ─── DRAWING ───────────────────────────────────────────────────────────────────
def draw_hud(frame, detections, fps):
    """Draw bounding boxes, HUD bar with FPS / count / status."""
    h, w = frame.shape[:2]

    # Bounding boxes
    for (x1, y1, x2, y2, conf) in detections:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background + text
        label = f"Person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 160, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Top HUD bar (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Human count
    count = len(detections)
    count_color = (0, 255, 80) if count > 0 else (140, 140, 140)
    cv2.putText(frame, f"Humans: {count}", (150, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2)

    # Status
    if count > 0:
        status, status_color = "HUMAN DETECTED", (0, 80, 255)
    else:
        status, status_color = "MONITORING", (200, 200, 200)
    cv2.putText(frame, status, (340, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Bottom alert banner when humans detected
    if count > 0:
        banner_text = f"ALERT: {count} PERSON(S) DETECTED"
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx = (w - bw) // 2
        cv2.rectangle(frame, (0, h - bh - 20), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, banner_text, (bx, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame


# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────
def main():
    # Load model
    print("[INIT] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("[INIT] Model loaded.")

    # Connect to stream
    print(f"[STREAM] Connecting to: {STREAM_URL}")
    grabber, ok = open_stream(STREAM_URL)
    if not ok:
        print("[ERROR] Cannot open stream. Make sure IP Webcam is running.")
        print(f"        URL: {STREAM_URL}")
        return

    # State
    fps_timer  = time.time()
    fps_count  = 0
    fps        = 0.0
    frame_idx  = 0
    last_notif = 0.0
    last_dets  = []          # persist detections across skipped frames
    fail_count = 0           # consecutive read failures

    print("[RUNNING] Press 'q' to quit.\n")

    while True:
        ret, frame = grabber.read()

        # ── Handle read failure ──────────────────────────────────────────────
        if not ret or frame is None:
            fail_count += 1
            if fail_count > 60:  # ~2 seconds of no frames
                print("[STREAM] Lost connection.")
                grabber.release()
                grabber = reconnect(STREAM_URL, MAX_RECONNECT_TRIES, RECONNECT_DELAY)
                if grabber is None:
                    print("[ERROR] Could not reconnect. Exiting.")
                    break
                fail_count = 0
                last_dets = []
            time.sleep(0.005)
            continue
        fail_count = 0

        orig_h, orig_w = frame.shape[:2]

        # ── Resize for display (work at display resolution) ──────────────────
        disp_scale = DISPLAY_WIDTH / orig_w
        disp_h = int(orig_h * disp_scale)
        frame = cv2.resize(frame, (DISPLAY_WIDTH, disp_h))

        # ── YOLO inference (every N frames) ──────────────────────────────────
        if frame_idx % SKIP_FRAMES == 0:
            # Resize from display frame to inference size
            inf_scale = INFERENCE_WIDTH / DISPLAY_WIDTH
            small = cv2.resize(frame, (INFERENCE_WIDTH, int(disp_h * inf_scale)))

            results = model(small, conf=CONF_THRESHOLD, classes=[0], verbose=False)

            # Map detections back to display resolution
            inv_inf_scale = 1.0 / inf_scale
            dets = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append((
                        int(x1 * inv_inf_scale), int(y1 * inv_inf_scale),
                        int(x2 * inv_inf_scale), int(y2 * inv_inf_scale),
                        conf,
                    ))
            last_dets = dets

            # Notification with cooldown
            if dets:
                now = time.time()
                if now - last_notif > NOTIFICATION_COOLDOWN:
                    send_notification(len(dets))
                    print(f"[ALERT] {len(dets)} person(s) detected — {time.strftime('%H:%M:%S')}")
                    last_notif = now

        # ── FPS calculation ──────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_timer = time.time()
            fps_count = 0

        # ── Display ──────────────────────────────────────────────────────────
        display = draw_hud(frame, last_dets, fps)
        cv2.imshow("Watchman — CCTV Human Detection", display)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if grabber is not None:
        grabber.release()
    cv2.destroyAllWindows()
    print("[EXIT] Watchman stopped.")


if __name__ == "__main__":
    main()