"""Phát hiện buồn ngủ từ camera trong cabin (hoặc webcam laptop khi thử ở bàn).

Đặc trưng từ 478 landmark của MediaPipe Face Landmarker:
- EAR (eye aspect ratio, Soukupová & Čech 2016): (|p2-p6| + |p3-p5|) / (2|p1-p4|). Mắt nhắm -> EAR tụt.
- MAR (mouth aspect ratio): độ mở miệng / độ rộng miệng. Ngáp -> MAR cao kéo dài.
- PERCLOS: % thời gian mắt nhắm trong cửa sổ 60s - chỉ số mỏi mệt được dùng trong nghiên cứu lái xe.

Ngưỡng EAR cố định (0.2-0.25 như nhiều tutorial) không hợp mọi người: mắt một mí / đeo kính có EAR
lúc mở đã thấp. Nên hiệu chỉnh: 10 giây đầu đo EAR lúc mở mắt của chính người đó, ngưỡng = 75% mức này.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

# chỉ số landmark trên face mesh (p1..p6 theo thứ tự của công thức EAR)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]  # khoé trái, khoé phải, môi trên, môi dưới


def ear(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-6))


def mar(pts: np.ndarray) -> float:
    left, right, top, bottom = pts
    return float(np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6))


def features(landmarks: np.ndarray) -> tuple[float, float]:
    """landmarks (478, 2|3) -> (EAR trung bình 2 mắt, MAR)."""
    lm = landmarks[:, :2]
    return (ear(lm[LEFT_EYE]) + ear(lm[RIGHT_EYE])) / 2, mar(lm[MOUTH])


@dataclass
class DrowsyState:
    level: str  # "ok" | "calibrating" | "warn" | "alarm" | "no_face"
    reason: str
    ear: float | None = None
    perclos: float | None = None


class DrowsinessMonitor:
    def __init__(self, fps: float = 15, calib_sec: float = 10, perclos_win_sec: float = 60,
                 closed_ratio: float = 0.75, eyes_closed_alarm_sec: float = 1.5, yawn_mar: float = 0.6,
                 yawn_sec: float = 1.5, perclos_warn: float = 0.15, perclos_alarm: float = 0.3):
        self.fps = fps
        self.calib_n = int(calib_sec * fps)
        self.calib: list[float] = []
        self.thr: float | None = None
        self.closed_ratio = closed_ratio
        self.window = deque(maxlen=int(perclos_win_sec * fps))
        self.closed_run = 0
        self.yawn_run = 0
        self.yawns: deque = deque(maxlen=10)  # thời điểm các lần ngáp
        self.alarm_frames = int(eyes_closed_alarm_sec * fps)
        self.yawn_mar, self.yawn_frames = yawn_mar, int(yawn_sec * fps)
        self.perclos_warn, self.perclos_alarm = perclos_warn, perclos_alarm
        self.frame = 0

    def update(self, landmarks: np.ndarray | None) -> DrowsyState:
        self.frame += 1
        if landmarks is None:
            return DrowsyState("no_face", "không thấy mặt")
        e, m = features(landmarks)
        if self.thr is None:
            self.calib.append(e)
            if len(self.calib) < self.calib_n:
                return DrowsyState("calibrating", "đang hiệu chỉnh", e)
            # trung vị thay vì trung bình: vài lần chớp mắt lúc hiệu chỉnh không kéo ngưỡng xuống
            self.thr = float(np.median(self.calib)) * self.closed_ratio
        closed = e < self.thr
        self.window.append(closed)
        self.closed_run = self.closed_run + 1 if closed else 0
        if m > self.yawn_mar:
            self.yawn_run += 1
            if self.yawn_run == self.yawn_frames:
                self.yawns.append(self.frame)
        else:
            self.yawn_run = 0
        perclos = sum(self.window) / len(self.window)
        recent_yawns = sum(1 for f in self.yawns if self.frame - f < 120 * self.fps)

        if self.closed_run >= self.alarm_frames:
            return DrowsyState("alarm", f"nhắm mắt {self.closed_run / self.fps:.1f}s", e, perclos)
        full = len(self.window) >= self.window.maxlen // 2
        if full and perclos >= self.perclos_alarm:
            return DrowsyState("alarm", f"PERCLOS {perclos:.0%}", e, perclos)
        if (full and perclos >= self.perclos_warn) or recent_yawns >= 3:
            why = f"PERCLOS {perclos:.0%}" if perclos >= self.perclos_warn else f"ngáp {recent_yawns} lần/2 phút"
            return DrowsyState("warn", why, e, perclos)
        return DrowsyState("ok", "", e, perclos)


class FaceLandmarks:
    """Bọc MediaPipe Face Landmarker (model .task ~3.8MB). Import muộn để test không cần mediapipe."""

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        self.det = vision.FaceLandmarker.create_from_options(opts)
        self.mp = mp
        self.t_ms = 0

    def __call__(self, frame_bgr: np.ndarray, dt_ms: int = 66) -> np.ndarray | None:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        self.t_ms += dt_ms
        res = self.det.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.t_ms)
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
