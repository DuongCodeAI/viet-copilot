import asyncio

import numpy as np

from copilot.bus import EventBus
from copilot.drowsiness import LEFT_EYE, MOUTH, RIGHT_EYE, DrowsinessMonitor, ear
from copilot.fines import parse_fine, sign_warning, speak_money


def test_bus_priority_order():
    async def run():
        bus = EventBus()
        q = bus.subscribe("speech", "drowsy", "sign")
        bus.publish("speech", "bật điều hoà")
        bus.publish("sign", "P.131a")
        bus.publish("drowsy", "alarm")
        return [(await q.get()).topic for _ in range(3)]

    assert asyncio.run(run()) == ["drowsy", "sign", "speech"]


def test_bus_drops_oldest_when_full():
    bus = EventBus()
    q = bus.subscribe("sign", maxsize=2)
    for i in range(5):
        bus.publish("sign", i)
    assert q.qsize() == 2


def _face(eye_open: float, mouth_open: float = 0.1) -> np.ndarray:
    """Landmark giả: mắt rộng 30px, cao eye_open*30; miệng rộng 50px."""
    lm = np.zeros((478, 2), np.float32)
    for eye, x0 in ((LEFT_EYE, 100), (RIGHT_EYE, 200)):
        h = eye_open * 30 / 2
        pts = [(x0, 100), (x0 + 10, 100 - h), (x0 + 20, 100 - h), (x0 + 30, 100), (x0 + 20, 100 + h), (x0 + 10, 100 + h)]
        lm[eye] = pts
    lm[MOUTH] = [(130, 200), (180, 200), (155, 200 - mouth_open * 25), (155, 200 + mouth_open * 25)]
    return lm


def test_ear_formula():
    assert abs(ear(_face(0.3)[LEFT_EYE]) - 0.3) < 1e-4


def test_calibration_then_alarm_on_long_eye_closure():
    mon = DrowsinessMonitor(fps=10, calib_sec=2, eyes_closed_alarm_sec=1.0)
    states = [mon.update(_face(0.32)).level for _ in range(20)]
    assert states[0] == "calibrating" and states[-1] == "ok"
    closed = [mon.update(_face(0.05)).level for _ in range(12)]
    assert closed[-1] == "alarm"


def test_personal_threshold_small_eyes():
    mon = DrowsinessMonitor(fps=10, calib_sec=2)
    for _ in range(20):
        mon.update(_face(0.2))
    assert mon.update(_face(0.19)).level == "ok"  # ngưỡng cố định 0.25 sẽ báo nhầm người này


def test_yawns_trigger_warning():
    mon = DrowsinessMonitor(fps=10, calib_sec=1, yawn_sec=1.0)
    for _ in range(10):
        mon.update(_face(0.3))
    last = None
    for _ in range(3):
        for _ in range(12):
            last = mon.update(_face(0.3, mouth_open=1.0))
        for _ in range(10):
            last = mon.update(_face(0.3))
    assert last.level == "warn" and "ngáp" in last.reason


def test_fines_text():
    assert parse_fine("1. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với") == (800_000, 1_000_000)
    assert speak_money(800_000) == "800 nghìn" and speak_money(1_500_000) == "1,5 triệu"
    hits = [{"text": "Điều 6...\n3. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với ...\ne) Đỗ xe nơi có biển"},
            {"text": "b) Thực hiện hành vi ... bị trừ điểm giấy phép lái xe 02 điểm"}]
    msg = sign_warning("Cấm đỗ xe", "ô tô", hits)
    assert msg == "Phía trước có biển cấm đỗ xe. Ô tô vi phạm bị phạt 800 nghìn đến 1 triệu đồng, trừ 2 điểm bằng lái."
