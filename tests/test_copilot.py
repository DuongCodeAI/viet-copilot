"""Test luồng điều phối với brain / luật giả (không cần model)."""

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

from copilot.bus import EventBus
from copilot.copilot import Copilot
from copilot.drowsiness import DrowsyState
from copilot.vehicle import VehicleSim


@dataclass
class FakeAction:
    kind: str
    calls: list = field(default_factory=list)
    text: str = ""


class FakeBrain:
    def decide(self, text, state):
        if "điều hoà" in text:
            return FakeAction("call", [SimpleNamespace(name="set_climate", arguments={"temperature": 22})])
        if "mở cửa" in text:
            return FakeAction("refuse", text="Xe đang chạy, mình không mở cửa được ạ.")
        return FakeAction("ask", text="Anh muốn đi đâu ạ?")


class FakeLaw:
    def lookup_sign(self, code, vehicle=None, k=3):
        if code != "P.131a":
            return None
        hit = {"text": "Điều 6\n3. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với ...\ne) đỗ xe nơi có biển"}
        return SimpleNamespace(name="Cấm đỗ xe", hits=[hit])


class VehicleNoVifc(VehicleSim):
    def state(self):
        return {"speed_kmh": self.speed_kmh}


def _copilot(clock):
    said = []
    cp = Copilot(EventBus(), VehicleNoVifc(speed_kmh=40), FakeBrain(), FakeLaw(), say=said.append, clock=clock)
    return cp, said


def test_speech_calls_tool_and_confirms():
    cp, said = _copilot(lambda: 0.0)
    asyncio.run(cp.on_speech("bật điều hoà 22 độ"))
    assert cp.vehicle.climate["temperature"] == 22
    assert said == ["Đã chỉnh điều hoà tất cả 22 độ."]


def test_refuse_is_spoken():
    cp, said = _copilot(lambda: 0.0)
    asyncio.run(cp.on_speech("mở cửa ra"))
    assert "không mở cửa" in said[0]


def test_sign_warning_has_fine_and_cooldown():
    t = [0.0]
    cp, said = _copilot(lambda: t[0])
    ev = SimpleNamespace(code="P.131a")
    asyncio.run(cp.on_sign(ev))
    t[0] = 30
    asyncio.run(cp.on_sign(ev))  # trong cooldown -> im lặng
    t[0] = 200
    asyncio.run(cp.on_sign(ev))
    assert len(said) == 2
    assert "800 nghìn đến 1 triệu" in said[0]


def test_unknown_sign_silent():
    cp, said = _copilot(lambda: 0.0)
    asyncio.run(cp.on_sign(SimpleNamespace(code="I.408")))
    assert said == []


def test_drowsy_alarm_and_warn():
    t = [0.0]
    cp, said = _copilot(lambda: t[0])
    asyncio.run(cp.on_drowsy(DrowsyState("warn", "ngáp 3 lần")))
    assert "mệt" in said[-1] and "trạm" in said[-1]
    t[0] = 25
    asyncio.run(cp.on_drowsy(DrowsyState("alarm", "nhắm mắt 2s")))
    assert "nhắm mắt" in said[-1]


def test_run_loop_priority():
    async def go():
        cp, said = _copilot(lambda: 0.0)
        stop = asyncio.Event()
        task = asyncio.create_task(cp.run(stop))
        await asyncio.sleep(0)
        cp.bus.publish("speech", "đi đâu đó")
        cp.bus.publish("drowsy", DrowsyState("alarm", "nhắm mắt"))
        await asyncio.sleep(0.3)
        stop.set()
        await task
        return said

    said = asyncio.run(go())
    assert "nhắm mắt" in said[0]  # buồn ngủ được nói trước dù đến sau


def test_law_lookup_uses_raw_text_not_restored():
    class LawBrain:
        def decide(self, text, state):
            return FakeAction("call", [SimpleNamespace(name="lookup_traffic_law", arguments={"question": text})])

    asked = []

    class Vehicle(VehicleNoVifc):
        def execute(self, name, args):
            asked.append(args["question"])
            return "ok"

    cp = Copilot(EventBus(), Vehicle(speed_kmh=40), LawBrain(), say=lambda t: None,
                 restorer=lambda t: "vượt đến do bị phát bao nhiêu")
    asyncio.run(cp.on_speech("vuot den do bi phat bao nhieu"))
    assert asked == ["vuot den do bi phat bao nhieu"]


def test_sign_tool_uses_vehicle_kind_and_speak_fine():
    seen = []

    class Law:
        def lookup_sign(self, code, vehicle=None, k=3):
            seen.append(vehicle)
            hit = {"text": "Điều 6\n3. Phạt tiền từ 3.000.000 đồng đến 5.000.000 đồng đối với ..."}
            return SimpleNamespace(name="tốc độ tối đa cho phép", hits=[hit], speak_fine=False)

    v = VehicleNoVifc(speed_kmh=40)
    Copilot(EventBus(), v, FakeBrain(), Law(), say=lambda t: None, vehicle_kind="xe máy")
    out = v._t_lookup_sign("P.127")
    assert seen == ["xe máy"]
    assert "triệu" not in out
