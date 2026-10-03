"""Nối thật với vi_fc.FunctionCaller (dự án 4), thay model bằng backend trả chuỗi cố định.

Kiểm tra cả đường: output thô kiểu Qwen3 -> parse -> guard an toàn -> xe giả lập -> câu nói.
Bỏ qua nếu chưa cài vi_fc.
"""

import asyncio

import pytest

vi_fc = pytest.importorskip("vi_fc.inference")

from copilot.bus import EventBus  # noqa: E402
from copilot.copilot import Copilot  # noqa: E402
from copilot.vehicle import VehicleSim  # noqa: E402


class CannedBackend:
    def __init__(self, text):
        self.text = text

    def generate(self, messages, tools, max_tokens, temperature):
        return vi_fc.GenResult(self.text)


def _run(raw: str, speed: float = 40):
    said = []
    brain = vi_fc.FunctionCaller(CannedBackend(raw))
    cp = Copilot(EventBus(), VehicleSim(speed_kmh=speed), brain, None, say=said.append)
    asyncio.run(cp.on_speech("lệnh bất kỳ"))
    return cp, said


def test_tool_call_reaches_vehicle():
    cp, said = _run('<tool_call>\n{"name": "set_climate", "arguments": {"temperature": 22}}\n</tool_call>')
    assert cp.vehicle.climate["temperature"] == 22
    assert said == ["Đã chỉnh điều hoà tất cả 22 độ."]


def test_two_calls_in_one_turn():
    raw = ('<tool_call>{"name": "set_climate", "arguments": {"temperature": 20}}</tool_call>\n'
           '<tool_call>{"name": "play_music", "arguments": {"query": "nhạc Trịnh"}}</tool_call>')
    cp, said = _run(raw)
    assert cp.vehicle.media == "nhạc Trịnh" and "Đang phát" in said[0]


def test_guard_blocks_unsafe_call_even_if_model_complies():
    cp, said = _run('<tool_call>{"name": "unlock_doors", "arguments": {}}</tool_call>', speed=40)
    assert cp.vehicle.doors_locked  # model muốn mở nhưng guard chặn
    assert "không an toàn" in said[0]


def test_unlock_allowed_when_parked():
    cp, _ = _run('<tool_call>{"name": "unlock_doors", "arguments": {}}</tool_call>', speed=0)
    assert not cp.vehicle.doors_locked
