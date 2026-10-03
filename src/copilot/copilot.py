"""Bộ điều phối: nhận sự kiện từ bus (lời nói, biển báo, buồn ngủ) -> quyết định nói gì / làm gì.

Nguyên tắc:
- Mỗi lúc chỉ nói một câu; việc gấp nói trước (buồn ngủ > biển báo > trả lời lệnh).
- Cảnh báo chủ động có cooldown, không thì cứ thấy lại biển cấm đỗ là nói lại -> tài xế tắt luôn trợ lý.
- Lệnh giọng nói đi qua model function-calling (vi-function-calling-slm), chạy trong thread riêng vì
  llama.cpp chặn; event loop vẫn nhận sự kiện camera trong lúc model đang nghĩ.
- Cảnh báo biển báo KHÔNG qua LLM: lấy mức phạt bằng regex từ đoạn luật (nhanh, không bịa số).
"""

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .bus import Event, EventBus
from .fines import sign_warning


@dataclass
class Turn:
    t: float
    source: str  # speech | sign | drowsy
    input: str
    output: str
    ms: dict = field(default_factory=dict)


class Copilot:
    def __init__(self, bus: EventBus, vehicle, brain=None, law=None, say: Callable | None = None,
                 vehicle_kind: str = "ô tô", restorer: Callable[[str], str] | None = None,
                 sign_cooldown_s: float = 90, drowsy_cooldown_s: float = 60,
                 clock: Callable[[], float] = time.monotonic):
        self.bus, self.vehicle, self.brain, self.law = bus, vehicle, brain, law
        self.say_fn = say or (lambda text: print(f"[trợ lý] {text}"))
        self.vehicle_kind = vehicle_kind
        self.restorer = restorer
        self.sign_cd, self.drowsy_cd = sign_cooldown_s, drowsy_cooldown_s
        self.clock = clock
        self._last_sign: dict[str, float] = {}
        self._last_drowsy = -1e9
        self.turns: list[Turn] = []
        if law is not None:
            vehicle.law = law

    async def say(self, text: str):
        r = self.say_fn(text)
        if inspect.isawaitable(r):
            await r

    # ---------- xử lý từng loại sự kiện ----------
    async def on_speech(self, text: str) -> Turn:
        t0 = time.perf_counter()
        ms = {}
        if self.restorer is not None:
            from vnlaw_rag.text import has_diacritics

            if not has_diacritics(text):
                text = self.restorer(text)
                ms["restore"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        action = await asyncio.to_thread(self.brain.decide, text, self.vehicle.state())
        ms["brain"] = (time.perf_counter() - t1) * 1e3
        if action.kind == "call" and action.calls:
            t2 = time.perf_counter()
            results = [await asyncio.to_thread(self.vehicle.execute, c.name, c.arguments) for c in action.calls]
            ms["tools"] = (time.perf_counter() - t2) * 1e3
            out = " ".join(results)
        else:
            out = action.text or "Xin lỗi, mình chưa hiểu ý anh."
        await self.say(out)
        return self._log("speech", text, out, ms)

    async def on_sign(self, ev) -> Turn | None:
        now = self.clock()
        if now - self._last_sign.get(ev.code, -1e9) < self.sign_cd or self.law is None:
            return None
        info = await asyncio.to_thread(self.law.lookup_sign, ev.code, self.vehicle_kind, 3)
        if info is None:  # biển không có trong bảng tra (biển chỉ dẫn...) -> im lặng
            return None
        self._last_sign[ev.code] = now
        out = sign_warning(info.name, self.vehicle_kind, info.hits)
        await self.say(out)
        return self._log("sign", ev.code, out)

    async def on_drowsy(self, st) -> Turn | None:
        now = self.clock()
        cd = self.drowsy_cd / 3 if st.level == "alarm" else self.drowsy_cd
        if st.level not in ("warn", "alarm") or now - self._last_drowsy < cd:
            return None
        self._last_drowsy = now
        if st.level == "alarm":
            out = "Anh ơi! Anh đang nhắm mắt khi lái. Hãy giảm tốc và tấp vào lề nghỉ ngay."
        else:
            stop = self.vehicle.execute("find_charging_station", {"max_distance_km": 10})
            out = f"Anh có vẻ mệt rồi, nên dừng nghỉ một chút. {stop}"
        await self.say(out)
        return self._log("drowsy", st.reason, out)

    def _log(self, source, inp, out, ms=None) -> Turn:
        turn = Turn(self.clock(), source, inp, out, ms or {})
        self.turns.append(turn)
        return turn

    # ---------- vòng lặp chính ----------
    async def run(self, stop: asyncio.Event | None = None):
        q = self.bus.subscribe("speech", "sign", "drowsy")
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                ev: Event = await asyncio.wait_for(q.get(), timeout=0.2)
            except TimeoutError:
                continue
            if ev.topic == "speech":
                await self.on_speech(ev.data)
            elif ev.topic == "sign":
                await self.on_sign(ev.data)
            elif ev.topic == "drowsy":
                await self.on_drowsy(ev.data)
