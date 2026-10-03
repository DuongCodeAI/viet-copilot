"""Event bus bằng asyncio: các module (nghe, nhìn đường, nhìn tài xế) chạy độc lập, chỉ nói chuyện qua sự kiện.

Vì sao không gọi hàm trực tiếp: camera chạy ~10-15 fps, STT chạy theo câu nói, LLM mất 1-2s.
Nếu camera phải chờ LLM nói xong thì bỏ lỡ biển báo. Mỗi nguồn là một task, đẩy sự kiện vào
bus; Copilot tiêu thụ và quyết định nói gì theo độ ưu tiên.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# độ ưu tiên khi nhiều thứ cùng muốn nói: số nhỏ = gấp hơn
PRIORITY = {"drowsy": 0, "sign": 1, "speech": 2, "system": 3}


@dataclass(order=True)
class Event:
    priority: int
    t: float = field(compare=False)
    topic: str = field(compare=False)
    data: Any = field(compare=False, default=None)


class EventBus:
    def __init__(self):
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self.log: list[Event] = []

    def subscribe(self, *topics: str, maxsize: int = 100) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.PriorityQueue(maxsize=maxsize)
        for t in topics:
            self._subs[t].append(q)
        return q

    def publish(self, topic: str, data: Any = None) -> Event:
        ev = Event(PRIORITY.get(topic, 5), time.monotonic(), topic, data)
        self.log.append(ev)
        for q in self._subs.get(topic, []):
            if q.full():
                q.get_nowait()  # bỏ sự kiện cũ nhất thay vì chặn producer (camera không được chờ)
            q.put_nowait(ev)
        return ev
