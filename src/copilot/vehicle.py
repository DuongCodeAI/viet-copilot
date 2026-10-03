"""Xe giả lập: nhận tool call từ model và trả câu xác nhận tiếng Việt.

Không có xe thật để nối CAN bus, nên mọi tool (điều hoà, cửa sổ, đèn...) cập nhật trạng thái trong
bộ nhớ. Hai tool tra luật thì gọi sang vn-traffic-law-rag thật. Tool "dẫn đường"/"tìm trạm sạc" trả
dữ liệu giả có cấu trúc giống API bản đồ để luồng hội thoại chạy trọn vẹn.
"""

from dataclasses import dataclass, field

DIRECTIONS = {"driver": "ghế lái", "passenger": "ghế phụ", "rear_left": "sau trái", "rear_right": "sau phải",
              "all": "tất cả"}


@dataclass
class VehicleSim:
    speed_kmh: float = 0
    gear: str = "P"
    is_night: bool = False
    battery_pct: int = 80
    range_km: int = 260
    doors_locked: bool = True
    climate: dict = field(default_factory=lambda: {"temperature": 25, "mode": "auto", "zone": "all", "fan": 3})
    windows: dict = field(default_factory=lambda: {k: 0 for k in ("driver", "passenger", "rear_left", "rear_right")})
    lights: str = "auto"
    drive_mode: str = "comfort"
    volume: int = 12
    media: str | None = None
    destination: str | None = None
    reminders: list = field(default_factory=list)
    law = None  # LawRAG, gắn từ ngoài

    def state(self):
        """VehicleState của vi_fc để đưa vào prompt (import muộn: chạy được khi chưa cài vi_fc)."""
        from vi_fc.state import VehicleState

        return VehicleState(speed_kmh=self.speed_kmh, gear=self.gear, is_night=self.is_night,
                            battery_pct=self.battery_pct, range_km=self.range_km, doors_locked=self.doors_locked)

    def execute(self, name: str, args: dict) -> str:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return f"Xe chưa hỗ trợ chức năng {name}."
        return fn(**args)

    # ---- các tool ----
    def _t_set_climate(self, temperature=None, zone="all", mode=None):
        if temperature is not None:
            self.climate["temperature"] = temperature
        if mode:
            self.climate["mode"] = mode
        self.climate["zone"] = zone
        return f"Đã chỉnh điều hoà {DIRECTIONS.get(zone, zone)} {self.climate['temperature']} độ."

    def _t_set_fan_speed(self, level):
        self.climate["fan"] = level
        return f"Quạt gió mức {level}."

    def _t_navigate_to(self, destination, avoid_tolls=False):
        self.destination = destination
        return f"Đang dẫn đường tới {destination}" + (", tránh trạm thu phí." if avoid_tolls else ".")

    def _t_find_charging_station(self, max_distance_km=10, fast_only=False):
        kind = "sạc nhanh " if fast_only else ""
        return f"Có trạm {kind}cách 3,2 km trên đường đi, còn 2 cổng trống. Anh có muốn dẫn đường tới đó không?"

    def _t_play_music(self, query):
        self.media = query
        return f"Đang phát {query}."

    def _t_set_volume(self, level):
        self.volume = level
        return f"Âm lượng {level}."

    def _t_call_contact(self, name):
        return f"Đang gọi {name}."

    def _t_read_messages(self, count=1):
        return "Không có tin nhắn mới."

    def _t_get_vehicle_status(self, field):
        return {"battery": f"Pin còn {self.battery_pct}%.", "range": f"Đi được khoảng {self.range_km} km nữa.",
                "tire_pressure": "Áp suất lốp bình thường.", "odometer": "Xe đã đi 12.480 km."}[field]

    def _t_open_window(self, position, percent=100):
        keys = list(self.windows) if position == "all" else [position]
        for k in keys:
            self.windows[k] = percent
        what = "Đóng" if percent == 0 else f"Mở {percent}%"
        return f"{what} cửa sổ {DIRECTIONS.get(position, position)}."

    def _t_lock_doors(self):
        self.doors_locked = True
        return "Đã khoá cửa."

    def _t_unlock_doors(self):
        self.doors_locked = False
        return "Đã mở khoá cửa."

    def _t_set_lights(self, mode):
        self.lights = mode
        return {"off": "Đã tắt đèn.", "auto": "Đèn tự động.", "low": "Đã bật đèn cốt.", "high": "Đã bật đèn pha."}[mode]

    def _t_set_drive_mode(self, mode):
        self.drive_mode = mode
        return f"Chế độ lái {mode}."

    def _t_set_reminder(self, text, minutes):
        self.reminders.append((minutes, text))
        return f"Sẽ nhắc anh sau {minutes} phút: {text}."

    def _t_lookup_traffic_law(self, question):
        if self.law is None:
            return "Chưa nạp dữ liệu luật."
        ans = self.law.answer(question, k=4)
        cite = f" (theo {ans.citations[0]['citation']})" if ans.citations else ""
        return ans.answer + cite

    def _t_lookup_sign(self, code):
        if self.law is None:
            return "Chưa nạp dữ liệu luật."
        info = self.law.lookup_sign(code)
        if info is None:
            return f"Mình chưa có thông tin biển {code}."
        from .fines import sign_warning

        return sign_warning(info.name, "xe", info.hits).replace("Phía trước có biển", "Biển này là")
