"""Biến đoạn luật thành câu nói ngắn cho tài xế, KHÔNG gọi LLM.

Cảnh báo chủ động khi thấy biển báo phải nhanh (<200ms) và chắc chắn đúng số tiền,
nên lấy thẳng mức phạt bằng regex từ đoạn luật mà retriever trả về.
"Phạt tiền từ 800.000 đồng đến 1.000.000 đồng" -> "phạt 800 nghìn đến 1 triệu đồng".
"""

import re

FINE_RE = re.compile(r"Phạt tiền từ\s+([\d.]+)\s*đồng\s+đến\s+([\d.]+)\s*đồng", re.IGNORECASE)
POINTS_RE = re.compile(r"trừ điểm giấy phép lái xe\s+(\d+)\s*điểm", re.IGNORECASE)


def speak_money(v: int) -> str:
    if v >= 1_000_000:
        m = v / 1_000_000
        return f"{m:g} triệu".replace(".", ",")
    if v >= 1_000:
        return f"{v // 1000} nghìn"
    return f"{v} đồng"


def parse_fine(text: str) -> tuple[int, int] | None:
    m = FINE_RE.search(text)
    if not m:
        return None
    return int(m.group(1).replace(".", "")), int(m.group(2).replace(".", ""))


def parse_points(text: str) -> int | None:
    m = POINTS_RE.search(text)
    return int(m.group(1)) if m else None


def _text(h) -> str:
    return h.chunk["text"] if hasattr(h, "chunk") else h["text"]


def sign_warning(sign_name: str, vehicle: str, hits: list) -> str:
    """hits: kết quả LawRAG.lookup_sign (đã đúng loại xe). Lấy mức phạt ở đoạn đầu tiên có mức phạt."""
    msg = f"Phía trước có biển {sign_name.lower()}."
    fine = next((f for f in (parse_fine(_text(h)) for h in hits) if f), None)
    if fine is None:
        return msg
    lo, hi = fine
    msg += f" {vehicle.capitalize()} vi phạm bị phạt {speak_money(lo)} đến {speak_money(hi)} đồng"
    pts = next((p for p in (parse_points(_text(h)) for h in hits) if p), None)
    return msg + (f", trừ {pts} điểm bằng lái." if pts else ".")
