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


def _id(h) -> str | None:
    return h.id if hasattr(h, "id") else h.get("id")


def _expanded_from(h) -> str | None:
    return h.expanded_from if hasattr(h, "expanded_from") else h.get("expanded_from")


def _citation(h) -> str:
    return h.citation if hasattr(h, "citation") else h.get("citation", "")


def sign_warning(sign_name: str, vehicle: str, hits: list, speak_fine: bool = True) -> str:
    """hits: kết quả LawRAG.lookup_sign (đã đúng loại xe). Lấy mức phạt ở đoạn đầu tiên có mức phạt.

    Số điểm bị trừ chỉ lấy từ đoạn được kéo theo do tham chiếu tới CHÍNH đoạn có mức phạt đó;
    lấy từ đoạn bất kỳ thì có thể đọc nhầm số điểm của hành vi khác.
    speak_fine=False cho biển mà mức phạt phụ thuộc tình huống (vd. tốc độ tối đa: quá 5 hay quá 35 km/h).
    """
    msg = f"Phía trước có biển {sign_name.lower()}."
    if not speak_fine:
        return msg
    main = next((h for h in hits if parse_fine(_text(h)) and not _expanded_from(h)), None)
    if main is None:
        return msg
    return msg + _fine_phrase(main, hits, vehicle)


def _fine_phrase(main, hits: list, vehicle: str) -> str:
    lo, hi = parse_fine(_text(main))
    msg = f" {vehicle.capitalize()} vi phạm bị phạt {speak_money(lo)} đến {speak_money(hi)} đồng"
    pts = next((p for h in hits if _expanded_from(h) and _expanded_from(h) == _id(main)
                for p in [parse_points(_text(h))] if p), None)
    return msg + (f", trừ {pts} điểm bằng lái." if pts else ".")


def law_brief(hits: list, vehicle: str) -> str | None:
    """Trả lời câu hỏi luật khi không có LLM (offline): đọc mức phạt của đoạn luật hạng đầu có mức phạt.

    Không tóm tắt được hành vi như LLM, nên luôn kèm trích dẫn để tài xế biết đang nghe điều nào.
    """
    main = next((h for h in hits if parse_fine(_text(h)) and not _expanded_from(h)), None)
    if main is None:
        return None
    return f"Theo {_citation(main)}:" + _fine_phrase(main, hits, vehicle)
