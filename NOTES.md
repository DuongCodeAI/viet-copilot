# Ghi chú quyết định

## Kiến trúc
- Mỗi nguồn (dashcam, cabin, mic, kịch bản) là một coroutine, model chạy trong `asyncio.to_thread` vì
  onnxruntime / llama.cpp / faster-whisper đều chặn. Event loop chỉ điều phối.
- `PriorityQueue` theo chủ đề: buồn ngủ (0) > biển báo (1) > lệnh (2). Trong test `test_run_loop_priority`,
  sự kiện buồn ngủ đến sau vẫn được nói trước.
- Hàng đợi đầy -> bỏ sự kiện cũ nhất: camera không bao giờ phải chờ.
- Replay đọc video theo thời gian thực, máy chậm thì bỏ frame (giống camera thật), không chạy chậm lại.

## Cảnh báo biển báo
- Không dùng LLM: cần nhanh và không được bịa số. Regex "Phạt tiền từ X đồng đến Y đồng" trên đoạn luật
  hàng đầu (đã lọc đúng loại xe ở repo luật).
- Lỗi đã gặp khi chạy với dữ liệu thật: lấy "trừ N điểm" từ đoạn bất kỳ trong kết quả → có thể là điểm trừ
  của hành vi khác. Sửa: chỉ lấy từ đoạn được kéo theo do tham chiếu tới chính đoạn có mức phạt.
- Biển tốc độ tối đa: lần đầu trợ lý nói "phạt 12-14 triệu" (mức quá >35 km/h) → gây hiểu lầm.
  Thêm cờ `speak_fine: false` trong sign_map.
- Tra luật lần đầu 1.3-2 s (reranker trên CPU) → cache theo (mã biển, loại xe) + tra trước lúc khởi động.

## Buồn ngủ
- EAR (Soukupová & Čech 2016), MAR cho ngáp, PERCLOS cửa sổ 60 s.
- Hiệu chỉnh bằng trung vị EAR 10 s đầu (trung vị để vài lần chớp mắt không kéo ngưỡng xuống).
- Mức: nhắm mắt liên tục >= 1.5 s hoặc PERCLOS >= 30% → báo động; PERCLOS >= 15% hoặc ngáp >= 3 lần / 2 phút → nhắc nghỉ.
- Chỉ phát sự kiện khi mức thay đổi, cooldown 60 s (báo động: 20 s).

## STT
- PhoWhisper-small (VinAI): WER VIVOS 6.33 theo paper. Đổi sang CTranslate2 int8 cho faster-whisper.
- beam_size 1: lệnh trong xe ngắn, beam 5 chậm gần gấp đôi.
- Fine-tune với tiếng ồn: 70% mẫu trộn ồn SNR 0-15 dB, 30% giữ sạch để không quên giọng sạch;
  20% đoạn ồn để riêng cho test (không để model "thuộc" đúng đoạn ồn).
- Ồn thật: thu tiếng gió + máy khi đi xe máy; không có thì dùng ồn tổng hợp (brown noise + ù 45 Hz).

## Việc cần làm
- Đo latency thật từng thành phần khi có model function-calling + STT (lệnh `copilot replay` ghi `logs/replay.json`).
- Quay video demo 2 phút (đã có web demo trong docs/, còn thiếu video xe chạy thật).

## 04/10 chạy thử pipeline chữ với Qwen3-1.7B Q4_K_M gốc (chưa fine-tune), laptop CPU 4 luồng
- warmup 25 s, brain p50 ~1.7 s (6 lệnh trong demo/script.yaml), thêm dấu 4 ms, TTS p50 ~0.3 s
- model gốc sai: "máy lạnh xuống 22 độ" -> chỉnh ghế phụ, "bật bài Lạc Trôi" -> quạt gió, hỏi luật -> "chưa rõ"
- guard chặn mở cửa khi xe đang chạy hoạt động đúng
