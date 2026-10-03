# viet-copilot

Trợ lý lái xe tiếng Việt **chạy offline trên laptop không GPU** (Ryzen 5 5625U, 16GB RAM):
nghe lệnh bằng giọng nói, nhìn biển báo qua camera hành trình, theo dõi tài xế buồn ngủ, và tra luật giao thông.

```
[camera đường] ─► vn-dashcam-vision ─► "sign" ──┐
[camera cabin] ─► MediaPipe + EAR/PERCLOS ─► "drowsy" ─┤
[micro] ─► PhoWhisper (int8) ─► "speech" ──────────────┤
                                                        ▼
                                  EventBus (ưu tiên: buồn ngủ > biển báo > lệnh)
                                                        ▼
                                                    Copilot
              ┌───────────────────────┬─────────────────┴──────────────────┐
     lệnh giọng nói              biển báo                           buồn ngủ
  vi-function-calling-slm     vn-traffic-law-rag              gợi ý dừng nghỉ /
  (Qwen3-1.7B GGUF) ─► tool   lookup_sign -> mức phạt          cảnh báo khẩn
  (điều hoà, cửa, đèn, ...)   (regex, không qua LLM)
                                                        ▼
                                                 Piper TTS (tiếng Việt)
```

Ghép 4 repo thành phần:

| repo | vai trò |
|---|---|
| [vn-traffic-law-rag](https://github.com/DuongCodeAI/vn-traffic-law-rag) | tra luật, biển báo -> điều khoản xử phạt |
| [vn-dashcam-vision](https://github.com/DuongCodeAI/vn-dashcam-vision) | nhận diện biển báo, phát sự kiện sau khi nhiều frame đồng ý |
| [vi-diacritics-transformer](https://github.com/DuongCodeAI/vi-diacritics-transformer) | thêm dấu cho lệnh gõ không dấu |
| [vi-function-calling-slm](https://github.com/DuongCodeAI/vi-function-calling-slm) | "bộ não" gọi tool, model 1.7B chạy offline |

## Một đoạn demo (chế độ replay)

```
[  4.0s] tài xế: nóng quá, cho máy lạnh xuống 22 độ đi
[  5.3s] trợ lý: Đã chỉnh điều hoà tất cả 22 độ.
[ 12.4s] trợ lý: Phía trước có biển cấm đi ngược chiều. Ô tô vi phạm bị phạt 18 triệu đến 20 triệu đồng, trừ 4 điểm bằng lái.
[ 28.0s] tài xế: mở cửa sau bên trái giúp mình
[ 29.1s] trợ lý: Xe đang chạy 40 km/h, mình không mở cửa được ạ.
[ 47.9s] trợ lý: Anh có vẻ mệt rồi, nên dừng nghỉ một chút. Có trạm cách 3,2 km trên đường đi...
```

> Đoạn trên minh hoạ định dạng log. Các câu cảnh báo biển báo là output thật (dữ liệu luật thật, đo trên laptop);
> phần lệnh giọng nói chờ model function-calling train xong.

## Latency trên laptop (CPU, 4 luồng)

| thành phần | p50 | ghi chú |
|---|---|---|
| tra luật cho biển báo (lần đầu) | ~1.3-2 s | hybrid retrieval + rerank; **2 ms** khi đã cache, tra trước lúc khởi động |
| biển báo -> câu cảnh báo | ~2 ms | regex lấy mức phạt, không gọi LLM |
| STT PhoWhisper-small int8 | chưa đo | |
| function calling Qwen3-1.7B Q4 | chưa đo | |
| TTS Piper | chưa đo | |
| mục tiêu lệnh giọng nói end-to-end | < 1.5 s | |

## Quyết định thiết kế

- **Event bus bất đồng bộ**: camera ~10-15 fps không được chờ LLM 1-2 s. Mỗi nguồn là một task; hàng đợi đầy
  thì bỏ sự kiện cũ thay vì chặn camera.
- **Ưu tiên + cooldown**: buồn ngủ nói trước; mỗi biển chỉ nhắc lại sau 90 s, không thì tài xế tắt trợ lý.
- **Cảnh báo biển báo không qua LLM**: mức phạt lấy bằng regex từ đúng đoạn luật (đúng loại xe); số điểm trừ chỉ
  lấy từ khoản tham chiếu tới chính hành vi đó. Biển tốc độ không đọc mức phạt (phụ thuộc quá bao nhiêu km/h).
- **Hai lớp an toàn cho lệnh nguy hiểm**: model được train để từ chối, và luật cứng (`safety.py` bên repo
  function-calling) chặn lại lần nữa nếu model vẫn gọi tool.
- **Buồn ngủ hiệu chỉnh theo người**: 10 s đầu đo EAR lúc mở mắt của chính tài xế, ngưỡng nhắm = 75%;
  ngưỡng cố định 0.25 báo nhầm người mắt một mí.
- **STT chịu ồn**: fine-tune PhoWhisper-small với tiếng ồn xe trộn ngẫu nhiên SNR 0-15 dB (notebook 01).

## Chạy

```bash
pip install -e ".[parts,live]"
python scripts/download_models.py           # ~1.6GB vào models/
copilot chat                                # gõ lệnh, không cần mic/camera
copilot replay --dashcam drive.mp4 --cabin face.mp4 --script demo/script.yaml --speak
```

Cấu hình đường dẫn trong `configs/default.yaml`. Thành phần nào thiếu model thì tự tắt và báo.

## Hạn chế

- Xe là giả lập (`vehicle.py`); tool dẫn đường / trạm sạc trả dữ liệu mẫu.
- VIVOS (dữ liệu fine-tune STT) là CC BY-NC-SA: model STT fine-tune chỉ dùng phi thương mại.
- Phát hiện buồn ngủ thử bằng webcam laptop, chưa thử trong cabin thật (ánh sáng, góc camera khác).
