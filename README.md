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

## Cảnh báo biển báo (output thật, dữ liệu luật thật, chạy trên laptop)

```
ô tô   P.102   Phía trước có biển cấm đi ngược chiều. Ô tô vi phạm bị phạt 18 triệu đến 20 triệu đồng, trừ 4 điểm bằng lái.
ô tô   P.131a  Phía trước có biển cấm đỗ xe. Ô tô vi phạm bị phạt 800 nghìn đến 1 triệu đồng.
xe máy P.102   Phía trước có biển cấm đi ngược chiều. Xe máy vi phạm bị phạt 4 triệu đến 6 triệu đồng, trừ 2 điểm bằng lái.
xe máy P.123a  Phía trước có biển cấm rẽ trái. Xe máy vi phạm bị phạt 600 nghìn đến 800 nghìn đồng.
ô tô   P.127   Phía trước có biển tốc độ tối đa cho phép.
ô tô   P.127*50  Phía trước có biển tốc độ tối đa cho phép 50 km/h.
ô tô   P.103a  Phía trước có biển cấm xe ô tô. Ô tô vi phạm bị phạt 4 triệu đến 6 triệu đồng, trừ 2 điểm bằng lái.
xe máy P.103a  Phía trước có biển cấm xe ô tô.
xe máy P.104   Phía trước có biển cấm xe mô tô và xe máy. Xe máy vi phạm bị phạt 2 triệu đến 3 triệu đồng, trừ 2 điểm bằng lái.
```

Bug đã sửa (04/10/2026), soát cả 52 mã biển của VNTS × 2 loại xe: trước chỉ 15/104 cặp ra câu cảnh báo đúng.
Detector trả mã kèm hậu tố (`P.127*50`, `P.124a*`) không khớp bảng tra nên im lặng; còn tìm kiếm thì có lúc lấy
nhầm khoản "gây tai nạn" (P.103a "cấm ô tô" lại phạt xe máy 10-14 triệu). Giờ biển cấm/hiệu lệnh ghim thẳng điểm
luật NĐ 168 theo loại xe (soát tay), biển không áp dụng cho loại xe đang lái thì chỉ đọc tên: 88/104 cặp có cảnh
báo, chưa thấy cặp nào sai. 16 cặp còn im lặng là biển chỉ dẫn/biển phụ.

Đoạn hội thoại bằng giọng nói sẽ được thêm sau khi model function-calling train xong (`copilot replay` ghi log vào `logs/replay.json`).

## Latency trên laptop (CPU, 4 luồng)

| thành phần | p50 | ghi chú |
|---|---|---|
| tra luật cho biển báo | < 1 ms | biển cấm/hiệu lệnh ghim sẵn điểm luật; biển còn lại (tốc độ, chiều cao) tìm kiếm ~0.7-1 s, có cache |
| biển báo -> câu cảnh báo | ~2 ms | regex lấy mức phạt, không gọi LLM |
| STT PhoWhisper-small int8 | chưa đo | |
| function calling Qwen3-1.7B Q4 | chưa đo | |
| TTS Piper (vi_VN-vais1000-medium) | ~0.3 s | câu ngắn ~0.2 s, câu cảnh báo dài nhất ~0.7 s; RTF ~0.1 |
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
- **STT: dùng PhoWhisper-small gốc, không dùng bản fine-tune có ồn** (xem kết quả dưới).

## Fine-tune STT với tiếng ồn: thử rồi, kém hơn bản gốc

Notebook 01: PhoWhisper-small, 400 step (batch 32, ~1,1 epoch VIVOS), lr 1e-5, 70% mẫu train trộn ồn tổng hợp
(đường, gió, máy) SNR ngẫu nhiên 0-15 dB. Đánh giá 200 câu VIVOS test, ồn test là đoạn ồn chưa dùng lúc train.

| WER (%) | sạch | SNR 10 dB | SNR 5 dB | SNR 0 dB |
|---|---|---|---|---|
| PhoWhisper-small gốc | **2.14** | **3.81** | **6.98** | **14.76** |
| fine-tune có ồn | 3.13 | 5.16 | 8.61 | 15.87 |

Fine-tune làm tệ hơn ở mọi mức, cả giọng sạch. Mình đoán: PhoWhisper đã fine-tune trên 844h tiếng Việt nhiều giọng, nên 1 epoch VIVOS (~15h)
không thêm gì mà chỉ làm lệch model; ồn tổng hợp cũng khác ồn thật. Bản gốc đã chịu ồn khá tốt tới SNR 5 dB (WER < 7%),
nên demo dùng bản gốc đổi sang CTranslate2 int8. Muốn thử lại thì cần ồn thu thật trong xe và giữ lr nhỏ hơn / đóng băng encoder.

## Chạy

```bash
pip install -e ".[parts,live]"
python scripts/download_models.py           # ~1.6GB vào models/
copilot chat                                # gõ lệnh, không cần mic/camera
copilot replay --dashcam drive.mp4 --cabin face.mp4 --script demo/script.yaml --speak
```

Cấu hình đường dẫn trong `configs/default.yaml`. Thành phần nào thiếu model thì tự tắt và báo.

Fine-tune PhoWhisper chịu ồn: `notebooks/01_finetune_phowhisper_noise` (Kaggle hoặc Colab, T4).
Chạy trên Colab: mở notebook từ GitHub (File → Open notebook → GitHub → DuongCodeAI/viet-copilot), chọn T4 GPU,
thêm Secret `HF_TOKEN`. Checkpoint + model lưu vào Google Drive (`MyDrive/ai-portfolio/viet-copilot`), tiếng ồn tự thu để ở `xe-noise/` trong đó.

## Hạn chế

- Xe là giả lập (`vehicle.py`); tool dẫn đường / trạm sạc trả dữ liệu mẫu.
- STT dùng PhoWhisper-small gốc (BSD-3). VIVOS (CC BY-NC-SA) chỉ dùng để đánh giá và thử fine-tune.
- Phát hiện buồn ngủ thử bằng webcam laptop, chưa thử trong cabin thật (ánh sáng, góc camera khác).
