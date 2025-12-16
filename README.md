# Hệ thống chuyển đổi âm thanh tiếng Việt sang văn bản

## 🔍 Tổng quan dự án

- Fine-tune mô hình Whisper-base cho bài toán speech-to-text tiếng Việt
- Huấn luyện trên ~54.000 mẫu dữ liệu âm thanh tiếng Việt
- Áp dụng huấn luyện tuần tự theo batch để tối ưu bộ nhớ GPU
- Sử dụng mixed precision (FP16) và gradient accumulation
- Đánh giá mô hình bằng chỉ số Word Error Rate (WER)

---

## 📊 Dữ liệu

- **Tổng số mẫu:** ~54.000
- **Tập huấn luyện:** ~48.600 mẫu
- **Tập validation:** ~5.400 mẫu
- Âm thanh được chuẩn hóa về **16kHz** và lọc theo độ dài để phù hợp với Whisper

### Định dạng dữ liệu (JSONL)
```json
{"audio": "path/to/audio.wav", "text": "nội dung phiên âm"}
🧠 Mô hình

Mô hình nền: openai/whisper-base

Bài toán: Nhận dạng tiếng nói tiếng Việt (Automatic Speech Recognition)

Framework: Hugging Face Transformers

⚙️ Chiến lược huấn luyện

Do giới hạn bộ nhớ GPU trên Kaggle, mô hình được huấn luyện theo chiến lược tối ưu bộ nhớ:

Chia dữ liệu huấn luyện thành 10 batch (~5.000 mẫu mỗi batch)

Huấn luyện tuần tự từng batch

Trọng số mô hình được cập nhật liên tục giữa các batch

Sử dụng mixed precision (FP16) để giảm tiêu thụ bộ nhớ GPU

Áp dụng gradient accumulation để mô phỏng batch size lớn
📈 Đánh giá

Chỉ số đánh giá: Word Error Rate (WER)

Tập validation: ~5.400 mẫu

Kết quả: Mô hình hội tụ ổn định qua các batch huấn luyện

Training loss cuối: ~0.27
🛠 Công nghệ sử dụng

Python

PyTorch

Hugging Face Transformers & Datasets

Librosa

Evaluate (WER)

Kaggle GPU
🚀 Mô hình đã huấn luyện

Mô hình Whisper đã fine-tune được công khai trên Hugging Face:

👉 Hugging Face Model:
https://huggingface.co/Kwann5002/Whisper_Base
