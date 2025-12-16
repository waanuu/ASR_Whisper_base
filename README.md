# 🇻🇳🎙️ Vietnamese Automatic Speech Recognition (ASR) - Fine-tuning Whisper-base

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Kwann5002%2FWhisper__Base-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/Kwann5002/Whisper_Base)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)

Dự án này thực hiện việc **Fine-tune mô hình Whisper-base của OpenAI** cho bài toán **Nhận dạng tiếng nói tiếng Việt (Automatic Speech Recognition - ASR)**. Chúng tôi tập trung xây dựng một pipeline huấn luyện hiệu quả, tối ưu hóa bộ nhớ GPU để hoạt động tốt trong môi trường tài nguyên hạn chế (ví dụ: Kaggle/Colab).



---

## 🚀 1. Tổng quan Dự án

| Tính năng | Chi tiết | Ghi chú |
| :--- | :--- | :--- |
| **Mô hình nền** | `openai/whisper-base` | Mô hình Sequence-to-Sequence mạnh mẽ. |
| **Dữ liệu huấn luyện** | ~**54.000 mẫu** âm thanh tiếng Việt. | Đảm bảo tính tổng quát hóa cho giọng nói tiếng Việt. |
| **Mục tiêu** | Đạt độ chính xác cao (WER thấp) cho ASR tiếng Việt. | |
| **Tối ưu GPU** | Áp dụng FP16 & Gradient Accumulation. | Huấn luyện hiệu quả với giới hạn VRAM. |

---

## ⚙️ 2. Chiến lược Huấn luyện Tối ưu GPU

Để xử lý tập dữ liệu lớn (~54.000 mẫu) với giới hạn bộ nhớ GPU (VRAM), chiến lược huấn luyện đặc biệt đã được áp dụng:

### 2.1. Phân Chia và Huấn Luyện Tuần Tự

1.  Tập huấn luyện (~48.600 mẫu) được chia thành **10 batch lớn** (khoảng ~5.000 mẫu/batch).
2.  Mô hình được huấn luyện **tuần tự** trên từng batch.
3.  Trọng số mô hình được **cập nhật liên tục** giữa các batch, duy trì quá trình học.

### 2.2. Kỹ thuật Tiết kiệm Bộ nhớ

* **Mixed Precision (FP16):**
    * Sử dụng kiểu dữ liệu 16-bit thay vì 32-bit cho tham số và gradient.
    * **Lợi ích:** Giảm tiêu thụ bộ nhớ **~50%**, đồng thời tăng tốc độ huấn luyện.

* **Gradient Accumulation:**
    * Thực hiện nhiều bước forward/backward nhỏ, tích lũy gradient, sau đó chỉ cập nhật trọng số **một lần**.
    * **Lợi ích:** Mô phỏng việc sử dụng **Batch Size lớn** mà không làm tràn bộ nhớ VRAM. 

---

## 📊 3. Dữ liệu và Định dạng

* **Tổng số mẫu:** ~54.000
* **Tập huấn luyện:** ~48.600
* **Tập validation:** ~5.400
* **Tiền xử lý:** Âm thanh được lấy mẫu lại về **16kHz** và lọc độ dài để tương thích với mô hình Whisper.
* **Định dạng JSONL:**

```json
{"audio": "path/to/audio.wav", "text": "nội dung phiên âm"}


## 📈 4. Kết quả Đánh giá

Chỉ số đánh giá chính cho dự án ASR này là **Word Error Rate (WER)**. Dưới đây là tóm tắt kết quả huấn luyện:

| Chỉ số | Giá trị | Mô tả |
| :--- | :--- | :--- |
| **Chỉ số chính** | Word Error Rate (WER) | Tỷ lệ phần trăm từ bị lỗi (thay thế, xóa, chèn). |
| **Tập đánh giá** | ~5.400 mẫu | Số lượng mẫu âm thanh được sử dụng để tính WER. |
| **Training Loss cuối** | $\approx 0.27$ | Giá trị Loss Function khi quá trình huấn luyện kết thúc. |
| **WER cuối cùng** | *[Cập nhật sau]* | Vui lòng thay thế bằng giá trị WER thực tế trên tập Validation. |

---

## 💻 5. Công nghệ và Thư viện

Dự án được xây dựng dựa trên các công nghệ và thư viện mã nguồn mở sau:

### 5.1. Ngôn ngữ & Framework

* **Python:** Ngôn ngữ lập trình chính.
* **PyTorch:** Framework Deep Learning cốt lõi.

### 5.2. Thư viện Xử lý và Học máy

* **Hugging Face Transformers & Datasets:** Để quản lý mô hình Whisper và tập dữ liệu ASR.
* **Librosa:** Xử lý và tiền xử lý âm thanh (chuẩn hóa 16kHz).
* **Evaluate (WER metric):** Thư viện của Hugging Face để tính toán chỉ số WER.

### 5.3. Môi trường Huấn luyện

* **Kaggle/Google Colab GPU:** Môi trường điện toán đám mây với tài nguyên GPU hạn chế.

---

