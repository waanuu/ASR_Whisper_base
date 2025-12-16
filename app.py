import gradio as gr
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_path = "./batch_10"  # Hoặc path local của bạn
import zipfile
import os

if not os.path.exists("batch_10"):
    with zipfile.ZipFile("batch_10.zip", 'r') as zip_ref:
        zip_ref.extractall("batch_10")
print("🔄 Đang load model...")
try:
    # Load processor và model Whisper
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Chuyển model sang GPU nếu có
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✅ Load model thành công! Đang chạy trên: {device}")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    print("💡 Sử dụng model Whisper mặc định từ OpenAI")
    # Fallback sang model Whisper mặc định
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

# ==============================================================================
# HÀM XỬ LÝ ÂM THANH VÀ CHUYỂN ĐỔI THÀNH TEXT
# ==============================================================================

def transcribe_audio(audio_input):
    """
    Nhận dạng giọng nói từ audio và chuyển thành text tiếng Việt
    
    Args:
        audio_input: tuple (sample_rate, audio_array) từ Gradio
    
    Returns:
        text: văn bản tiếng Việt được nhận dạng
    """
    try:
        # Kiểm tra input
        if audio_input is None:
            return "❌ Vui lòng ghi âm hoặc upload file audio"
        
        # Gradio trả về tuple (sample_rate, audio_data) hoặc đường dẫn file
        if isinstance(audio_input, str):
            # Nếu là đường dẫn file
            audio_array, sample_rate = librosa.load(audio_input, sr=16000)
        else:
            # Nếu là tuple từ microphone
            sample_rate, audio_array = audio_input
            # Resample về 16kHz nếu cần
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    y=audio_array.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=16000
                )
        
        # Chuyển sang mono nếu là stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Chuẩn hóa audio
        audio_array = audio_array.astype(np.float32)
        
        # Xử lý với processor
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        input_features = input_features.to(device)
        
        # Generate token ids với forced_decoder_ids để output tiếng Việt
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="vi",
            task="transcribe"
        )
        
        # Inference
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
        
        # Decode prediction
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        # Trả về kết quả
        if transcription.strip() == "":
            return "⚠️ Không nhận dạng được giọng nói. Vui lòng thử lại với âm thanh rõ ràng hơn."
        
        return f"📝 Văn bản nhận dạng:\n\n{transcription}"
    
    except Exception as e:
        return f"❌ Lỗi xảy ra: {str(e)}\n\nVui lòng thử lại hoặc kiểm tra file audio."

# ==============================================================================
# TẠO GIAO DIỆN GRADIO
# ==============================================================================

# Custom CSS để làm đẹp giao diện
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.output-text {
    font-size: 18px;
    line-height: 1.6;
}
"""

# Tạo interface
with gr.Blocks(css=custom_css, title="Speech to Text Tiếng Việt - Whisper") as demo:
    gr.Markdown(
        """
        # 🎤 Chuyển đổi Giọng nói thành Văn bản Tiếng Việt
        ### Powered by Whisper Model
        
        Ghi âm giọng nói của bạn hoặc upload file audio, hệ thống sẽ tự động chuyển đổi thành văn bản tiếng Việt.
        
        ### 📌 Hướng dẫn sử dụng:
        1. **Ghi âm trực tiếp**: Click vào nút microphone để ghi âm
        2. **Upload file**: Hoặc upload file audio có sẵn (hỗ trợ .wav, .mp3, .flac, .m4a)
        3. Nhấn **"Chuyển đổi"** để nhận dạng giọng nói
        
        ### ⚡ Lưu ý:
        - Nói rõ ràng, tốc độ vừa phải
        - Giảm thiểu tiếng ồn xung quanh
        - File audio nên dưới 30 giây để kết quả tốt nhất
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input audio với cả microphone và upload file
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎙️ Ghi âm hoặc Upload Audio"
            )
            
            # Nút submit
            submit_btn = gr.Button(
                "🚀 Chuyển đổi sang Text", 
                variant="primary",
                size="lg"
            )
            
            # Nút xóa
            clear_btn = gr.Button("🗑️ Xóa", size="sm")
        
        with gr.Column(scale=1):
            # Output text
            output_text = gr.Textbox(
                label="📄 Kết quả nhận dạng",
                placeholder="Văn bản sẽ hiển thị ở đây...",
                lines=10,
                elem_classes="output-text"
            )
    
    gr.Markdown(
        """
        ---
        ### 💡 Tips để có kết quả tốt nhất:
        - ✅ Môi trường yên tĩnh
        - ✅ Phát âm rõ ràng
        - ✅ Giọng nói tự nhiên
        - ✅ Tránh nói quá nhanh hoặc quá chậm
        
        ### 🔧 Thông tin kỹ thuật:
        - Model: Whisper (fine-tuned cho tiếng Việt)
        - Sampling rate: 16kHz
        - Device: {}
        - Language: Vietnamese
        """.format("GPU 🚀" if torch.cuda.is_available() else "CPU 💻")
    )
    
    # Kết nối các sự kiện
    submit_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=output_text
    )
    
    clear_btn.click(
        fn=lambda: (None, ""),
        outputs=[audio_input, output_text]
    )

# ==============================================================================
# KHỞI CHẠY DEMO
# ==============================================================================

if __name__ == "__main__":
    # Launch với cấu hình phù hợp cho Kaggle
    demo.launch(
        share=True,  # Tạo public link
        debug=True,  # Hiển thị lỗi chi tiết
        show_error=True
    )