import os
import whisper

# Tải mô hình Whisper
model = whisper.load_model("large")  # Hoặc "medium" nếu máy yếu

# Đường dẫn đến thư mục chứa dữ liệu (sửa lỗi cú pháp)
dataset_path = "D:/nhandanggiongnoi/samples"
speakers = ["chinh", "viet", "VietLoi"]

# Tạo transcript cho từng file âm thanh
for speaker in speakers:
    speaker_path = os.path.join(dataset_path, speaker)
    for file in os.listdir(speaker_path):
        if file.endswith(".wav"):
            file_path = os.path.join(speaker_path, file)
            print(f"Đang tạo transcript cho {file_path}...")
            # Chuyển âm thanh thành văn bản
            result = model.transcribe(file_path, language="vi")
            transcript = result["text"]
            # Lưu transcript vào file .txt cùng tên
            transcript_file = file.replace(".wav", ".txt")
            transcript_path = os.path.join(speaker_path, transcript_file)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"Transcript: {transcript}")