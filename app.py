# app.py
import pyaudio
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import wave
import os
import time

# Cấu hình Silero VAD
torch.set_num_threads(1)
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, read_audio, *_) = utils

# Cấu hình ghi âm
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_DURATION = 1.0
SPEECH_THRESHOLD = 0.5
MIN_SPEECH_DURATION = 0.3

# Cấu hình mô hình Speech-to-Text
STT_MODEL_PATH = "wav2vec2_vietnamese_speech_to_text_model"
stt_processor = Wav2Vec2Processor.from_pretrained(STT_MODEL_PATH)
stt_model = Wav2Vec2ForCTC.from_pretrained(STT_MODEL_PATH)
stt_model.eval()

# Cấu hình mô hình Speaker Identification
SPEAKER_MODEL_PATH = "wav2vec2_speaker_id_model"
speaker_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SPEAKER_MODEL_PATH)
speaker_model = Wav2Vec2ForSequenceClassification.from_pretrained(SPEAKER_MODEL_PATH)
speaker_model.eval()

# Ánh xạ nhãn người nói
label_map = {0: "Chinh", 1: "Viet", 2: "VietLoi"}

# Khởi tạo PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# Hàm nhận dạng người nói
def identify_speaker(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

    inputs = speaker_feature_extractor(
        waveform.squeeze(0),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=80000
    )
    with torch.no_grad():
        logits = speaker_model(inputs.input_values).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    speaker = label_map[predicted_id]
    return speaker


# Hàm chuyển đổi âm thanh thành văn bản
def speech_to_text(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

    inputs = stt_processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = stt_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = stt_processor.batch_decode(predicted_ids)[0]
    return transcription


# Hàm lưu đoạn âm thanh vào file .wav
def save_audio(frames, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Hàm phát hiện giọng nói bằng Silero VAD
def detect_speech(audio_chunk):
    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    audio_tensor = torch.tensor(audio_float32)
    speech_prob = vad_model(audio_tensor, RATE).item()
    return speech_prob > SPEECH_THRESHOLD


# Hàm lưu kết quả vào file
def save_result(speaker, transcription):
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(f"{speaker}: {transcription}\n")


# Hàm chính để ghi âm và xử lý theo thời gian thực
def main():
    print("Bắt đầu chương trình Speech-to-Text với VAD và Speaker Identification...")
    print("Đang lắng nghe... Nói để ghi âm, im lặng trong 1 giây để dừng.")

    audio_buffer = []
    is_speaking = False
    silence_start = None
    recording_count = 0

    # Xóa file results.txt nếu đã tồn tại
    if os.path.exists("results.txt"):
        os.remove("results.txt")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)

            if detect_speech(data):
                if not is_speaking:
                    print("🗣️ Phát hiện giọng nói, bắt đầu ghi âm...")
                    is_speaking = True
                    audio_buffer = []
                audio_buffer.append(data)
                silence_start = None
            else:
                if is_speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        print("🤫 Im lặng, dừng ghi âm...")
                        is_speaking = False

                        recording_count += 1
                        audio_path = f"recording_{recording_count}.wav"
                        save_audio(audio_buffer, audio_path)

                        waveform, _ = torchaudio.load(audio_path)
                        duration = waveform.shape[1] / RATE
                        if duration < MIN_SPEECH_DURATION:
                            print(f"Đoạn âm thanh quá ngắn ({duration:.2f} giây), bỏ qua...")
                            os.remove(audio_path)
                            continue

                        print("Nhận dạng người nói...")
                        speaker = identify_speaker(audio_path)

                        print("Chuyển đổi âm thanh thành văn bản...")
                        transcription = speech_to_text(audio_path)
                        print(f"{speaker}: {transcription}")

                        # Lưu kết quả vào file
                        save_result(speaker, transcription)

                        os.remove(audio_path)

            if not is_speaking:
                audio_buffer = []

        except KeyboardInterrupt:
            print("\nDừng chương trình...")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()