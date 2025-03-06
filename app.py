import os
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa
import noisereduce as nr
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from silero_vad import get_speech_timestamps, load_silero_vad
from flask import Flask, render_template
from flask_socketio import SocketIO
import pickle

# Kiểm tra CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Khởi tạo Flask & SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load mô hình Wav2Vec2
MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
speech_to_text_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
embedding_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
vad_model = load_silero_vad().to(device)

# Cấu hình âm thanh
SAMPLING_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = SAMPLING_RATE * 3  # 3 giây
audio_buffer = []

# Tiền xử lý âm thanh
def preprocess_audio(audio_data, sr=SAMPLING_RATE):
    audio_data = librosa.util.normalize(audio_data)
    audio_data = nr.reduce_noise(y=audio_data, sr=sr)
    return audio_data

# Trích xuất chuỗi embedding
def extract_sequence_embeddings(audio_data, max_length=4500):
    audio_data = preprocess_audio(audio_data)
    inputs = processor(audio_data, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    if embeddings.size(0) > max_length:
        embeddings = embeddings[:max_length]
    else:
        padding = torch.zeros(max_length - embeddings.size(0), embeddings.size(1)).to(device)
        embeddings = torch.cat([embeddings, padding], dim=0)
    return embeddings.cpu().numpy()

# Xây dựng mô hình LSTM
class SpeakerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpeakerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Tải mô hình và label_map
def load_model_and_label_map(model_path="speaker_model.pth", label_map_path="label_map.pkl", input_size=768, hidden_size=128, num_layers=2):
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
    num_classes = len(label_map)
    model = SpeakerLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Mô hình và label_map đã được tải!")
    return model, label_map

# Nhận diện người nói
def identify_speaker(audio_data, model, label_map, max_length=4500):
    embedding = extract_sequence_embeddings(audio_data, max_length)
    embedding_tensor = torch.tensor([embedding], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(embedding_tensor)
        probas = torch.softmax(output, dim=1)[0]
        print(f"Probabilities for speakers {list(label_map.keys())}: {probas.tolist()}")
        best_idx = torch.argmax(probas)
        best_speaker = [k for k, v in label_map.items() if v == best_idx.item()][0]
        return best_speaker if probas[best_idx] > 0.5 else None

# Chuyển đổi giọng nói → văn bản
def speech_to_text(audio_data):
    audio_data = preprocess_audio(audio_data)
    inputs = processor(audio_data, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = speech_to_text_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

# Xử lý âm thanh thời gian thực
def process_audio(model, label_map):
    print("🔴 Hệ thống đang lắng nghe...")

    def callback(indata, frames, time, status):
        global audio_buffer
        if status:
            print(f"⚠️ Lỗi: {status}")
        audio_buffer.extend(indata[:, 0])
        while len(audio_buffer) >= CHUNK_SIZE:
            chunk = audio_buffer[:CHUNK_SIZE]
            audio_buffer = audio_buffer[CHUNK_SIZE:]
            audio_tensor = torch.tensor(np.array(chunk), dtype=torch.float32)
            speech_timestamps = get_speech_timestamps(audio_tensor.to(device), vad_model, sampling_rate=SAMPLING_RATE)
            if speech_timestamps:
                print("🎙 Phát hiện giọng nói...")
                speaker = identify_speaker(audio_tensor.numpy(), model, label_map)
                if speaker:
                    text = speech_to_text(audio_tensor.numpy())
                    print(f"🗣 [{speaker}]: {text}")
                    socketio.emit("transcription", {"speaker": speaker, "text": text})
                else:
                    print("⚠️ Giọng nói không xác định, bỏ qua...")

    with sd.InputStream(samplerate=SAMPLING_RATE, channels=CHANNELS, callback=callback):
        socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

# Trang giao diện web
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    model, label_map = load_model_and_label_map()
    process_audio(model, label_map)