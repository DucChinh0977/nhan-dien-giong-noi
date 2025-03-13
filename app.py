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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")

app = Flask(__name__)
socketio = SocketIO(app)

MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
speech_to_text_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
embedding_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
vad_model = load_silero_vad().to(device)

SAMPLING_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = SAMPLING_RATE * 3
audio_buffer = []

def preprocess_audio(audio_data, sr=SAMPLING_RATE):
    audio_data = librosa.util.normalize(audio_data)
    audio_data = nr.reduce_noise(y=audio_data, sr=sr, stationary=False, prop_decrease=0.95, time_constant_s=2.0)
    return audio_data

def extract_sequence_embeddings(audio_data, max_length=1000):  # Tăng max_length
    audio_data = preprocess_audio(audio_data)
    inputs = processor(audio_data, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    embeddings = nn.functional.adaptive_avg_pool1d(embeddings.T, max_length).T
    if embeddings.size(0) < max_length:
        padding = torch.zeros(max_length - embeddings.size(0), embeddings.size(1)).to(device)
        embeddings = torch.cat([embeddings, padding], dim=0)
    return embeddings.cpu().numpy()

class SpeakerTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads=4, dropout=0.3):
        super(SpeakerTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

def load_model_and_label_map(model_path="speaker_model.pth", label_map_path="label_map.pkl", input_size=768, hidden_size=128, num_layers=2):
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        raise FileNotFoundError("Model hoặc label_map file không tồn tại. Hãy chạy train.py trước!")
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
    num_classes = len(label_map)
    model = SpeakerTransformer(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Mô hình và label_map đã được tải!")
    return model, label_map

def identify_speaker(audio_data, model, label_map, max_length=1000):
    embedding = extract_sequence_embeddings(audio_data, max_length)
    embedding_tensor = torch.tensor([embedding], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(embedding_tensor)
        probas = torch.softmax(output, dim=1)[0]
        print(f"Probabilities for speakers {list(label_map.keys())}: {probas.tolist()}")
        best_idx = torch.argmax(probas)
        best_speaker = [k for k, v in label_map.items() if v == best_idx.item()][0]
        return best_speaker if probas[best_idx] > 0.3 else None

def speech_to_text(audio_data):
    audio_data = preprocess_audio(audio_data)
    inputs = processor(audio_data, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = speech_to_text_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

def process_audio(model, label_map):
    print("🔴 Hệ thống đang lắng nghe...")
    def callback(indata, frames, time, status):
        global audio_buffer
        if status:
            print(f"⚠️ Lỗi: {status}")
        indata = np.clip(indata, -1.0, 1.0)
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

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    model, label_map = load_model_and_label_map()
    process_audio(model, label_map)