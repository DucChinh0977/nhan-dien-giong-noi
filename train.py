import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import noisereduce as nr
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Kiểm tra CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Load mô hình Wav2Vec2
MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    embedding_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
except Exception as e:
    print(f"Lỗi khi load Wav2Vec2: {e}")
    exit(1)

# Cấu hình âm thanh
SAMPLING_RATE = 16000

# Tiền xử lý âm thanh
def preprocess_audio(audio_data, sr=SAMPLING_RATE):
    audio_data = librosa.util.normalize(audio_data)
    audio_data = nr.reduce_noise(y=audio_data, sr=sr)
    return audio_data

# Trích xuất chuỗi embedding
def extract_sequence_embeddings(audio_data, max_length=4500):
    try:
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
    except Exception as e:
        print(f"Lỗi khi trích xuất embedding: {e}")
        return None

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

# Huấn luyện mô hình
def train_speaker_model(directory, input_size=768, hidden_size=128, num_layers=2, epochs=100, max_length=4500):
    embeddings = []
    labels = []
    label_map = {}
    idx = 0

    if not os.path.exists(directory):
        print(f"Thư mục {directory} không tồn tại!")
        return

    for speaker_name in os.listdir(directory):
        if speaker_name not in label_map:
            label_map[speaker_name] = idx
            idx += 1
        speaker_path = os.path.join(directory, speaker_name)
        if os.path.isdir(speaker_path):
            for file in os.listdir(speaker_path):
                if file.endswith(".wav"):
                    path = os.path.join(speaker_path, file)
                    try:
                        audio_data, _ = librosa.load(path, sr=SAMPLING_RATE)
                        embedding = extract_sequence_embeddings(audio_data, max_length)
                        if embedding is None:
                            continue
                        print(f"File: {path}, Embedding shape: {embedding.shape}")
                        embeddings.append(torch.tensor(embedding, dtype=torch.float32))
                        labels.append(label_map[speaker_name])
                    except Exception as e:
                        print(f"Lỗi khi xử lý file {path}: {e}")

    if not embeddings:
        print("Không có dữ liệu để huấn luyện!")
        return

    try:
        X = pad_sequence(embeddings, batch_first=True).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)
        print(f"Label map: {label_map}")
        print(f"Labels: {labels}")
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        num_classes = len(label_map)
        model = SpeakerLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Lưu mô hình và label_map
        torch.save(model.state_dict(), "speaker_model.pth")
        with open("label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)
        print("Mô hình và label_map đã được lưu!")

        # Giải phóng bộ nhớ GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print("Đã giải phóng bộ nhớ GPU!")
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện hoặc lưu file: {e}")

if __name__ == "__main__":
    train_speaker_model("samples/")