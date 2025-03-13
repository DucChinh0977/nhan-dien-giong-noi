import os
import torch
import torch.nn as nn
import librosa
import noisereduce as nr
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
import numpy as np
from tqdm import tqdm

# Kiểm tra CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Load mô hình Wav2Vec2
MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
embedding_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)

# Cấu hình âm thanh
SAMPLING_RATE = 16000

# Dataset tùy chỉnh để tải dữ liệu song song
class AudioDataset(Dataset):
    def __init__(self, directory, max_length=500):  # Giảm max_length
        self.audio_files = []
        self.labels = []
        self.label_map = {}
        idx = 0

        for speaker_name in os.listdir(directory):
            if speaker_name not in self.label_map:
                self.label_map[speaker_name] = idx
                idx += 1
            speaker_path = os.path.join(directory, speaker_name)
            if os.path.isdir(speaker_path):
                for file in os.listdir(speaker_path):
                    if file.endswith(".wav"):
                        self.audio_files.append(os.path.join(speaker_path, file))
                        self.labels.append(self.label_map[speaker_name])
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        audio_data, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
        embedding = self.extract_embedding(audio_data)
        return torch.tensor(embedding, dtype=torch.float32), label

    def extract_embedding(self, audio_data):
        audio_data = librosa.util.normalize(audio_data)
        audio_data = nr.reduce_noise(y=audio_data, sr=SAMPLING_RATE, stationary=False, prop_decrease=0.95, time_constant_s=2.0)
        inputs = processor(audio_data, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
        # Pooling để giảm sequence length
        embeddings = nn.functional.adaptive_avg_pool1d(embeddings.T, self.max_length).T  # (max_length, 768)
        return embeddings.cpu().numpy()

# Mô hình Transformer nhẹ hơn
class SpeakerTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads=4, dropout=0.1):
        super(SpeakerTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,  # Giảm FFN size
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)
        x = self.transformer(x)  # (batch, seq_len, hidden_size)
        x = x.mean(dim=1)  # (batch, hidden_size)
        out = self.fc(x)  # (batch, num_classes)
        return out

# Huấn luyện mô hình
def train_speaker_model(directory, input_size=768, hidden_size=128, num_layers=2, epochs=200, max_length=500):
    dataset = AudioDataset(directory, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # Song song hóa với 4 workers

    print(f"Label map: {dataset.label_map}")
    print(f"Total samples: {len(dataset)}")

    num_classes = len(dataset.label_map)
    model = SpeakerTransformer(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "speaker_model.pth")
            with open("label_map.pkl", "wb") as f:
                pickle.dump(dataset.label_map, f)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}!")
            break

    print("Mô hình và label_map đã được lưu!")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("Đã giải phóng bộ nhớ GPU!")

if __name__ == "__main__":
    train_speaker_model("samples/")