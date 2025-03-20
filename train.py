# train.py
import os
import torch
import torchaudio
import numpy as np
import random
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


# Hàm fine-tune cho Speaker Identification
def train_speaker_identification():
    # Định nghĩa nhãn cho 3 người nói
    label_map = {"chinh": 0, "viet": 1, "VietLoi": 2}
    sample_rate = 16000

    # Chuẩn bị dữ liệu
    dataset_path = "D:/nhandang/samples"
    data = []
    for speaker, label in label_map.items():
        speaker_path = os.path.join(dataset_path, speaker)
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                data.append({"path": file_path, "label": torch.tensor(label, dtype=torch.long)})

    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("path", Audio(sampling_rate=sample_rate))

    # Tải feature extractor từ mô hình nhỏ hơn
    model_name = "facebook/wav2vec2-base"
    print("Tải feature extractor cho speaker identification...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        print("Feature extractor đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải feature extractor: {e}")
        return

    # Hàm tăng cường dữ liệu
    def augment_audio(waveform, sample_rate):
        print(f"Waveform dtype before augmentation: {waveform.dtype}")

        # Thêm tiếng ồn
        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

        # Thay đổi âm lượng (Volume Perturbation)
        volume_factor = random.uniform(0.5, 1.5)
        waveform = waveform * volume_factor

        print(f"Waveform dtype after augmentation: {waveform.dtype}")
        return waveform

    # Hàm tiền xử lý dữ liệu
    def preprocess_function(batch):
        audio_array = batch["path"]["array"]
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Chuyển đổi thành tensor với dtype=float32
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_length)

        # Tăng cường dữ liệu
        waveform = augment_audio(waveform, sample_rate)

        # Xử lý đầu vào bằng feature extractor
        inputs = feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=80000
        )
        # Đảm bảo input_values có shape [seq_length]
        batch["input_values"] = inputs.input_values.squeeze(0)  # Shape: [seq_length]

        # Đảm bảo nhãn là torch.long
        label = batch["label"]
        if isinstance(label, torch.Tensor):
            batch["label"] = label.to(dtype=torch.long)
        else:
            batch["label"] = torch.tensor(label, dtype=torch.long)

        return batch

    dataset = dataset.map(preprocess_function, remove_columns=["path"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Tải mô hình để fine-tune
    print("Tải mô hình để fine-tune speaker identification...")
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_map), problem_type="single_label_classification"
        )
        model.config.dropout = 0.3
        print("Mô hình đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Cấu hình huấn luyện
    training_args = TrainingArguments(
        output_dir="./wav2vec2_speaker_id",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=15,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        gradient_checkpointing=True  # Sửa cảnh báo gradient_checkpointing
    )

    # Hàm tính toán độ chính xác
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"eval_accuracy": accuracy}

    # Hàm collate tùy chỉnh
    def custom_collate_fn(batch):
        # Đảm bảo input_values là tensor và có shape [seq_length]
        input_values = [item["input_values"] if isinstance(item["input_values"], torch.Tensor) else torch.tensor(
            item["input_values"], dtype=torch.float32) for item in batch]
        input_values = torch.stack(input_values)  # Shape: [batch_size, seq_length]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        print(f"Input values shape: {input_values.shape}, dtype: {input_values.dtype}")
        return {"input_values": input_values, "labels": labels}

    # Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor,
        data_collator=custom_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )]
    )

    # Huấn luyện và lưu mô hình
    print("Bắt đầu huấn luyện mô hình speaker identification...")
    trainer.train()
    trainer.evaluate()
    trainer.save_model("wav2vec2_speaker_id_model")
    feature_extractor.save_pretrained("wav2vec2_speaker_id_model")
    print("Huấn luyện speaker identification hoàn tất!")


if __name__ == "__main__":
    train_speaker_identification()