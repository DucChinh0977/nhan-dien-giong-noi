# train1.py
import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback  # Thêm EarlyStoppingCallback
)
import torchaudio.transforms as T
import random


# Hàm fine-tune cho Speech-to-Text
def train_speech_to_text():
    dataset_path = "D:/nhandang/samples"
    speakers = ["chinh", "viet", "VietLoi"]
    target_duration = 30  # Mục tiêu: 30 giây
    sample_rate = 16000  # Tần số lấy mẫu
    data = []

    # Duyệt qua các file âm thanh và padding/cắt nếu cần
    for speaker in speakers:
        speaker_path = os.path.join(dataset_path, speaker)
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                transcript_file = file.replace(".wav", ".txt")
                transcript_path = os.path.join(speaker_path, transcript_file)
                if os.path.exists(transcript_path):
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()

                    # Đọc file âm thanh
                    waveform, file_sample_rate = torchaudio.load(file_path)

                    # Chuyển đổi tần số lấy mẫu nếu cần
                    if file_sample_rate != sample_rate:
                        waveform = torchaudio.transforms.Resample(file_sample_rate, sample_rate)(waveform)

                    # Kiểm tra độ dài file âm thanh
                    duration = waveform.shape[1] / sample_rate
                    print(f"File: {file_path}, Duration: {duration:.2f} seconds")

                    # Padding nếu file ngắn hơn 30 giây
                    if duration < target_duration:
                        target_length = target_duration * sample_rate
                        padding_length = target_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding_length), mode='constant', value=0)
                        print(f"Padded file: {file_path}, New duration: {target_duration:.2f} seconds")
                    # Cắt nếu file dài hơn 30 giây
                    elif duration > target_duration:
                        target_length = target_duration * sample_rate
                        waveform = waveform[:, :target_length]
                        print(f"Truncated file: {file_path}, New duration: {target_duration:.2f} seconds")

                    # Tăng cường dữ liệu (Data Augmentation)
                    # Thêm tiếng ồn
                    with torch.no_grad():  # Tắt gradient khi tạo noise
                        noise = torch.randn_like(waveform) * 0.005  # Tiếng ồn nhẹ
                    waveform = waveform + noise

                    # Thay đổi cao độ (Pitch Shift)
                    pitch_transform = T.PitchShift(sample_rate, n_steps=2.0)
                    waveform = pitch_transform(waveform)

                    # Detach tensor để không theo dõi gradient trước khi lưu
                    waveform = waveform.detach()

                    # Lưu lại file đã xử lý
                    torchaudio.save(file_path, waveform, sample_rate)
                    new_duration = waveform.shape[1] / sample_rate

                    # Kiểm tra độ dài sau khi xử lý
                    if new_duration != target_duration:
                        print(
                            f"Warning: File {file_path} does not have exactly 30 seconds (duration: {new_duration:.2f} seconds)")

                    data.append({"path": file_path, "transcript": transcript})

    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Sử dụng mô hình công khai từ Hugging Face
    model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    print("Tải tokenizer và feature extractor cho speech-to-text...")
    try:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        print("Processor đã được tải thành công!")
        print("Tokenizer vocabulary:", tokenizer.get_vocab())
    except Exception as e:
        print(f"Lỗi khi tải: {e}")
        return

    def preprocess_function(batch):
        audio_array = batch["path"]["array"]
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        duration = len(audio_array) / 16000
        print(f"Processing file: {batch['path']['path']}, Duration: {duration:.2f} seconds")
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480000  # 30 giây (30 * 16000)
        )
        batch["input_values"] = inputs.input_values.squeeze(0)
        # Sử dụng tokenizer để mã hóa transcript
        labels = processor.tokenizer(batch["transcript"], return_tensors="pt").input_ids.squeeze(0)
        batch["labels"] = labels
        print(f"Transcript: {batch['transcript']}, Labels: {labels}")
        return batch

    dataset = dataset.map(preprocess_function, remove_columns=["path"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print("Tải mô hình để fine-tune speech-to-text...")
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.config.dropout = 0.2  # Thêm dropout để giảm overfitting
        print("Mô hình đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    training_args = TrainingArguments(
        output_dir="./wav2vec2_vietnamese_speech_to_text",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,  # Giảm learning rate
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=15,  # Giảm số epoch
        weight_decay=0.2,  # Tăng weight decay
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        lr_scheduler_type="linear",
        warmup_steps=50,
        load_best_model_at_end=True,  # Tải mô hình tốt nhất
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,  # Dừng nếu eval_loss không giảm sau 3 epoch
            early_stopping_threshold=0.01  # Cải thiện tối thiểu
        )]
    )

    print("Bắt đầu huấn luyện mô hình speech-to-text...")
    trainer.train()
    trainer.evaluate()
    trainer.save_model("wav2vec2_vietnamese_speech_to_text_model")
    processor.save_pretrained("wav2vec2_vietnamese_speech_to_text_model")
    print("Huấn luyện speech-to-text hoàn tất!")


# Hàm chính để chạy huấn luyện
def main():
    print("Bắt đầu huấn luyện Speech-to-Text...")
    train_speech_to_text()
    print("Hoàn tất huấn luyện Speech-to-Text!")


if __name__ == "__main__":
    main()