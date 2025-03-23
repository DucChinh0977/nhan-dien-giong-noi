This project creates a speech recognition system for online meetings, capable of identifying speakers (e.g., chinh, viet, or VietLoi) and transcribing their speech into text in real-time. It uses Wav2Vec2 models from Hugging Face for training, combined with Silero VAD for automatic voice activity detection. The results (speaker and transcribed text) are displayed on a web interface.

Features
Speaker Identification: Recognizes three speakers: chinh, viet, and VietLoi.
Speech-to-Text (STT): Transcribes Vietnamese speech into text in real-time.
Voice Activity Detection (VAD): Automatically starts and stops recording when speech is detected using Silero VAD.
Web Interface: Displays the meeting transcript with speaker labels (e.g., [Viet]: Xin chào mọi người).
Data Augmentation: Applies techniques like noise addition and pitch shifting to improve model robustness.

File Structure
app.py: Main application script that records audio, detects speech, identifies speakers, transcribes speech, and sends results to the web app.
train.py: Script to train the speaker identification model using Wav2Vec2.
train1.py: Script to train the speech-to-text (STT) model using Wav2Vec2.
web_app.py: Flask web application to display the meeting transcript on a web interface.
templates/index.html: HTML template for the web interface.
static/: Directory for static files (e.g., CSS, JavaScript) used by the web interface (currently empty).
whisp.py: (Optional) Additional script for speech processing (not used in the main workflow).

Requirements
Python: 3.8 or higher

Hardware: A microphone for recording audio; GPU recommended for faster training and inference

Operating System: Windows, macOS, or Linux

Python Libraries
Install the required libraries using the following command:
pip install torch torchaudio transformers datasets sounddevice wavio flask requests numpy noisereduce

Data Preparation
The project requires audio data for training the models. The data should be organized as follows:

Directory Structure:
Create a directory D:/nhandang/samples (or modify the path in the scripts to match your setup).
Inside D:/nhandang/samples, create subdirectories for each speaker
 
Data Requirements:
For train.py (Speaker Identification): Only .wav files are needed. Each subdirectory (person1, person2, person3) should contain audio files for the corresponding speaker.

For train1.py (Speech-to-Text): Each .wav file must have a corresponding .txt file with the same name, containing the transcript of the audio. For example, audio1.wav should have a audio1.txt file with the text
content (e.g., "Xin chào mọi người").
ecommendation: Use at least 50-100 audio files per speaker for better model performance.

Audio Specifications:
Format: .wav
Sampling rate: 16 kHz (if different, the scripts will resample automatically)
Duration: Any length (scripts will handle padding/truncation for STT)


How It Works

Speaker Identification (train.py):

Fine-tunes a Wav2Vec2 model (facebook/wav2vec2-base) for classifying speakers.

Applies data augmentation (noise addition, volume perturbation) to improve robustness.
Outputs a model that identifies 
Speech-to-Text (train1.py):
Fine-tunes a pre-trained Vietnamese Wav2Vec2 model (nguyenvulebinh/wav2vec2-base-vietnamese-250h) for STT.
Standardizes audio to 30 seconds, applies data augmentation (noise, pitch shift).
Outputs a model that transcribes Vietnamese speech into text.

Speech Processing (app.py):
Uses Silero VAD to detect speech and automatically start/stop recording.
Identifies the speaker using the model from train.py.
Transcribes the speech using the model from train1.py.
Sends the results (speaker and transcription) to the web app.
Web Interface (web_app.py and templates/index.html):
Displays the meeting transcript with speaker labels.
Updates in real-time as new speech is processed.

Troubleshooting
Error: "No speech detected":
Ensure your microphone is working and there is no excessive background noise.
Adjust the VAD_THRESHOLD in app.py (e.g., lower to 0.3 if speech is not detected).
Poor STT accuracy:
The STT model may not perform well due to limited training data. Add more .wav and .txt files to D:/nhandang/samples.
Alternatively, use Google Speech-to-Text for better accuracy.

Slow performance:
Use a GPU for faster training and inference.
Reduce CHUNK_DURATION in app.py to process smaller audio chunks.

Web interface not updating:
Ensure web_app.py is running before starting app.py.
Check the console for errors in web_app.py.

Future Improvements
Improve STT accuracy: Use Google Speech-to-Text or collect more Vietnamese training data.
Add more speakers: Expand the speaker identification model to support more speakers.
Enhance the web interface: Add features like saving transcripts, user authentication, or video integration.
Advanced noise reduction: Integrate more sophisticated denoising techniques (e.g., using deep learning models).
Contributing
Contributions are welcome! If you have ideas for improvements or bug fixes, please:

