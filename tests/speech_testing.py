import sounddevice as sd
import numpy as np
import torchaudio
import io
from scipy.io.wavfile import write
from transformers import pipeline
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import sounddevice as sd  # For playback

# Load models once
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Load models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embedding
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Done.")
    wav_io = io.BytesIO()
    write(wav_io, sample_rate, audio)
    wav_io.seek(0)
    return wav_io

def transcribe_audio(wav_io):
    waveform, sr = torchaudio.load(wav_io)
    result = result = asr(
    {"raw": waveform.squeeze().numpy(), "sampling_rate": sr},
    generate_kwargs={"language": "en"}  # force English transcription
)
    return result['text']

def speak_text(text):
    """Convert text to speech and play it."""
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sd.play(speech.numpy(), samplerate=16000)
    sd.wait()  # Wait until playback finishes

def main():
    while True:
        audio_io = record_audio(duration=5)
        text = transcribe_audio(audio_io)
        print("User:", text)

        # Placeholder response logic (echo)
        response = f"You said: {text}"
        print("Bot:", response)
        speak_text(text)

if __name__ == "__main__":
    main()
