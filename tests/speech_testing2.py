from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import sounddevice as sd  # For playback

# Load models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Prepare text
inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")

# Load speaker embedding
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Generate speech waveform
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# Convert to numpy and play
sd.play(speech.numpy(), samplerate=16000)
sd.wait()  # Wait until playback finishes

def speak_text(text):
    """Convert text to speech and play it."""
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sd.play(speech.numpy(), samplerate=16000)
    sd.wait()  # Wait until playback finishes