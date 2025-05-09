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
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re 
import base64
import os
import io
import sounddevice as sd
import soundfile as sf
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

from openai import OpenAI

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Parameters
model = "playai-tts"
voice = "Arista-PlayAI"
text = "I love building and shipping new features for our users!"
response_format = "wav"

class SpeechUtils:
    def __init__(self):
        # Load models once
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

        # Load models
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load speaker embedding
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)


    def listen(self):
        wav_io = self.record_audio()
        text = self.transcribe_audio(wav_io)
        return text
    
    def speak(self, text, use_groq=True):
        # if use_groq:
        #     self.groq_speak_text(text)
        # else:
        #     self.speak_text(text)

        self.openai_speak_text(text)    

    
    def record_audio(self, duration=10, sample_rate=16000):
        print("Please speak! Listening...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Done.")
        wav_io = io.BytesIO()
        write(wav_io, sample_rate, audio)
        wav_io.seek(0)
        return wav_io

    def transcribe_audio(self, wav_io):
        waveform, sr = torchaudio.load(wav_io)
        result = self.asr(
        {"raw": waveform.squeeze().numpy(), "sampling_rate": sr},
        generate_kwargs={"language": "en"}  # force English transcription
    )
        return result['text']

    def speak_text_v0(self, text):
        """Convert text to speech and play it."""
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        sd.play(speech.numpy(), samplerate=16000)
        sd.wait()  # Wait until playback finishes

    def speak_text(self, text, max_tokens=500):
        """
        Splits long text into smaller chunks and speaks them one by one.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=[",", "\n", "."])

        text = self.preprocess_for_tts(text)
        chunks = text_splitter.split_text(text)
        print(f"\n>>> Splitting text into {len(chunks)} chunks.")
        for chunk in chunks:
            inputs = self.processor(text=chunk, return_tensors="pt")
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            sd.play(speech.numpy(), samplerate=16000)
            sd.wait()

    @staticmethod
    def preprocess_for_tts(text: str) -> str:
        text = text.replace('\n', ' ')
        text = text.replace("*", "")
        text = text.replace("-", " ")

        # 1. Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)

        # 2. Normalize punctuation (remove stray colons, fix spacing)
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)

        # 3. Optionally remove "Assistant:" or "User:" labels
        text = re.sub(r'^\s*(Assistant|User):\s*', '', text, flags=re.IGNORECASE)

        # 4. Trim overall
        return text.strip()
    
    @staticmethod
    def remove_empty_lines(text: str) -> str:
        # Split the text into lines
        lines = text.split("\n")

        # Remove empty lines and strip whitespace from each line
        non_empty_lines = [line.strip() for line in lines if line.strip()]


        spaced_lines = []
        for line in non_empty_lines: 
            line = line + " "
            spaced_lines.append(line)
        
        # Join the non-empty lines back into a single string
        return "\n".join(spaced_lines)
    
    def groq_speak_text(self, text):
        """Convert text to speech using Groq and play it."""

        text = self.remove_empty_lines(text)

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )
        
        # Use .read() to get bytes from BinaryAPIResponse
        audio_bytes = response.read()

        # Use BytesIO to wrap audio bytes for in-memory decoding
        audio_io = io.BytesIO(audio_bytes)

        # Decode and play
        data, samplerate = sf.read(audio_io)
        sd.play(data, samplerate=samplerate)
        sd.wait()

    def openai_speak_text(self, text):
        completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": text
            }
        ]
    )   
        
        # audio_bytes = completion.choices[0].message.audio.data
        audio_io = io.BytesIO(base64.b64decode(completion.choices[0].message.audio.data))
        
        # audio_io = base64.b64decode(completion.choices[0].message.audio.data)
        
        # Decode and play
        data, samplerate = sf.read(audio_io)
        sd.play(data, samplerate=samplerate)
        sd.wait()
