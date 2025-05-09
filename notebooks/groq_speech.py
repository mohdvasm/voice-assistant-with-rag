# import os
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()

# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# speech_file_path = "speech.wav" 
# model = "playai-tts"
# voice = "Fritz-PlayAI"
# text = "I love building and shipping new features for our users!"
# response_format = "wav"

# response = client.audio.speech.create(
#     model=model,
#     voice=voice,
#     input=text,
#     response_format=response_format
# )

# response.write_to_file(speech_file_path)

import os
import io
import sounddevice as sd
import soundfile as sf
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Parameters
model = "playai-tts"
voice = "Arista-PlayAI"
text = "I love building and shipping new features for our users!"
response_format = "wav"

# Generate speech
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

def speak_groq(text):
    """Convert text to speech using Groq and play it."""
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
