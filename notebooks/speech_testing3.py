# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = None

# # load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
# # input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt", language="en").input_features 

# inputs = processor(
#     sample["array"], 
#     sampling_rate=sample["sampling_rate"], 
#     return_tensors="pt", 
#     language="en",  # optional
#     return_attention_mask=True
# )

# input_features = inputs.input_features
# attention_mask = inputs.attention_mask

# predicted_ids = model.generate(input_features, attention_mask=attention_mask)


# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# print(transcription[0])

import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz
DURATION = 5  # seconds to record

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.squeeze()  # Convert from (n, 1) to (n,)

def transcribe_audio(audio_array, sample_rate=16000):
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language="en",  # Set if you want transcription in English
        return_attention_mask=True,
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features,
            attention_mask=inputs.attention_mask
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Main flow
if __name__ == "__main__":
    audio_data = record_audio(DURATION, SAMPLE_RATE)
    text = transcribe_audio(audio_data, SAMPLE_RATE)
    print("Transcription:", text)
