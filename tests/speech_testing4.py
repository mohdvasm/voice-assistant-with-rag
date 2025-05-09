# # !pip install -q kokoro>=0.9.2 soundfile
# # !apt-get -qq -y install espeak-ng > /dev/null 2>&1

# from kokoro import KPipeline
# from IPython.display import display, Audio
# import soundfile as sf
# import torch
# pipeline = KPipeline(lang_code='a')
# text = '''
# [Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
# '''
# generator = pipeline(text, voice='af_heart')
# for i, (gs, ps, audio) in enumerate(generator):
#     print(i, gs, ps)
#     display(Audio(data=audio, rate=24000, autoplay=i==0))
#     sf.write(f'{i}.wav', audio, 24000)

# import pyttsx3

# def speak_with_pyttsx3(text: str):
#     # Initialize the TTS engine
#     engine = pyttsx3.init()
    
#     # Optional: adjust properties
#     engine.setProperty('rate', 160)     # Speed of speech (default ~200)
#     engine.setProperty('volume', 1.0)   # Volume (0.0 to 1.0)
    
#     # Speak the text
#     engine.say(text)
#     engine.runAndWait()

# # Example usage
# llm_output = """
# Assistant: Based on the context, you're interested in learning C programming. 
# Here are a few books: "The C Programming Language", "Programming in C", and 
# "Data Structures and Algorithms in C" by Mark Allen Weiss.
# """
# speak_with_pyttsx3(llm_output)

import pyttsx3

engine = pyttsx3.init(driverName='espeak')  # Force espeak

engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
for i, voice in enumerate(voices):
    print(f"{i}: {voice.name} - {voice.id}")
# Then try:
engine.setProperty('voice', voices[1].id)


engine.say("Hello, this is a test using espeak on Ubuntu.")
engine.runAndWait()
