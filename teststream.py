import requests
import os
import simpleaudio as sa
from io import BytesIO
from pydub import AudioSegment

DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

payload = {
    "text": "Hello, how can I help you today? My name is Emily and I'm very glad to meet you. What do you think of this new text-to-speech API?"
}

headers = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

audio_file_path = "output.mp3"  # Path to save the audio file

def synthesize_audio(text):
    payload = {"text": text}
    
    try:
        # Send the request to the Deepgram API and stream the audio
        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            r.raise_for_status()  # Check for errors in the response
            # Process the audio data and store it
            audio_data = b""
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    audio_data += chunk

            return audio_data

    except Exception as e:
        print(f"Error during audio synthesis: {e}")
        return None

def play_audio(audio_data):
    if audio_data:
        # Convert audio data into an AudioSegment and play it
        audio = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
        
        # Play the audio using simpleaudio
        play_obj = sa.play_buffer(audio.raw_data, 
                                  num_channels=audio.channels,
                                  bytes_per_sample=audio.sample_width,
                                  sample_rate=audio.frame_rate)
        play_obj.wait_done()

# Synthesize the audio
audio_data = synthesize_audio(payload["text"])

if audio_data:
    # Play the synthesized audio
    play_audio(audio_data)
    print("Audio playback complete")
else:
    print("Audio synthesis failed")
