import wave
import numpy as np
import soundfile as sf
from piper import PiperVoice
import pygame

# Načítanie hlasu
VOICE = PiperVoice.load(
    "voices/sk_SK-lili-medium.onnx",
    "voices/sk_SK-lili-medium.onnx.json"
)

def pitch_shift(data, semitones, sr):
    factor = 2 ** (semitones / 12)
    idx = np.arange(0, len(data), 1/factor)
    idx = idx[idx < len(data)]
    return np.interp(idx, np.arange(len(data)), data)

def speed_change(data, speed_factor):
    idx = np.arange(0, len(data), speed_factor)
    idx = idx[idx < len(data)]
    return data[idx.astype(int)]

def anime_voice_process(in_file, out_file, semitones=2.8, speed=1.12):
    data, sr = sf.read(in_file)
    data = pitch_shift(data, semitones, sr)
    data = speed_change(data, speed)
    data = data / np.max(np.abs(data))
    sf.write(out_file, data, sr)

def speak_piper(text):
    pygame.mixer.init()

    # 1) TTS -> raw.wav
    with wave.open("raw.wav", "wb") as wav:
        VOICE.synthesize(text, wav)

    # 2) Anime úprava
    anime_voice_process("raw.wav", "final.wav")

    # 3) Prehrávanie
    sound = pygame.mixer.Sound("final.wav")
    channel = sound.play()
    return channel
