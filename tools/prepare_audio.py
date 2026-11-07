"""Decode existing Google TTS MP3s to PCM WAV for codec-independent Windows playback."""
from pathlib import Path
import wave
import pygame

ROOT = Path(__file__).resolve().parents[1]

def main():
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    rate, sample_format, channels = pygame.mixer.get_init()
    if sample_format != -16:
        raise RuntimeError("Formato PCM não suportado")
    for source in (ROOT / "tts_cache").glob("*.mp3"):
        sound = pygame.mixer.Sound(str(source))
        destination = source.with_suffix(".wav")
        with wave.open(str(destination), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(sound.get_raw())
    pygame.mixer.quit()
    print("Cache WAV preparado a partir dos áudios Google existentes.")

if __name__ == "__main__":
    main()
