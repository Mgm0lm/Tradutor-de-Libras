"""Decode existing Google TTS MP3s to PCM WAV for codec-independent Windows playback."""
from pathlib import Path
import argparse
import os
import wave
import pygame

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", metavar="LETRAS", help="Gera os MP3 ausentes com o Google TTS")
    args = parser.parse_args()
    cache = ROOT / "tts_cache"
    cache.mkdir(exist_ok=True)
    if args.generate:
        from gtts import gTTS
        for letter in dict.fromkeys(args.generate.upper()):
            if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                parser.error(f"Letra inválida: {letter!r}")
            path = cache / f"{letter}.mp3"
            if not path.exists():
                temporary = path.with_suffix(".mp3.part")
                gTTS(text=letter, lang="pt", timeout=(5, 15)).save(str(temporary))
                os.replace(temporary, path)
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    rate, sample_format, channels = pygame.mixer.get_init()
    if sample_format != -16:
        raise RuntimeError("Formato PCM não suportado")
    for source in cache.glob("*.mp3"):
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
