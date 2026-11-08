"""Decode existing Google TTS MP3s to PCM WAV for codec-independent Windows playback."""
from pathlib import Path
import argparse
import os
import wave
import numpy as np
import pygame

ROOT = Path(__file__).resolve().parents[1]

def trim_silence(pcm, rate, channels):
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
    block = max(1, rate // 100)
    complete = len(samples) // block * block
    if not complete:
        return pcm
    blocks = samples[:complete].astype(np.float64).reshape(-1, block, channels)
    rms = np.sqrt(np.mean(blocks ** 2, axis=(1, 2)))
    voiced = np.flatnonzero(rms > max(45, rms.max() * 0.015))
    if not voiced.size:
        return pcm
    # Leave a margin around speech so quiet consonants are preserved.
    start = max(0, int(voiced[0]) * block - rate * 25 // 1000)
    end = min(len(samples), (int(voiced[-1]) + 1) * block + rate * 35 // 1000)
    return pcm[start * channels * 2:end * channels * 2]

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
        temporary = destination.with_suffix(".wav.part")
        with wave.open(str(temporary), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(trim_silence(sound.get_raw(), rate, channels))
        os.replace(temporary, destination)
    pygame.mixer.quit()
    print("Cache WAV preparado a partir dos áudios Google existentes.")

if __name__ == "__main__":
    main()
