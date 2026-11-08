"""
Fala as letras reconhecidas usando o Google Text-to-Speech (gTTS).

Duas preocupacoes praticas pra uso ao vivo numa apresentacao:

1) gTTS PRECISA de internet (ele so gera o mp3 chamando um endpoint do
   Google Translate) - se a rede da sala cair ou for lenta, a fala
   atrasa ou falha bem na hora H. Solucao: gerar e cachear o audio de
   cada letra (sao só 21) em disco, uma unica vez. Da segunda execucao
   em diante (ou mesmo na primeira, apos o cache "esquentar" no
   inicio), a reproducao usa o arquivo local, sem depender mais de
   rede.

2) Chamar a rede ou tocar audio direto na thread da UI trava a
   interface. Por isso isso tudo roda numa QThread separada, com uma
   fila (Queue) de letras a serem faladas.
"""

import logging
import os
import queue
import time

from PySide6.QtCore import QThread, Signal

import config

logger = logging.getLogger(__name__)


class TTSWorker(QThread):
    cache_ready = Signal()
    speak_error = Signal(str)
    speaking = Signal(str)  # emitido quando comeca a falar uma letra

    def __init__(self, letters, parent=None):
        super().__init__(parent)
        self.letters = letters  # lista de letras unicas (config.LETTERS.values())
        self._queue = queue.Queue(maxsize=1)
        self._running = False
        self._mixer_ready = False

    def enqueue(self, letter):
        """Chamado pela thread principal (main.py) quando uma letra é
        confirmada. Não bloqueia."""
        if not config.ENABLE_TTS:
            return
        self.clear_pending()
        self._queue.put_nowait(letter)

    def clear_pending(self):
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def run(self):
        self._running = True
        os.makedirs(config.TTS_CACHE_DIR, exist_ok=True)

        self._warm_cache()
        self._init_mixer()

        while self._running:
            try:
                letter = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if config.ENABLE_TTS:
                self._play(letter)

    def _cache_path(self, letter):
        return os.path.join(config.TTS_CACHE_DIR, f"{letter}.mp3")

    def _warm_cache(self):
        """Gera (só se ainda não existir) o mp3 de cada letra. Roda uma
        vez no início - se a internet estiver ok, deixa tudo pronto
        para o resto da apresentação funcionar offline."""
        missing = [l for l in self.letters if not os.path.exists(self._cache_path(l))]
        if not missing:
            self.cache_ready.emit()
            return

        try:
            from gtts import gTTS
        except Exception:
            logger.exception("Pacote 'gTTS' não encontrado - TTS ficará desativado.")
            self.speak_error.emit("Pacote gTTS não instalado (pip install gTTS).")
            self.cache_ready.emit()
            return

        for letter in missing:
            if not self._running and letter != missing[0]:
                break
            try:
                tts = gTTS(text=letter, lang=config.TTS_LANG)
                tts.save(self._cache_path(letter))
                logger.info("Áudio TTS gerado para a letra %s", letter)
            except Exception:
                logger.exception(
                    "Falha ao gerar áudio TTS para '%s' (sem internet?). "
                    "Essa letra ficará muda até haver conexão.", letter
                )
        self.cache_ready.emit()

    def _init_mixer(self):
        try:
            import pygame
            pygame.mixer.init()
            self._mixer_ready = True
        except Exception:
            logger.exception("Falha ao iniciar o mixer de áudio (pygame).")
            self.speak_error.emit("Não foi possível iniciar o áudio (pygame.mixer).")
            self._mixer_ready = False

    def _play(self, letter):
        path = self._cache_path(letter)
        if not os.path.exists(path):
            logger.warning("Sem áudio em cache para '%s' - pulando fala.", letter)
            return

        self.speaking.emit(letter)

        if not self._mixer_ready:
            return

        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self._running:
                time.sleep(0.05)
        except Exception:
            logger.exception("Falha ao reproduzir áudio de '%s'.", letter)

    def stop(self):
        self._running = False
