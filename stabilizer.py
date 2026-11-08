"""Rastreia por quanto tempo a MESMA letra vem sendo detectada de
forma ininterrupta, pra so "confirmar" (e falar) depois de um tempo
minimo estavel - evita que ruido de 1-2 frames dispare o TTS toda hora.

Mesma ideia do MIN_DETECTION_SECONDS / COOLDOWN_SECONDS do projeto do
caixa com visão computacional, adaptada pra letras em vez de produtos.
"""

import time

import config


class LetterStabilizer:
    def __init__(self):
        self._current_letter = None
        self._since = None
        self._last_confirmed_letter = None
        self._last_confirmed_at = 0.0
        self._confirmed_current = False

    def update(self, letter):
        """Recebe a letra do frame atual (ou None se nada detectado
        acima do threshold). Retorna uma tupla:
            (letter_to_show, progress_0_to_1, just_confirmed)
        - letter_to_show: a letra "candidata" atual (ou None)
        - progress_0_to_1: quao perto de 1.5s estavel ela está (pra
          barra de progresso na interface)
        - just_confirmed: True só no frame em que ela acabou de virar
          "confirmada" (é a hora de chamar o TTS)
        """
        now = time.monotonic()

        if letter != self._current_letter:
            self._current_letter = letter
            self._since = now
            self._confirmed_current = False

        if letter is None or self._since is None:
            return None, 0.0, False

        held_for = now - self._since
        progress = min(1.0, held_for / config.MIN_STABLE_SECONDS)

        just_confirmed = False
        if held_for >= config.MIN_STABLE_SECONDS and not self._confirmed_current:
            cooldown_ok = (
                letter != self._last_confirmed_letter
                or (now - self._last_confirmed_at) >= config.COOLDOWN_SECONDS
            )
            if cooldown_ok:
                self._last_confirmed_letter = letter
                self._last_confirmed_at = now
                self._confirmed_current = True
                just_confirmed = True

        return letter, progress, just_confirmed

    def reset(self):
        self._current_letter = None
        self._since = None
        self._confirmed_current = False
