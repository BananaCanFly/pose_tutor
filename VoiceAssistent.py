# VoiceAssistant.py
import queue
import json
import time
import threading

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3


class VoiceAssistant:
    def __init__(
        self,
        model_path: str,
        get_suggestion_func,
        sample_rate: int = 16000,
        cooldown: float = 5.0,
    ):
        """
        model_path: vosk 模型路径
        get_suggestion_func: 回调函数，返回当前「主要姿势建议」
        cooldown: 两次播报最小间隔（秒）
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.get_suggestion = get_suggestion_func

        self.q = queue.Queue()
        self.last_speak_time = 0
        self.running = False

        self.tts_queue = queue.Queue()
        self.tts_lock = threading.Lock()

        # 关键词（你可以随时改）
        self.trigger_words = [
            "建议",
            "姿势",
            "怎么",
            "调整",
            "我该",
            "现在",
        ]

        # TTS
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)

        # Vosk
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

    # ================= 音频回调 =================
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.q.put(bytes(indata))

    def _tts_loop(self):
        while self.running:
            text = self.tts_queue.get()
            if text is None:
                continue
            with self.tts_lock:
                self.engine.say(text)
                self.engine.runAndWait()

    # ================= 关键词判断 =================
    def _is_trigger(self, text: str) -> bool:
        return any(word in text for word in self.trigger_words)

    # ================= 语音播报 =================
    def _speak(self, text):
        self.tts_queue.put(text)

    # ================= 主监听循环 =================
    def _listen_loop(self):
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            print("🎙 语音助手已启动（Vosk 本地识别）")

            while self.running:
                data = self.q.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()

                    if not text:
                        continue

                    print("🎤 识别到：", text)

                    if not self._is_trigger(text):
                        continue

                    now = time.time()
                    if now - self.last_speak_time < self.cooldown:
                        continue

                    suggestion = self.get_suggestion()
                    if suggestion:
                        self._speak(suggestion)
                        self.last_speak_time = now

    # ================= 对外接口 =================
    def start(self):
        if self.running:
            return
        self.running = True

        threading.Thread(target=self._listen_loop, daemon=True).start()
        threading.Thread(target=self._tts_loop, daemon=True).start()

    def stop(self):
        self.running = False
