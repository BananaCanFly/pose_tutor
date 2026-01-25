# import pyttsx3
#
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)  # 语速
# engine.setProperty('volume', 1.0)  # 音量
#
# def speak_suggestion(text):
#     engine.say(text)
#     engine.runAndWait()
#
#
# speak_suggestion("胳膊向上移动")

import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import queue
import json

MODEL_PATH = "models/vosk-model-small-cn-0.22"
SAMPLE_RATE = 16000

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

with sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=8000,
    dtype='int16',
    channels=1,
    callback=audio_callback
):
    print("🎙 开始监听，说话试试...")

    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                print("识别结果:", text)
