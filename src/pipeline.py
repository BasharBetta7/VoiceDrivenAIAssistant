from src.speaker_classifier import SpeakerClassifier
from src.llm_model import LLMChat
from src.asr_model import SpeechRecognizer

import numpy as np
import torch
import speech_recognition as sr
import asyncio
import logging
from src.utils import *


class VoiceAssistant:
    def __init__(self, config:dict|None = None):
        self.speaker_classifier = SpeakerClassifier(config['speaker_classifier'])
        self.asr_model = SpeechRecognizer(config['speech_recognizer'])
        self.llm = LLMChat(config['llm_model'])
        self.speaker_classifier.enroll_speaker(n_trials=1, duration=10)
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold= 1.0 #1 second pause
        self.mic = sr.Microphone(sample_rate=16000)


   
    def listen(self):
        # One event loop for the entire session, not one loop per utterance.
        asyncio.run(self._listen_async())

    async def _listen_async(self):
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue(maxsize=8)

        def enqueue(audio):
            if audio_queue.full():
                logging.warning("Audio queue is full; dropping the newest phrase")
            else:
                audio_queue.put_nowait(audio)

        def callback(recognizer:sr.Recognizer, audio):
            # SpeechRecognition calls this from its microphone thread.
            # Queue the audio; all asynchronous clients stay on the main loop.
            loop.call_soon_threadsafe(enqueue, audio)

        stop_listening = self.recognizer.listen_in_background(self.mic, callback, phrase_time_limit=None)
        try:
            while True:
                audio = await audio_queue.get()
                try:
                    authorized = await asyncio.to_thread(
                        self.speaker_classifier.authorized_speaker,
                        convert_audio_to_tensor(audio),
                    )
                    if authorized:
                        text = await asyncio.to_thread(
                            self.asr_model.transcribe, convert_audio_to_numpy(audio)
                        )
                        print(f"YOU SAID: {text}")
                        await self.llm.llm_chat(text)
                    else:
                        print("UNAUTHORIZED SPEAKER!")
                except Exception:
                    logging.exception("Error while handling audio")
                finally:
                    audio_queue.task_done()
        finally:
            stop_listening(wait_for_stop=True)
            


if __name__ == '__main__':
    import yaml
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)
    assistant = VoiceAssistant(config)
    assistant.listen()
    
