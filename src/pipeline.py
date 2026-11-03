from src.speaker_classifier import SpeakerClassifier
from src.llm_model import LLMChat
from src.asr_model import SpeechRecognizer
from src.events import AgentEvent, EventHandler

import speech_recognition as sr
import asyncio
import logging
from src.utils import convert_audio_to_numpy, convert_audio_to_tensor


class VoiceAssistant:
    def __init__(self, config:dict|None = None, llm=None):
        config = config or {}
        self.speaker_classifier = SpeakerClassifier(config.get('speaker_classifier'))
        self.asr_model = SpeechRecognizer(config.get('speech_recognizer'))
        self.llm = llm if llm is not None else LLMChat(config.get('llm_model'))
        self.speaker_classifier.enroll_speaker(n_trials=1, duration=10)
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold= 1.0 #1 second pause
        self.mic = sr.Microphone(sample_rate=16000)

    def _capture_command(self):
        # Open the microphone only for this command, with bounded waiting/recording.
        with self.mic as source:
            return self.recognizer.listen(source, timeout=5, phrase_time_limit=30)

    async def transcribe_once(self, on_event: EventHandler | None = None):
        """Capture and verify one phrase. Return None when no prompt is accepted."""
        def emit(kind, text):
            if on_event is not None:
                on_event({"type": kind, "data": {"text": text}})

        emit("status", "Listening... speak now (up to 30 seconds).")
        try:
            audio = await asyncio.to_thread(self._capture_command)
        except sr.WaitTimeoutError:
            emit("error_message", "No speech detected. Press F2 to try again.")
            return None

        emit("status", "Verifying speaker...")
        authorized = await asyncio.to_thread(
            self.speaker_classifier.authorized_speaker, convert_audio_to_tensor(audio)
        )
        if not authorized:
            emit("error_message", "Sorry, I didn't recognize your voice as authorized.")
            return None

        emit("status", "Transcribing...")
        text = (await asyncio.to_thread(
            self.asr_model.transcribe, convert_audio_to_numpy(audio)
        )).strip()
        if not text:
            emit("error_message", "No text recognized. Press F2 to try again.")
            return None
        emit("user_prompt", text)
        return text


   
    def listen(self):
        # One event loop for the entire session, not one loop per utterance.
        asyncio.run(self._listen_async())

    async def _listen_async(self, on_event :EventHandler|None = None):
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue(maxsize=8)
        def emit(event:AgentEvent|None = None) -> None:
            if on_event is not None:
                on_event(event)

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
                        emit({
                            "type":"user_prompt",
                            "data":{
                                "text":text,
                            }
                        })
                        #print(f"YOU SAID: {text}")
                        await self.llm.llm_chat(text)
                    else:
                        emit({
                            "type":"error_message",
                            "data":{"text":"Sorry, I didn't recognize your voice as authorized."}
                        })
                        #print("UNAUTHORIZED SPEAKER!")
                except Exception:
                    logging.exception("Error while handling audio")
                finally:
                    audio_queue.task_done()
        finally:
            stop_listening(wait_for_stop=True)
            


if __name__ == '__main__':
    from preuse.prompt_toolkit_test import main
    asyncio.run(main())
    
