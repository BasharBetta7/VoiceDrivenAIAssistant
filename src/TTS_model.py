import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel
import logging 
import os
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


DEFAULTS = {
    "model_name":"Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "device":"cuda:0",
    "dtype":torch.bfloat16,
    "attn_implementation":"sdpa",
    "ref_audio_path":"./audio_wavs/tts_ref_audio.wav",
    "ref_text":"Hello there! This is a sample test.",
}

class TTSModel:
    def __init__(self, config:dict|None = None):
        if config is None:
            self.model = Qwen3TTSModel.from_pretrained(
                DEFAULTS['model_name'],
                device_map=DEFAULTS['device'],
                dtype=DEFAULTS['dtype'],
                attn_implementation=DEFAULTS['attn_implementation'],
            )
            self.ref_audio = DEFAULTS['ref_audio_path']
            self.ref_text = DEFAULTS['ref_text']
        else:
            self.model = Qwen3TTSModel.from_pretrained(
                config['model_name'],
                device_map=config['device'],
                dtype=config['dtype']
            )
            self.ref_audio =config['ref_audio_path']
            self.ref_text = config['ref_text']
        logging.info("Registering Agent voice from referencce audio")
        self.voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=self.ref_audio,
            ref_text=self.ref_text
        )

    def speak(self, text:str, language:str = 'English'):
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
           voice_clone_prompt=self.voice_clone_prompt,
        )

        sd.play(wavs[0], sr)
        sd.wait()
    


if __name__ == '__main__':
    load_dotenv()
    tts_model =TTSModel()
    tts_model.speak("Oh hello again. How can I assist you?")
    


