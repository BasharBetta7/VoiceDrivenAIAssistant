from qwen_asr import Qwen3ASRModel
import torch
import numpy as np
from src.speaker_classifier import record_audio
import src

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)

DEFAULT_ASR_MODEL = 'Qwen/Qwen3-ASR-0.6B'

class SpeechRecognizer:
    def __init__(self, config:dict|None = None):
        if config is not None:
            self.model_name = config.get('model_name', DEFAULT_ASR_MODEL)
        else:
            self.model_name = DEFAULT_ASR_MODEL
        
        self.model = Qwen3ASRModel.from_pretrained(
                        self.model_name,
                        dtype=torch.bfloat16,
                        device_map="cuda:0",
                        # attn_implementation="flash_attention_2",
                        max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
                        max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
        )

                
    def __call__(self,audio:np.array ):
        return self.transcribe(audio)
    
    def transcribe(self, audio, language:str = 'English'):
        
        return self.model.transcribe(audio, language=language)[0].text
    

if __name__ =='__main__':
    asr_recognizer = SpeechRecognizer()
    input("when you're ready, press Enter and speak for 3 seconds")
    audio = record_audio(16000, 3)
    print(asr_recognizer(audio))
    
        