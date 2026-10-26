import speech_recognition as sr
import numpy as np
import torch


def convert_audio_to_numpy(audio:sr.AudioSource, sr=16000) -> tuple:
    """return (np.ndarray, sr)"""
    raw_audio = audio.get_raw_data(
        convert_rate=16000,
        convert_width=2,
    )
    wav= np.frombuffer(raw_audio,np.int16)[None,:]
    return wav, sr


def convert_audio_to_tensor(audio:sr.AudioSource, sr=16000) -> torch.Tensor:
    raw_audio  =audio.get_raw_data(
        convert_rate=16000,
        convert_width=2
    )


    audio_tensor = torch.from_numpy(
        np.frombuffer(raw_audio, np.int16).copy().astype(np.float32)
    )[None,:]
    return audio_tensor

