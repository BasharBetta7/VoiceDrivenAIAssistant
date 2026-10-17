import sounddevice as sd
from scipy.io.wavfile import write

sample_rate = 24000
duration = 5

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="int16",
)
sd.wait()

write("recording.wav", sample_rate, audio)