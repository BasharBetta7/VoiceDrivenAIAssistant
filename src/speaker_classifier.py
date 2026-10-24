import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import sounddevice as sd
from scipy.io.wavfile import write
import torch
import torch.nn.functional as F 



DEFAULT_MODEL_PATH = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_THRESHOLD = 0.6


def record_audio(sr, duration) :
    print(f"Recording audio for {duration} seconds...")
    audio = (sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype="int16",
    ), sr)
    sd.wait()
    return audio


class SpeakerClassifier:
    def __init__(self, config:dict|None = None):
        if config is None:
            config = {}
        self.model_path = config.get("model_path", DEFAULT_MODEL_PATH)
        self.spk_threshold = config.get("threshold", DEFAULT_THRESHOLD)
        self.authorized_speaker_embedding = None
        try:
            self.model = EncoderClassifier.from_hparams(source=self.model_path, savedir="pretrained_models/spkrec-ecapa-voxceleb")
        except:
            self.model = EncoderClassifier.from_hparams(DEFAULT_MODEL_PATH)


    def enroll_speaker(self, n_trials:int = 1, duration:int= 5, threshold:float= DEFAULT_THRESHOLD) -> torch.tensor:
        """Enrolls new speaker after listening to multiple speech recordings"""
        if not self.authorized_speaker_embedding is None:
            ans = input(f"A speaker voice has already been verified, Do you want to overwrite it ? (y,n)")
            if ans == 'n':
                return
        print(f"You will be asked to speak {n_trials} times, each time you can say whatever you want for {duration} seconds, the model will save your voiceprint afterwards")
        embeddings = []
        for i in range(n_trials):
            input(f"Press Enter to Start trial {i+1}...")
            print(f"Trial number {i+1} : ")
            audio, sr = record_audio(16000, duration)
            audio = torch.tensor(audio, dtype=torch.float32)
            audio = audio.T
            audio = audio.mean(dim=0, keepdim=True) # in case of multiple channels, ecapa accepts mono audio with sr = 16000
            audio = torchaudio.functional.resample(audio, sr, 16000)
            embeddings.append(self.model.encode_batch(audio, normalize=False))
            if i > 0:
                similarity = F.cosine_similarity(embeddings[-1][0,:,:], embeddings[-2][0,:,:])
                assert similarity.item() > self.spk_threshold, "Your recorded voice is very different from the previous, try again"
        res = torch.cat(embeddings, dim=0)
        res = F.normalize(res, p=2, dim=-1)
        self.authorized_speaker_embedding = F.normalize(res.mean(dim=0), dim=-1)
        print(f"Finished Enrollment Procedure")

    
    def authorized_speaker(self, audio, sr:int = 16000) -> bool:
        """Checks the similarity with the enrolled speaker """
        assert self.authorized_speaker, "No authorized speakers to compare with, please enroll a speaker before verification"
    
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
            if audio.shape[0] != 1:
                audio = audio.T
        audio = audio.mean(dim=0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, 16000)
        audio_dim = self.model.encode_batch(audio, normalize=False)[0]
        audio_dim = F.normalize(audio_dim, p=2, dim=-1)
        sim = F.cosine_similarity(audio_dim, self.authorized_speaker_embedding)
        print(f"cos_sim = {sim.item(): .4f}")
        return sim.item() > self.spk_threshold



if __name__ == '__main__':
    import yaml
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    classifier = SpeakerClassifier(config=config['speaker_classifier'])
    classifier.enroll_speaker()
    audio, sr = record_audio(16000, 3)
    print(classifier.authorized_speaker(audio, sr))
    

