from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from torchaudio.functional import resample

def evaluate_model(params=None):
    # Sorting keeps recordings from the same speaker next to one another.
    audio_files = sorted(
        path for path in Path("data/librispeech_subset").glob("*/*")
        if path.suffix.lower() in {".wav", ".flac"}
    )
    if len(audio_files) != 50:
        raise ValueError(f"Expected 50 audio files, found {len(audio_files)}")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    embedding_list = []
    with torch.inference_mode():
        for path in audio_files:
            # SoundFile detects the actual encoding, even for FLAC bytes named .wav.
            samples, sr = sf.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(samples.T.copy()).mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = resample(waveform, sr, 16000)
            if waveform.numel() == 0 or not torch.isfinite(waveform).all():
                raise ValueError(f"Empty or non-finite audio: {path}")
            embedding = classifier.encode_batch(waveform, normalize=False)
            embedding_list.append(embedding.reshape(1, -1).cpu())

        embeddings = F.normalize(torch.cat(embedding_list, dim=0), p=2, dim=-1)
        similarity_matrix = embeddings @ embeddings.T  # [50, 50]

    labels = [path.stem for path in audio_files]
    similarity_df = pd.DataFrame(similarity_matrix.numpy(), index=labels, columns=labels)
    similarity_df.index.name = "clip_id"
    output_path = Path("data/speaker_similarity.csv")
    similarity_df.to_csv(output_path)
    print(f"Saved {similarity_matrix.shape} matrix to {output_path}")


    speakers = np.array([str(x).split('-')[0] for x in similarity_df.index])
    same_speaker = speakers[:, None] == speakers[None,:]
    unique_pairs = np.triu(np.ones(similarity_df.shape,dtype=bool), k=1)
    scores = similarity_df.to_numpy()[same_speaker & unique_pairs]

    x = np.linspace(0,1,retstep=0.1)[0]
    recalls = []
    precisions = []
    for threshold in x:
        true_acceptance_rate = (scores > threshold).mean()
        recalls.append(true_acceptance_rate)
        precision = sum(scores > threshold) / sum(similarity_df.to_numpy()[unique_pairs] > threshold)
        precisions.append(precision)

    recalls = np.array(recalls)
    precisions = np.array(precisions)


    import matplotlib.pyplot as plt 
    import seaborn as sns

    plt.figure(figsize=(14,12))

    sns.heatmap(
        data=similarity_df > 0.5,
        cmap='coolwarm',
    )
    plt.xlabel('Audio clip')
    plt.ylabel('Audio clip')
    plt.title('Speaker similarity - 50 clips')
    plt.tight_layout()
    plt.savefig('metrics/speaker_similarity.png')
    plt.show()


    plt.figure(figsize=(12,6))

    plt.plot(x,recalls, label= 'recall')
    plt.plot(x, precisions, label = 'precision' )
    plt.axvspan(0.37, 0.45, color='orange', alpha=0.4, label='efficient region')
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(0,1.1,0.05))
    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    plt.title('Speaker Classifier Performance w.r.t threshold decision')
    plt.tight_layout()
    plt.savefig('metrics/speaker_precision_recall_vs_threshold.png')
    plt.show()
