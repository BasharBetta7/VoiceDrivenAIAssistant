from collections import Counter
from pathlib import Path
from datasets import load_dataset, Audio

dataset = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="test",
    streaming=True,
).cast_column("audio", Audio(decode=False))

counts = Counter()
max_speakers = 10
clips_per_speaker = 5
subset_df = []
for sample in dataset:
    speaker = sample["speaker_id"]

    if speaker not in counts and len(counts) >= max_speakers:
        continue
    if counts[speaker] >= clips_per_speaker:
        continue
    subset_df.append(sample)
    folder = Path("data/librispeech_subset") / str(speaker)
    folder.mkdir(parents=True, exist_ok=True)

    audio = sample["audio"]
    filepath = (folder / f"{sample['id']}.wav")
    filepath.write_bytes(audio["bytes"])

    counts[speaker] += 1

    if len(counts) == max_speakers and all(
        n == clips_per_speaker for n in counts.values()
    ):
        break

print(dict(counts))