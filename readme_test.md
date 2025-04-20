# Universal TTS Fine‑tuning & Dataset Prep Guide

## Introduction
This guide covers **everything you need** to:
- Prepare your own dataset (audio + text pairs)
- Train a **Generic TTS** model 
- Organize your files properly

Works for **small** (10h) to **large** (100h) datasets.

**Links to Tools & Repos are included.**

---

## 1. Dataset Preparation

### 1.1. Recording / Collecting Audio

- Collect **clean audio** (radio, podcasts, audiobooks, etc.).
- **Target**: Single speaker, minimal background noise.
- **Audio format**: WAV, 16kHz or 22kHz preferred.

> Tip: If it's long recordings (1hr+), chunk it into smaller clips.

### 1.2. Automatic Chunking (split into ~10s clips)

Use [pydub](https://github.com/jiaaro/pydub) to auto-chunk:

```bash
pip install pydub
```

```python
from pydub import AudioSegment
from pydub.silence import split_on_silence

sound = AudioSegment.from_file("input.wav", format="wav")
chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=-40)

for i, chunk in enumerate(chunks):
    chunk.export(f"chunk_{i}.wav", format="wav")
```

- **Parameters**:
  - `min_silence_len=500`: Minimum 500ms silence to split.
  - `silence_thresh=-40`: Threshold for silence detection (dBFS).


### 1.3. Volume Normalization

Normalize volume across all clips:

```python
from pydub import AudioSegment

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

sound = AudioSegment.from_file("chunk.wav", format="wav")
normalized_sound = match_target_amplitude(sound, -20.0)
normalized_sound.export("normalized_chunk.wav", format="wav")
```

- Recommended target: **-20 dBFS**.


### 1.4. Transcription (Text Pair Creation)

- Use [Gemini 2.5 Pro](https://gemini.google.com) (or [Whisper](https://github.com/openai/whisper)) for auto-transcribing each clip.
- **One transcription per clip**.

Organize like:
```bash
/audio/
   chunk_0.wav
   chunk_1.wav
/texts/
   chunk_0.txt
   chunk_1.txt
```

Each `.txt` contains clean text transcription for that `.wav`.

> **Important**: Remove timestamps, filler words ("uh", "um") if possible.


### 1.5. Manifest File Creation

Format needed for Generic TTS training:
```text
path/to/audio.wav|transcription|speaker_id
```

Example (for single speaker):
```text
audio/chunk_0.wav|Hello, welcome to Lagos.|spk1
audio/chunk_1.wav|The market opens at dawn.|spk1
```

- Save as `train_list.txt` and `val_list.txt` (90/10 split).

**Quick Python manifest generator:**
```python
import os

manifest = []

for filename in sorted(os.listdir("audio")):
    if filename.endswith(".wav"):
        audio_path = f"audio/{filename}"
        text_path = f"texts/{filename.replace('.wav', '.txt')}"
        
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        manifest.append(f"{audio_path}|{text}|spk1")

with open("train_list.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(manifest))
```

---

## 2. Data Quality Requirements

| Aspect                  | Best Practice                          |
|--------------------------|----------------------------------------|
| Audio length             | 2-10 seconds per clip                 |
| Background noise         | Low / None                             |
| Speakers per clip        | Only one                              |
| Audio format             | WAV, PCM 16-bit preferred             |
| Volume                   | Normalized to around -20 dBFS          |
| Text transcription       | Clean, without special characters     |
| Matching naming          | Filename.wav + Filename.txt            |

---

## 3. Training Setup

### 3.1. Environment Setup

Clone your TTS framework of choice:

```bash
git clone <YOUR_TTS_REPO_URL>
cd <YOUR_TTS_DIR>
```

Install requirements:
```bash
pip install -r requirements.txt # ensure you have CUDA‑enabled PyTorch if needed
```

> Note: You need **PyTorch** with GPU (CUDA) support.
> [Install PyTorch](https://pytorch.org/get-started/locally/) properly based on your GPU.<br>
> **Use** `uv` as it's faster than `pip` for installing packages.

---

### 3.2. Folder Structure Example

```bash
Generic-TTS/
├── train_list.txt
├── val_list.txt
├── audio/ (your wav files)
├── checkpoints/
└── configs/
```

- `train_list.txt`, `val_list.txt` = your manifests.
- `audio/` = your clipped and normalized WAVs.
- `configs/` = where your config YAML lives.
- `checkpoints/` = where models will save.


---

## 4. Configuring Training

Copy an example config from `configs/`:
# generic training entrypoint
```bash
cp configs/config_ft_small.yaml my_config.yaml
```

Edit it:
```yaml
train_manifest: "train_list.txt"     # rename if your framework uses e.g. `train_manifest`
validation_manifest: "val_list.txt"  # adjust key names as needed
audio_dir: "audio"
output_dir: "checkpoints"            # might be `output_dir`, `model_dir`, etc.
epochs: 500                   # adjust based on your dataset size
batch_size: 16               # adjust based on your GPU memory
learning_rate: 1e-4
sampling_rate: 22050

# You can tune parameters like epochs/batch size based on your GPU.
# For example, if you have a 24GB GPU, you can increase batch size to 32 or 64.
# For small datasets (<10h), 500 epochs is usually enough.  
# For large datasets (>50h), you might want more epochs (~1000).  
# Ensure to monitor training loss and validation metrics closely to avoid overfitting. 
```

---

## 5. Starting Training

Basic command:
```bash
python train.py --config my_config.yaml
```

Training will start and checkpoint models will be saved every few epochs into `checkpoints/`.

> Tip: Save your best model based on lowest validation loss.


---

## 6. Notes on Fine-tuning

- **Multi-speaker training**:
  - Make sure your manifest uses different `speaker_id` for each voice.

- **Resume training**:
  - If you crash, you can resume from last checkpoint:
    ```bash
    python train.py --config my_config.yaml --resume checkpoints/ckpt_XXX.pth
    ```

- **FP16/Automatic Mixed Precision**:
  - If you have VRAM issues, enable AMP for faster and lighter training.


---

## 7. Inference (Testing Your Model)

After training, generate speech from your model.

Example script:
```bash
python inference.py --checkpoint checkpoints/ckpt_XXX.pth --text "Hello world." --output generated.wav
```

You can also build a simple GUI or Colab notebook to try different voices/texts.


---

## 8. Useful Resources

- [Official StyleTTS2 repo](https://github.com/yl4579/StyleTTS2)
- [StyleTTS2 Discussions (Training Tips)](https://github.com/yl4579/StyleTTS2/discussions)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [pydub Documentation](https://github.com/jiaaro/pydub)


---

You now know how to:
- Prepare your own dataset
- Fine-tune Generic TTS - most of the TTS on new languages or voices
- Train & test your custom models 🚀

---

> **Pro Tip**: Save your `README.md` inside your dataset folder for future reference.


## 9. Running Inference (Generate Audio from Your Model)

After you finish training, here’s how to **use your Generic TTS model** to synthesize speech.

---

### 9.1. Basic Single Sentence Inference

```bash
python inference.py \
  --checkpoint checkpoints/ckpt_best.pth \
  --text "Hello, welcome to the world of TTS." \
  --output output.wav
```

**Parameters:**
- `--checkpoint`: Path to your trained model.
- `--text`: The text you want to synthesize.
- `--output`: Where to save the generated WAV.

> Tip: Try different sentences to check model stability and style.

---

### 9.2. Batch Inference (Multiple Sentences)

Prepare a text file, `sentences.txt`:
```text
Hello, how are you?
Today is a beautiful day.
This is a custom TTS model.
```

Then run:
```bash
python inference_batch.py \
  --checkpoint checkpoints/ckpt_best.pth \
  --input sentences.txt \
  --output_dir generated/
```

Each line will be turned into a WAV file in `generated/` folder.

---

### 9.3. Using Speaker Embeddings (Optional Advanced)

If you trained a **multi-speaker** for your Generic TTS model, you can specify speaker embeddings.

Example (pseudo-code):
```python
synthesizer.infer(text="Hello", speaker_id="spk1")
```

> **Single speaker training?** No need to specify, defaults automatically.

---

## 10. Quick Example: Minimal API for Synthesis

You can build a tiny Python script for easy synthesis:

```python
from your_tts_model import Synthesizer

synth = Synthesizer(checkpoint_path="checkpoints/ckpt_best.pth")

output = synth.synthesize("This is a demo.")
synth.save_wav(output, "demo.wav")
```

- Wrap it into a web server, simple GUI, or even a Discord bot easily.


---

## 11. Troubleshooting Common Issues

| Problem | Solution |
|:---|:---|
| Output sounds robotic/noisy | Check data quality; normalize audio and clean transcripts better |
| Output cuts off early | Increase max audio generation time settings |
| Model overfits (memorizes training) | Use more diverse data, validate properly |
| CUDA OOM during inference | Reduce batch size or use FP16 (half precision) |

---

## 12. Bonus: Style/Emotion Control (Generic TTS Extras)

Depending on Generic TTS's configs, you might also control:
- Speaking **speed**
- **Emotion/Style tokens**
- **Diffusion steps** for better quality

Example if available:
```python
synth.synthesize(text, speed=1.1, style_token="happy")
```

> **(⚡ Bonus)** Some forks also support "zero-shot" style transfer by using a reference audio clip!


---


## 📦 14. Packaging Your Trained Model

Once you have a trained TTS model, it's important to **package it properly** so:
- You can use it later without confusion.
- Others (or your future self) can understand and run it easily.
- It stays compatible with different TTS engines.

---

## 14.1 Organize Your Model Files

Create a clean folder structure for each trained model version.

Example:
```bash
my_model_v1/
├── checkpoints/
│   ├── ckpt_best.pth
│   ├── ckpt_last.pth
├── config.yaml
├── training_info.txt
├── README.md
├── samples/
│   ├── sample_hello.wav
│   ├── sample_paragraph.wav
```

**What's inside:**
- `checkpoints/`: All your saved model weights (best and last checkpoints).
- `config.yaml`: The exact config used for training (hyperparameters, paths, etc.).
- `training_info.txt`: Info like dataset size, training duration, final loss, notes.
- `README.md`: A short human-readable explanation.
- `samples/`: A few example audio clips synthesized by your model.

> **Tip:** Always keep the config and checkpoints together!


---

## 14.2 Writing a Good README.md

Minimal template you should include:

```markdown
# My TTS Model - v1.0

## About
- Trained on 50 hours of Nigerian Yoruba radio speech.
- Single speaker, clean studio recordings.

## Training Details
- Training time: 3 days on RTX 4090.
- Final validation loss: 0.78.
- Style modeling: Enabled.

## How to Use
1. Load checkpoint `checkpoints/ckpt_best.pth`
2. Use config `config.yaml`
3. Sampling rate: 22,050 Hz
4. Single speaker model (no speaker embeddings needed).

## Samples
- [Hello sample](samples/sample_hello.wav)
- [Paragraph sample](samples/sample_paragraph.wav)
```

**Optional Extras:**
- List of any style tokens/emotions trained.
- Speaker ID or speaker embedding info if multi-speaker.

---

## 14.3 Model Versioning Tips

When you retrain or fine-tune:
- Use folder names like `my_model_v2/`, `my_model_v3/`.
- Update the README with changes.
- Keep old versions — sometimes older models sound better!

> Treat your TTS models like software releases.

---

## 14.4 Sharing Models

If you want to distribute or archive your model:
- Zip the full model folder.
- Share via:
  - Google Drive / Dropbox
  - HuggingFace Spaces/Models
  - Private GitHub Releases (for small models)

**Important:**
- Follow any dataset licensing rules if you trained on public data.
- Mention if the model is for non-commercial or research-only use.

---

# 🎯 Summary

- Always save checkpoints, configs, samples, and documentation together.
- Write a short README so anyone can load and test easily.
- Version your models clearly to avoid confusion.


➡️ Your model is now ready to be used, tested, or shared with the world! 🚀
