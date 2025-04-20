# Guide 6: Troubleshooting and Resources

**Navigation:** [Main README]({{ site.baseurl }}/){: .btn .btn-primary} | [Previous Step: Packaging and Sharing](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

This guide provides solutions for common issues encountered during the TTS data preparation, training, and inference process, along with a list of useful tools and resources.

---

## 8. Troubleshooting Common Issues

Refer to this table when you encounter problems. Issues often trace back to data quality or configuration settings.

| Problem Category         | Specific Issue                                      | Possible Causes & Solutions                                                                                                                                                              | Relevant Guide(s)                                                                 |
| :----------------------- | :-------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Data Preparation**   | Script errors during chunking/normalization       | Incorrect file paths; unsupported audio format initially; missing dependencies (`ffmpeg`, `pydub`); extremely noisy/silent audio confusing silence detection. **Check script paths, install dependencies, adjust silence parameters.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md)                                  |
|                          | Manifest generation skips many files                | Mismatched filenames between audio and transcripts; empty transcript files; incorrect paths specified in the script; non-UTF8 encoding in text files. **Verify naming, check paths, ensure text files have content & UTF-8 encoding.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md)                                  |
| **Training Setup**     | `pip install` fails                               | Missing system libraries (e.g., `libsndfile-dev`); incompatible Python version; network issues; conflicts between packages. **Read error messages carefully, install system libs, use virtual env, check framework docs for prerequisites.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md)                                    |
|                          | PyTorch `cuda is not available`                   | Incorrect PyTorch version installed (CPU-only); incompatible NVIDIA driver/CUDA toolkit version; GPU not detected by OS. **Reinstall PyTorch with correct CUDA version from official site, update drivers.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md)                                    |
| **Training Execution** | CUDA Out-of-Memory (OOM) error at start/during train | `batch_size` too large for GPU VRAM; model architecture too complex; memory leak in framework/custom code. **Reduce `batch_size` in config; enable Automatic Mixed Precision (AMP/FP16) if available; check for framework updates.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | Training Loss is `NaN` or diverges (explodes)     | Learning rate too high; unstable gradients; bad data batch (e.g., corrupted audio/text); numerical precision issues. **Lower learning rate; check data quality; use gradient clipping (often enabled by default); try FP32 if using AMP/FP16.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | Training Loss stagnates (doesn't decrease)        | Learning rate too low; poor data quality/variety; model stuck in local minimum; incorrect model configuration. **Increase learning rate slightly; improve/augment data; check config (esp. audio params); try different optimizer.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | Validation Loss increases while Training Loss decreases (Overfitting) | Model memorizing training data; insufficient/unrepresentative validation set; training for too long. **Stop training early (based on best val loss); add more diverse training data; use regularization (weight decay, dropout - check config); improve validation set.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
| **Inference Quality**  | Output sounds robotic/monotonic                   | Insufficient training; poor prosody in training data; model architecture limitations; text normalization issues. **Train longer; improve data variety/quality; try different model architecture; ensure text is punctuated/normalized well.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md), [4_INFERENCE.md](./4_INFERENCE.md) |
|                          | Output is noisy/garbled/unintelligible            | Bad data quality (noise baked in); model didn't converge; mismatch between training config and inference config/checkpoint; incorrect sampling rate used in inference. **Clean training data rigorously; train longer; ensure EXACT config/checkpoint match; verify audio parameters.** | All Guides                                                                        |
|                          | Output sounds like the wrong speaker (fine-tuning) | Pre-trained model not loaded correctly; learning rate too high initially; insufficient fine-tuning data/steps; speaker ID mismatch. **Verify `pretrained_model_path` and `ignore_layers` in config; use lower LR for fine-tuning; train longer; check speaker ID.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md), [4_INFERENCE.md](./4_INFERENCE.md) |
|                          | Inference cuts off early or speaks too fast/slow  | Model limitation (duration prediction); inference setting limiting max output length; length scale/speed parameter incorrect. **Check framework docs for max decoder steps / max length settings; adjust speed control parameters.** | [4_INFERENCE.md](./4_INFERENCE.md)                                                |
| **Model Usage**        | Cannot load checkpoint file                       | Corrupted download/file; using checkpoint with incompatible framework version or config file; incorrect file path. **Re-download/verify file integrity; use the correct config; ensure framework version matches the one used for training; check path.** | [5_PACKAGING_AND_SHARING.md](./5_PACKAGING_AND_SHARING.md), [4_INFERENCE.md](./4_INFERENCE.md) |

---

## 10. Useful Resources & Tools

This list includes software, libraries, and communities helpful for TTS projects.

### Audio Processing & Analysis:

*   **[Audacity](https://www.audacityteam.org/):** Free, open-source, cross-platform audio editor. Excellent for manual inspection, cleaning, labeling, and basic processing of audio files.
*   **[FFmpeg](https://ffmpeg.org/):** The swiss-army knife command-line tool for audio/video conversion, resampling, channel manipulation, format changes, and much more. Essential for scripting batch operations.
*   **[SoX (Sound eXchange Compiled)](http://sox.sourceforge.net/) or [Sox - Source Code](https://codeberg.org/sox_ng/sox_ng/):** Command-line utility for audio processing. Useful for effects, format conversion, and getting audio information (`soxi` command).
*   **[pydub](https://github.com/jiaaro/pydub):** Python library for easy audio manipulation (slicing, format conversion, volume adjustment, silence detection). Uses ffmpeg/libav backend.
*   **[librosa](https://librosa.org/doc/latest/index.html):** Python library for advanced audio analysis, feature extraction (like Mel spectrograms), and visualization. Often used internally by TTS frameworks.
*   **[soundfile](https://python-soundfile.readthedocs.io/en/latest/):** Python library for reading/writing audio files, based on libsndfile. Supports many formats.
*   **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm):** Python library for loudness normalization (LUFS), generally preferred over simple peak normalization for perceived consistency.

### Transcription (ASR):

*   **[OpenAI Whisper](https://github.com/openai/whisper):** High-quality open-source ASR model, supports many languages. Good baseline, but punctuation often needs review. Can run locally (GPU recommended) or via API. Various community implementations exist.
*   **[Google Gemini Models (via API/AI Studio)](https://ai.google.dev/):** Capable models for transcription, often perform well on clear audio, potentially best on pre-chunked segments. Check API/Studio for current capabilities and pricing/free tiers.
*   **Cloud ASR Services:**
    *   [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
    *   [AWS Transcribe](https://aws.amazon.com/transcribe/)
    *   [Azure Speech Service](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/)
    *   *Often reliable, pay-as-you-go, may have initial free quotas.*
*   **[Hugging Face Transformers - ASR Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition):** Hub for many pre-trained ASR models, including fine-tuned versions of Whisper and others. Explore models fine-tuned for specific languages or punctuation improvement.
*   **[ElevenLabs Speech To Text (Scribe)](https://elevenlabs.io/speech-to-text):** *Commercial Service.* Known for very high accuracy in both transcription and punctuation, but is a paid service and can be relatively expensive compared to others. Worth considering if budget allows and maximum out-of-the-box accuracy is required.

### TTS Frameworks & Codebases (Examples - Check for active forks/successors):

*   **[StyleTTS2 (Research Repo)](https://github.com/yl4579/StyleTTS2):** Influential work on style control. Look for actively maintained forks that have implemented training/inference pipelines.
*   **[VITS (Research Repo)](https://github.com/jaywalnut310/vits):** Popular end-to-end architecture. Many forks and implementations exist.
*   **[Coqui TTS (Archived)](https://github.com/coqui-ai/TTS):** Was a very popular and comprehensive library. Although archived, its codebase and concepts remain influential. Many active forks might exist.
*   **[ESPnet](https://github.com/espnet/espnet):** End-to-end speech processing toolkit, including TTS recipes for various models. More research-oriented.
*   **Search GitHub:** Use keywords like "TTS", "VITS training", "StyleTTS2 training", "PyTorch TTS" to find current projects.

### Python Environment & Deep Learning:

*   **[Python](https://www.python.org/):** The core programming language.
*   **[PyTorch](https://pytorch.org/):** The primary deep learning library used by most modern TTS frameworks.
*   **[TensorBoard](https://www.tensorflow.org/tensorboard):** Essential for visualizing training progress (works with PyTorch too).
*   **[pip](https://pip.pypa.io/en/stable/) / [uv](https://github.com/astral-sh/uv):** Python package installers. `uv` is a newer, often much faster alternative.
*   **[conda](https://docs.conda.io/en/latest/) / [venv](https://docs.python.org/3/library/venv.html):** Tools for creating isolated Python environments.
*   **[Git](https://git-scm.com/):** Version control system, essential for cloning repositories and managing code.
*   **[Hugging Face Hub](https://huggingface.co/):** Platform for sharing models (including TTS), datasets, and code.

### Communities:

*   **TTS Framework GitHub Discussions/Issues:** Check the specific repository you are using for community questions and answers.
*   **Discord Servers:** Many AI/ML communities (like LAION, EleutherAI, specific framework servers) have channels dedicated to TTS.
*   **Reddit:** Subreddits like r/SpeechSynthesis, r/MachineLearning.

---

This concludes the main series of guides. Remember that building good TTS models often involves iteration – revisiting data preparation or adjusting training parameters based on results is common practice. Good luck!

---
**Navigation:** [Main README]({{ site.baseurl }}/){: .btn .btn-primary} | [Previous Step: Packaging and Sharing](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | [Back to Top](#top){: .btn .btn-primary}