# Guide 4: Inference - Generating Speech with Your Model

**Navigation:** [Main README](/Universal-TTS-Guide/) | [Previous Step: Model Training](./3_MODEL_TRAINING.md) | [Next Step: Packaging and Sharing](./5_PACKAGING_AND_SHARING.md)

You have successfully trained or fine-tuned a TTS model and selected a promising checkpoint! Now, let's use that model to convert new text into speech audio – a process called **inference** or **synthesis**.

---

## 7. Inference: Synthesizing Speech

This section explains how to run the inference process using your trained model.

### 7.1. Locate the Inference Script and Best Checkpoint

-   **Inference Script:** Find the Python script within your TTS framework designed for generating audio. Common names include `inference.py`, `synthesize.py`, `infer.py`, `tts.py`.
-   **Best Checkpoint:** Identify the path to the model checkpoint (`.pth`, `.pt`, `.ckpt`) you want to use. This is typically the one saved as `best_model.pth` (based on validation loss) or another checkpoint you selected based on listening to validation samples during training. It will be located within your training output directory (e.g., `../checkpoints/my_yoruba_voice_run1/best_model.pth`).
-   **Configuration File:** You will almost always need the configuration file (`.yaml`, `.json`) that was used *during the training* of the checkpoint you are using. The inference script needs this to know the model architecture, audio parameters (like sampling rate), and other settings. Often, a copy of the config is saved alongside the checkpoints.

### 7.2. Basic Single Sentence Inference

-   **Goal:** Generate audio for a single piece of text provided directly via the command line.
-   **Command Structure:** The exact arguments will vary, but a typical command looks like this:

    ```bash
    # Activate your virtual environment first!
    # Example command:
    python inference.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --text "Hello, this is a test of my custom trained voice." \
      --output_wav_path ./output_sample.wav
      # Optional/Framework-dependent arguments below:
      # --speaker_id "main_speaker"  # Needed for multi-speaker models
      # --device "cuda"              # To specify GPU usage (often default)
    ```
-   **Key Arguments:**
    *   `--config` or `-c`: Path to the training configuration file.
    *   `--checkpoint_path` or `--model_path` or `-m`: Path to the model checkpoint file.
    *   `--text` or `-t`: The input sentence you want to synthesize. Remember to enclose it in quotes.
    *   `--output_wav_path` or `--out_path` or `-o`: The desired path and filename for the generated WAV file.
    *   `--speaker_id` or `--spk`: **Required** if you trained a multi-speaker model. Provide the exact speaker ID used in your manifest files for the desired voice. For single-speaker models, this might be optional or ignored.
    *   `--device`: Often optional, defaults to `cuda` if available, otherwise `cpu`. Inference is much faster on GPU.

-   **Execution:** Run the command. It will load the model, process the text, generate the audio waveform, and save it to the specified output file. Listen to the output WAV file to check the quality.

### 7.3. Batch Inference (Synthesizing from a File)

-   **Goal:** Generate audio for multiple sentences listed in a text file, saving each as a separate WAV file.
-   **Prepare Input File:** Create a plain text file (e.g., `sentences.txt`) where each line contains one sentence you want to synthesize:
    ```text
    This is the first sentence.
    Here is another sentence to synthesize.
    The model should handle different punctuation marks, like questions?
    And also exclamations!
    ```
-   **Command Structure:** Many frameworks provide a separate script or specific arguments for batch processing.

    ```bash
    # Example command (script name and arguments may vary):
    python inference_batch.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      КОНФИГУРАЦИЯ: ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --input_file sentences.txt \
      --output_dir ./generated_batch_audio/
      # Optional/Framework-dependent arguments below:
      # --speaker_id "main_speaker"  # Needed for multi-speaker models
      # --device "cuda"
    ```
-   **Key Arguments:**
    *   `--input_file` or `--text_file`: Path to the text file containing sentences (one per line).
    *   `--output_dir` or `--out_dir`: Path to the directory where the generated WAV files should be saved. Ensure this directory exists or the script creates it. Output filenames are often based on the line number or the input text itself (e.g., `output_0.wav`, `output_1.wav`).
    *   Other arguments (`--config`, `--checkpoint_path`, `--speaker_id`, `--device`) are typically the same as for single sentence inference.

-   **Execution:** Run the command. The script will iterate through each line in the input file, synthesize the audio, and save the results in the specified output directory.

### 7.4. Multi-Speaker Model Inference

-   As mentioned above, if your model was trained on data from multiple speakers, you **must** specify which speaker's voice you want to use during inference.
-   Use the `--speaker_id` (or equivalent) argument, providing the exact ID that corresponds to the desired speaker in your training manifest files (e.g., `speaker0`, `mary_smith`, `yoruba_male_spk1`).
-   If you omit the speaker ID for a multi-speaker model, the script might fail, default to a specific speaker (often speaker 0), or produce averaged/garbled results.

### 7.5. Advanced Inference Controls (Framework Dependent)

-   Some advanced TTS models and frameworks offer additional controls during inference, often passed as command-line arguments or parameters in a Python API:
    *   **Speaking Rate/Speed:** Arguments like `--speed` or `--length_scale` might allow you to make the voice speak faster or slower (e.g., `1.0` is normal, `<1.0` is faster, `>1.0` is slower).
    *   **Pitch Control:** Less common, but some models might allow pitch adjustments.
    *   **Style/Emotion Control:** If the model was trained with style tokens or reference audio capabilities (like StyleTTS2 or models with style embeddings), you might provide arguments like `--style_text` or `--style_wav` to influence the output prosody or emotion.
    *   **Vocoder Settings (if applicable):** For older Tacotron2-style models or others using separate vocoder models (like HiFi-GAN, MelGAN), there might be settings related to the vocoder (e.g., denoising strength).
    *   **Diffusion Models:** For diffusion-based TTS models, parameters controlling the number of diffusion steps (trading quality for speed) might be available.
-   **Consult Documentation:** Always refer to your specific TTS framework's documentation or inference script help (`python inference.py --help`) to see which controls are available.

### 7.6. Potential Inference Issues

-   **CUDA Out-of-Memory (OOM):** Even if training worked, very long sentences during inference might consume more memory. Try shorter sentences or see if the framework offers options for segmented synthesis. Running on CPU (`--device cpu`) uses system RAM but is significantly slower.
-   **Model/Config Mismatch:** Using a checkpoint with the wrong configuration file is a common error, leading to loading failures or garbage output. Ensure they correspond to the same training run.
-   **Incorrect Speaker ID:** Providing a non-existent speaker ID for multi-speaker models will cause errors.
-   **Quality Issues (Noise, Instability):** If the output quality is poor, revisit Guide 1 (Data Preparation) and Guide 3 (Model Training). It might indicate issues with data quality, insufficient training, or choosing a suboptimal checkpoint.

---

Now that you can generate speech using your trained model, the next logical step is to organize your model files properly for future use, sharing, or deployment.

**Next Step:** [Packaging and Sharing](./5_PACKAGING_AND_SHARING.md)