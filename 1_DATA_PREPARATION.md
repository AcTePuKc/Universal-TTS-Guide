# Guide 1: Data Preparation for TTS Training

**Navigation:** [Main README](./README.md) | [Next Step: Training Setup (`2_TRAINING_SETUP.md`)](./2_TRAINING_SETUP.md)

This guide covers the critical first phase of any TTS project: preparing high-quality, correctly formatted audio and text data. The quality of your dataset directly impacts the quality of your final TTS model.

---

## 1. Dataset Preparation Steps

Follow these steps systematically to transform raw audio into a training-ready dataset.

### 1.1. Audio Acquisition & Initial Processing

-   **Gather Audio:** Collect your raw audio files (common formats include WAV, MP3, FLAC, OGG, M4A). Ensure you have the rights to use this audio.
-   **Convert to WAV:** Most TTS frameworks expect WAV format. Use tools like `ffmpeg` or audio libraries (`pydub`, `soundfile`) to convert your audio. Aim for a standard WAV encoding like PCM 16-bit.
    ```bash
    # Example using ffmpeg to convert MP3 to WAV
    ffmpeg -i input_audio.mp3 output_audio.wav
    ```
-   **Standardize Channels (Mono):** TTS models typically train on single-channel (mono) audio. Convert stereo tracks to mono.
    ```bash
    # Example using ffmpeg to convert stereo WAV to mono WAV
    ffmpeg -i stereo_input.wav -ac 1 mono_output.wav
    ```
    *   `-ac 1`: Sets the number of audio channels to 1.
-   **Resample Audio:** Ensure all audio files have the **exact same sampling rate**. Choose your target rate based on your project goals and framework compatibility (e.g., 16000 Hz, 22050 Hz, 48000 Hz). 22050 Hz is common for many models.
    ```bash
    # Example using ffmpeg to resample to 22050 Hz
    ffmpeg -i input.wav -ar 22050 resampled_output.wav
    ```
    *   `-ar 22050`: Sets the audio sampling rate (samples per second).

### 1.2. Audio Chunking (Splitting into Segments)

-   **Goal:** Break long audio files (like chapters of an audiobook or podcast episodes) into shorter, manageable segments. Ideal segment length is typically between **2 to 15 seconds**.
-   **Why Chunk?**
    *   Aligns audio duration with typical sentence lengths.
    *   Makes transcription feasible (transcribing hours-long files is difficult).
    *   Helps manage memory during training.
    *   Allows filtering out unsuitable segments (e.g., pure silence, noise, music).
-   **Method:** Use tools that detect silence to split the audio. `pydub` is a popular Python library for this.

    ```python
    # Example using pydub for silence-based splitting
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import os

    input_file = "resampled_mono_audio.wav" # Use the output from step 1.1
    output_dir = "audio_chunks"             # Create this directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading audio file: {input_file}")
    sound = AudioSegment.from_wav(input_file)
    print("Audio loaded. Splitting based on silence...")

    chunks = split_on_silence(
        sound,
        min_silence_len=500,    # Minimum duration of silence in milliseconds to trigger a split. Adjust as needed.
        silence_thresh=-40,     # Silence threshold in dBFS (decibels relative to full scale). Lower values (e.g., -50) detect quieter silences. Adjust based on your audio's noise floor.
        keep_silence=200        # Optional: Amount of silence (in ms) to leave at the beginning/end of each chunk. Helps avoid abrupt cuts.
    )

    print(f"Found {len(chunks)} potential chunks before duration filtering.")

    # --- Filtering and Exporting ---
    min_duration_sec = 2.0  # Minimum chunk length in seconds
    max_duration_sec = 15.0 # Maximum chunk length in seconds
    target_sr = 22050       # Ensure chunks retain the correct sample rate (pydub usually handles this)

    exported_count = 0
    for i, chunk in enumerate(chunks):
        duration_sec = len(chunk) / 1000.0
        if min_duration_sec <= duration_sec <= max_duration_sec:
            # Ensure the chunk uses the target sample rate if necessary (pydub tries to preserve it)
            # chunk = chunk.set_frame_rate(target_sr) # Usually not needed if source was correctly sampled
            
            chunk_filename = f"segment_{exported_count:05d}.wav" # Use padding for easier sorting
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            print(f"Exporting chunk {i} ({duration_sec:.2f}s) to {chunk_path}")
            chunk.export(chunk_path, format="wav")
            exported_count += 1
        else:
             print(f"Skipping chunk {i} due to duration: {duration_sec:.2f}s")


    print(f"\nExported {exported_count} chunks meeting duration criteria ({min_duration_sec}-{max_duration_sec}s) to '{output_dir}'.")
    ```
-   **Review:** Listen to a sample of the generated chunks. Are the splits logical? Is speech cut off? Adjust `min_silence_len` and `silence_thresh` and re-run if necessary. Manually splitting or refining splits in an audio editor (like Audacity) might be needed for tricky audio.

### 1.3. Volume Normalization

-   **Goal:** Ensure all audio chunks have a consistent volume level. This prevents quiet or loud segments from disproportionately affecting training.
-   **Methods:**
    *   **Peak Normalization:** Adjusts the audio so the loudest point reaches a specific level (e.g., -3.0 dBFS). Simple, but doesn't guarantee consistent *perceived* loudness.
    *   **Loudness Normalization (LUFS):** Adjusts the audio to meet a target perceived loudness level (e.g., -23 LUFS is common for broadcast). Generally preferred as it better reflects human hearing. Requires libraries like `pyloudnorm`.
-   **Apply Consistently:** Apply the chosen normalization method to *all* chunks created in the previous step. Save the normalized files to a **new directory** (e.g., `normalized_chunks`) to keep originals intact.

    ```python
    # Example using pydub for PEAK normalization
    from pydub import AudioSegment
    import os
    import glob

    input_chunk_dir = "audio_chunks"
    output_norm_dir = "normalized_chunks"
    os.makedirs(output_norm_dir, exist_ok=True)
    
    target_dBFS = -3.0 # Target peak amplitude

    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    print(f"Normalizing chunks from '{input_chunk_dir}' to '{output_norm_dir}' with target peak {target_dBFS} dBFS.")
    
    wav_files = glob.glob(os.path.join(input_chunk_dir, "*.wav"))
    
    for i, wav_file in enumerate(wav_files):
        filename = os.path.basename(wav_file)
        output_path = os.path.join(output_norm_dir, filename)
        
        try:
            sound = AudioSegment.from_wav(wav_file)
            # Only apply gain if the sound is not silent (dBFS is not -inf)
            if sound.dBFS > -float('inf'):
              normalized_sound = match_target_amplitude(sound, target_dBFS)
              normalized_sound.export(output_path, format="wav")
            else:
              print(f"Skipping silent file: {filename}")
              # Optionally copy silent files or handle them as needed
              # shutil.copy(wav_file, output_path) 
            
            if (i + 1) % 50 == 0: # Print progress
                 print(f"Processed {i+1}/{len(wav_files)} files...")
                 
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nNormalization complete. Normalized files saved in '{output_norm_dir}'.")
    ```
    *   **Note:** For LUFS normalization, you'd use a library like `pyloudnorm`, iterating through files similarly.

### 1.4. Transcription: Creating Text Pairs

-   **Goal:** Obtain an accurate text transcript for *every single normalized audio chunk*. The text should represent *exactly* what is spoken in the audio.
-   **Methods:**
    *   **Automatic Speech Recognition (ASR):** Best for large datasets. Use high-quality ASR models.
        *   **[OpenAI Whisper](https://github.com/openai/whisper):** Excellent multilingual, open-source option. Runs locally (requires GPU for speed) or via API.
        *   **Cloud Services:** Google Cloud Speech-to-Text, AWS Transcribe, Azure Speech Service offer robust APIs.
        *   **Other Models:** Explore Hugging Face for other fine-tuned ASR models specific to your language.
    *   **Manual Transcription:** Most accurate but very time-consuming. Suitable for small, high-value datasets or for correcting ASR outputs.
    *   **Existing Transcripts:** If your source audio comes with aligned transcripts (e.g., some audiobooks, broadcast archives), you may need scripts to parse and align them with your chunks.
-   **Output Format:** Create one `.txt` file for each corresponding `.wav` file in your `normalized_chunks` directory. The filenames must match exactly (e.g., `normalized_chunks/segment_00001.wav` needs `transcripts/segment_00001.txt`).
-   **Text Cleaning and Normalization:** **This is crucial!**
    *   **Remove Non-Speech:** Delete timestamps (like `[00:01:05]`), speaker labels ("SPEAKER A:", "John Doe:"), sound event tags (`[laughter]`, `[music]`), transcription comments.
    *   **Handle Filler Words:** Decide whether to keep or remove common fillers ("uh," "um," "ah"). Keeping them might make the TTS sound more natural but can also introduce unwanted hesitations. Removing them leads to cleaner, more direct speech. Consistency is key.
    *   **Punctuation:** Ensure consistent and appropriate punctuation. Commas, periods, question marks help the model learn prosody. Avoid excessive or non-standard punctuation.
    *   **Numbers, Acronyms, Symbols:** Expand them into words (e.g., "101" -> "one hundred one", "USA" -> "U S A" or "United States of America", "%" -> "percent"). How you expand depends on how you want the TTS to pronounce them. Create a normalization dictionary/ruleset if needed.
    *   **Case:** Usually convert text to a consistent case (e.g., lowercase) unless your TTS framework/tokenizer handles casing appropriately. Check framework docs.
    *   **Special Characters:** Remove or replace characters that might confuse the tokenizer (e.g., emojis, control characters).

    ```
    # Example structure:
    my_tts_dataset/
    ├── normalized_chunks/
    │   ├── segment_00001.wav
    │   ├── segment_00002.wav
    │   └── ...
    └── transcripts/
        ├── segment_00001.txt  # Contains "Hello world."
        ├── segment_00002.txt  # Contains "This is a test sentence."
        └── ...
    ```

### 1.5. Data Structuring & Manifest File Creation

-   **Goal:** Create index files (manifests) that tell the TTS training script where to find the audio files and their corresponding transcriptions.
-   **Manifest Format:** The most common format is a plain text file where each line represents one audio-text pair, separated by a delimiter (usually a pipe `|`).
    ```
    path/to/audio_chunk.wav|The corresponding transcription text|speaker_id
    ```
    *   `path/to/audio_chunk.wav`: Relative path to the normalized audio file from the directory where the training script will be run.
    *   `The corresponding transcription text`: The cleaned, normalized text from the `.txt` file.
    *   `speaker_id`: An identifier for the speaker (e.g., `speaker0`, `mary_smith`). For single-speaker datasets, use the same ID for all lines. For multi-speaker datasets, use unique IDs for each distinct speaker.
-   **Splitting Data (Train/Validation):** Divide your data into a training set (used to update model weights) and a validation set (used to monitor performance on unseen data and prevent overfitting). A common split is 90-98% for training and 2-10% for validation. **Crucially, ensure that segments from the *same original long recording* do not end up in both train and validation sets if possible, to avoid data leakage.** If splitting randomly, shuffle first.
-   **Generate Manifests Script:**

    ```python
    import os
    import random

    # --- Configuration ---
    dataset_name = "my_tts_dataset"
    normalized_audio_dir = os.path.join(dataset_name, "normalized_chunks")
    transcripts_dir = os.path.join(dataset_name, "transcripts")
    output_dir = dataset_name # Where manifest files will be saved

    train_manifest_path = os.path.join(output_dir, "train_list.txt")
    val_manifest_path = os.path.join(output_dir, "val_list.txt")

    speaker_id = "main_speaker" # Use a consistent ID for single speaker datasets
                                # For multi-speaker, determine ID based on filename or source
    val_split_ratio = 0.05    # 5% for validation set
    random_seed = 42          # For reproducible splits
    # ---------------------

    manifest_entries = []
    print("Reading audio and transcript files...")

    # Iterate through normalized audio files
    wav_files = sorted([f for f in os.listdir(normalized_audio_dir) if f.endswith(".wav")])

    for wav_filename in wav_files:
        base_filename = os.path.splitext(wav_filename)[0]
        txt_filename = base_filename + ".txt"
        
        audio_path = os.path.join(normalized_audio_dir, wav_filename)
        # Use os.path.relpath if your training script runs from a different root
        # relative_audio_path = os.path.relpath(audio_path, start=training_script_dir) 
        relative_audio_path = audio_path # Assuming script runs from root containing 'my_tts_dataset'

        transcript_path = os.path.join(transcripts_dir, txt_filename)

        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                # Basic cleaning: remove pipe chars, trim extra whitespace
                transcript = transcript.replace('|', ' ').strip()
                transcript = ' '.join(transcript.split()) # Normalize whitespace

                if transcript: # Ensure transcript is not empty after cleaning
                    manifest_entries.append(f"{relative_audio_path}|{transcript}|{speaker_id}")
                else:
                    print(f"Warning: Empty transcript for {wav_filename}. Skipping.")
            except Exception as e:
                print(f"Error reading or processing transcript {txt_filename}: {e}. Skipping.")
        else:
            print(f"Warning: Missing transcript file {txt_filename} for {wav_filename}. Skipping.")

    print(f"Found {len(manifest_entries)} valid audio-transcript pairs.")

    # Shuffle and split
    random.seed(random_seed)
    random.shuffle(manifest_entries)

    split_idx = int(len(manifest_entries) * (1 - val_split_ratio))
    train_entries = manifest_entries[:split_idx]
    val_entries = manifest_entries[split_idx:]

    # Write manifest files
    try:
        with open(train_manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_entries))
        print(f"Successfully wrote {len(train_entries)} entries to {train_manifest_path}")

        with open(val_manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_entries))
        print(f"Successfully wrote {len(val_entries)} entries to {val_manifest_path}")
    except Exception as e:
        print(f"Error writing manifest files: {e}")

    ```

---

## 2. Data Quality Checklist

Before moving to training setup, rigorously review your prepared dataset using this checklist. Fixing issues now saves significant time later.

| Aspect                  | Check                                                                 | Why Important?                                             | Action if Failed                                                                      |
| :---------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Audio Completeness**  | Do all listed `.wav` files in manifests actually exist?               | Training will crash if files are missing.                | Re-run manifest generation; check file paths; ensure no files were accidentally deleted. |
| **Transcript Match**    | Does each `.wav` have a corresponding, accurate `.txt`/transcript?    | Mismatched pairs teach the model incorrect associations. | Verify filenames; review ASR output; manually correct transcripts.                     |
| **Audio Length**        | Are most segments within the desired range (e.g., 2-15s)? Few outliers? | Very short/long segments can destabilize training.       | Re-run chunking with adjusted parameters; manually filter outliers from manifests.      |
| **Audio Quality**       | Listen to random samples: Low background noise? No music/reverb/echo? | Garbage In, Garbage Out. Model learns the noise.         | Improve source audio; apply noise reduction (carefully!); filter out bad segments.     |
| **Speaker Consistency** | For single-speaker: Is it always the target voice? No other speakers? | Prevents voice dilution or instability.                    | Manually review/filter segments; check chunking boundaries.                         |
| **Format & Specs**      | All WAV? **Identical** sampling rate? Mono channels? PCM 16-bit?      | Inconsistencies cause errors or poor performance.        | Re-run conversion/resampling steps (Section 1.1). Verify with tools like `soxi`.      |
| **Volume Levels**       | Listen to random samples: Are volumes relatively consistent?          | Drastic volume shifts can hinder learning.               | Re-run normalization (Section 1.3); check normalization parameters.                 |
| **Transcription Cleanliness** | No timestamps, speaker labels? Fillers handled consistently? Punctuation standard? Numbers/symbols expanded? | Ensures text maps cleanly to speech sounds/prosody.      | Re-run text cleaning scripts; perform manual review and correction.                   |
| **Manifest Format**     | Correct `path|text|speaker_id` structure? Paths valid? No extra lines? | Parser errors will prevent data loading.                 | Check delimiter (`|`); validate paths relative to training script location; check encoding (UTF-8 preferred). |
| **Train/Val Split**     | Are validation files truly unseen during training? No overlap?        | Overlapping data gives misleading validation scores.     | Ensure random shuffle before splitting; check splitting logic.                        |

**Tip:** Use tools like `soxi` (from SoX) or `ffprobe` to batch-check audio properties (sampling rate, channels, duration). Write small scripts to verify file existence and basic manifest formatting.

---

Once your dataset passes this quality check, you are ready to proceed to setting up the training environment.

**Next Step:** [Training Setup (`2_TRAINING_SETUP.md`)](./2_TRAINING_SETUP.md)
