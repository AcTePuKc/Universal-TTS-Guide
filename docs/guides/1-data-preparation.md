# TTS Dataset Preparation Guide


This guide covers the critical first phase of any TTS project: preparing high-quality, correctly formatted audio and text data for TTS training or fine-tuning. The quality of your dataset directly impacts the quality of your final TTS model.

If an audio term feels unfamiliar while you read, use the [glossary](../glossary.md#glossary-of-technical-terms). This page only pauses to explain terms when they matter for the step you are doing.

---

## Dataset Preparation Steps

Follow these steps systematically to transform raw audio into a training-ready dataset.

### 1. Audio Acquisition and Initial Processing

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
-   **Resample Audio:** Ensure all audio files have the **exact same [sampling rate](../glossary.md#glossary-sampling-rate)**. Sampling rate means how many audio samples are stored per second. Choose your target rate based on your project goals and framework compatibility (e.g., 16000 Hz, 22050 Hz, 48000 Hz). 22050 Hz is common for many models.
    ```bash
    # Example using ffmpeg to resample to 22050 Hz
    ffmpeg -i input.wav -ar 22050 resampled_output.wav
    ```
    *   `-ar 22050`: Sets the audio sampling rate (samples per second).

**Small but important warning:** do not mix sample rates, channel layouts, or clipping-heavy files in the same dataset and hope the model will figure it out. These inconsistencies usually show up later as noisy, unstable, or unnatural speech.

### 2. Advanced Audio Cleaning (Noise/Music Removal)

-   **Goal:** To remove unwanted background sounds like noise (hum, hiss, fans), music, reverb, or other interfering voices from your source audio, isolating the target speaker's voice as much as possible. This step is crucial if your source audio is not studio quality.
-   **Why?** TTS models learn from the audio they are given. If the audio contains background noise or music, the resulting TTS voice will likely inherit these characteristics, sounding noisy or "muddy". Cleaner audio leads to a cleaner TTS voice.

-   **Tools & Techniques:**
    *   **AI Source Separation Tools (Recommended for Music/Voice):** These tools use AI models to separate audio into different stems (vocals, music, drums, bass, other).
        *   **[Ultimate Vocal Remover (UVR)](https://ultimatevocalremover.com/)**: A popular, free, open-source GUI application that provides access to various state-of-the-art AI separation models. It's excellent for removing background music or isolating dialogue.
            *   **Models (like those mentioned):** UVR allows you to use different AI models. `MDX-Inst-HQ3` is one such model often good at separating vocals from instruments (hence "Inst"). Other MDX models, Demucs models (like `htdemucs`), and potentially models like Mel-Roformer (if integrated or available standalone) are designed for similar tasks, each with slightly different strengths and weaknesses. Experimentation is key. Choose models focused on **vocal isolation**.
        *   **Other Tools:** Online services (e.g., Lalal.ai) or other standalone software might use similar underlying models (often Demucs or Spleeter variants).
    *   **Traditional Noise Reduction Tools:** Often found in Digital Audio Workstations (DAWs) or audio editors.
        *   **[Audacity](https://www.audacityteam.org/):** Contains built-in noise reduction effects (requires sampling a noise profile). Can be effective for constant background noise (like hiss or hum).
        *   **Commercial Plugins (e.g., Izotope RX, Waves Clarity):** Offer more sophisticated AI-powered noise, reverb, and voice isolation tools, but come at a cost.
    *   **Spectral Editing:** Manually removing unwanted sounds in a spectral editor (like Adobe Audition, Izotope RX, Acon Digital Acoustica). Powerful but very time-consuming.

-   **Workflow Considerations:**
    *   **When to Apply:** It's generally recommended to apply cleaning to your **longer audio files *before* chunking (Step 1.3 below)**. This allows the AI models to work with more context and can be more efficient than processing thousands of small chunks. However, if cleaning introduces too many artifacts on long files, you might try cleaning individual problematic chunks later.
    *   **Process:**
        1.  Load your standardized WAV file (from Step 1.1) into the chosen tool (e.g., UVR).
        2.  Select an appropriate vocal isolation model (e.g., an MDX or Demucs vocal model).
        3.  Process the audio to generate a "vocals only" track.
        4.  **Listen Carefully:** Critically evaluate the separated vocal track. Check for:
            *   **Artifacts:** AI separation can sometimes introduce "watery" sounds, glitches, or parts of the voice being mistakenly removed.
            *   **Remaining Noise/Music:** How effectively was the unwanted sound removed?
        5.  **Iterate:** You might need to try different models, adjust settings within the tool, or even apply a secondary noise reduction pass (e.g., using Audacity's noise reduction on the AI-separated vocals) for best results.
    *   **Save Output:** Save the cleaned vocal track as a new WAV file (e.g., `original_file_cleaned.wav`). Use these cleaned files as the input for the *next* step (Chunking).

-   **Caveats:**
    *   **Artifacts are Possible:** Aggressive cleaning can degrade the naturalness of the target voice. Aim for a balance between removing noise and preserving voice quality.
    *   **Computational Cost:** AI separation models can be computationally intensive and may take significant time, especially on long audio files and without a powerful GPU.


### 3. Audio Chunking (Splitting into Segments)

-   **Goal:** Break long audio files (like chapters of an audiobook or podcast episodes) into shorter, manageable segments. Ideal segment length is typically between **2 to 15 seconds**.
-   **Why Chunk?**
    *   Aligns audio duration with typical sentence lengths.
    *   Makes transcription feasible (transcribing hours-long files is difficult).
    *   Helps manage memory during training.
    *   Allows filtering out unsuitable segments (e.g., pure silence, noise, music).
-   **Method:** Use tools that detect silence to split the audio. `pydub` is a popular Python library for this.
-   **Important:** Chunk boundaries should make later transcription straightforward. If a split cuts off the beginning or end of a word, fix the split first instead of hoping the model will learn around a bad audio-text pair.

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

### 4. Volume Normalization

-   **Goal:** Ensure all audio chunks have a consistent volume level. This prevents quiet or loud segments from disproportionately affecting training.
-   **Methods:**
    *   **Peak Normalization:** Adjusts the audio so the loudest point reaches a specific level (e.g., -3.0 [dBFS](../glossary.md#glossary-dbfs)). Simple, but doesn't guarantee consistent *perceived* loudness.
    *   **Loudness Normalization (LUFS):** [LUFS](../glossary.md#glossary-lufs) measures how loud audio sounds to people, not just how high the waveform peaks. This adjusts the audio to meet a target perceived loudness level (e.g., -23 LUFS is common for broadcast). It is often more consistent than peak normalization, but it requires libraries like `pyloudnorm`.
    *   **Beginner Tip:** If LUFS feels too advanced for a first pass, consistent peak normalization is still much better than leaving clips at mixed volume levels.
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

### 5. Transcription: Creating Text Pairs

-   **Goal:** Obtain an accurate text transcript for *every single normalized audio chunk*. A transcript is simply the text version of what is spoken in the audio. It should represent *exactly* what is spoken in the chunk.
-   **Methods:**
    *   **Automatic Speech Recognition (ASR):** Best for large datasets. Use high-quality ASR models.
    *   **[OpenAI Whisper](https://github.com/openai/whisper):** Excellent multilingual, open-source option. Runs locally (GPU recommended) or via API. *Note: While powerful for word accuracy, Whisper's punctuation and capitalization may require careful review and correction during the cleaning step.* Various community fine-tuned Whisper models (often found on Hugging Face) may offer improvements.
    *   **[Google audio transcription tools and APIs](https://ai.google.dev/):** Google may offer audio-capable models or services that can help with transcription. Product names, limits, and free tiers change over time, so verify the current documentation before depending on a specific workflow.
    *   **Cloud Services:** Google Cloud Speech-to-Text, AWS Transcribe, Azure Speech Service offer robust APIs, often with pay-as-you-go pricing and potentially free tiers initially.
    *   **Other Models:** Explore [Hugging Face Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) for other open-source or fine-tuned ASR models specific to your language.
    *   **Manual Transcription:** Most accurate but very time-consuming. Suitable for small, high-value datasets or for *correcting ASR outputs*.
    *   **Existing Transcripts:** If your source audio comes with aligned transcripts (e.g., some audiobooks, broadcast archives), you may need scripts to parse and align them with your chunks.
-   **Warning:** Vendor model names, API limits, and pricing can change quickly. Always verify current documentation before building a workflow around a specific hosted ASR service.
-   **Output Format:** Create one `.txt` file for each corresponding `.wav` file in your `normalized_chunks` directory. The filenames must match exactly (e.g., `normalized_chunks/segment_00001.wav` needs `transcripts/segment_00001.txt`).
-   **Text Cleaning and Normalization:** **This is crucial!**
    *   **Remove Non-Speech:** Delete timestamps (like `[00:01:05]`), speaker labels ("SPEAKER A:", "John Doe:"), sound event tags (`[laughter]`, `[music]`), transcription comments.
    *   **Handle Filler Words:** Decide whether to keep or remove common fillers ("uh," "um," "ah"). Keeping them might make the TTS sound more natural but can also introduce unwanted hesitations. Removing them leads to cleaner, more direct speech. Consistency is key.
    *   **Punctuation:** Ensure consistent and appropriate punctuation. Commas, periods, question marks help the model learn prosody. Avoid excessive or non-standard punctuation.
    *   **Numbers, Acronyms, Symbols:** Expand them into words (e.g., "101" -> "one hundred one", "USA" -> "U S A" or "United States of America", "%" -> "percent"). How you expand depends on how you want the TTS to pronounce them. Create a normalization dictionary/ruleset if needed.
    *   **Case:** Usually convert text to a consistent case (e.g., lowercase) unless your TTS framework/tokenizer handles casing appropriately. Check framework docs.
    *   **Special Characters:** Remove or replace characters that might confuse the tokenizer (e.g., emojis, control characters).

    ```mermaid
    flowchart TD
        dataset["my_tts_dataset/"]
        audio["normalized_chunks/"]
        wav1["segment_00001.wav"]
        wav2["segment_00002.wav"]
        wavMore["..."]
        transcripts["transcripts/"]
        txt1["segment_00001.txt<br/>Contains: Hello world."]
        txt2["segment_00002.txt<br/>Contains: This is a test sentence."]
        txtMore["..."]

        dataset --> audio
        audio --> wav1
        audio --> wav2
        audio --> wavMore
        dataset --> transcripts
        transcripts --> txt1
        transcripts --> txt2
        transcripts --> txtMore
    ```

#### Example: Raw Transcript to Clean Training Text

Use consistent cleanup rules. For example:

```text
Raw transcript:
[00:01:05] SPEAKER A: Um, I paid $12.50 for 3 apples...

Cleaned training text:
I paid twelve dollars and fifty cents for three apples.
```

The goal is not to make the text look literary. The goal is to make the text match what you want the model to say, in a form the tokenizer and model can learn consistently.

### 6. Data Structuring and Manifest File Creation

-   **Goal:** Create index files called [manifest files](../glossary.md#glossary-manifest-file) that tell the TTS training script where to find the audio files and their corresponding transcriptions.
-   **Manifest Format:** The most common format is a plain text file where each line represents one audio-text pair, separated by a delimiter (usually a pipe `|`).
    ```text
    path/to/audio_chunk.wav|The corresponding transcription text|speaker_id
    ```
    *   `path/to/audio_chunk.wav`: Relative path to the normalized audio file from the directory where the training script will be run.
    *   `The corresponding transcription text`: The cleaned, normalized text from the `.txt` file.
    *   `speaker_id`: An identifier for the speaker (e.g., `speaker0`, `mary_smith`). For single-speaker datasets, use the same ID for all lines. For multi-speaker datasets, use unique IDs for each distinct speaker.
-   **Splitting Data (Train/Validation):** Divide your data into a training set (used to update model weights) and a validation set (used to monitor performance on unseen data and prevent overfitting). A common split is 90-98% for training and 2-10% for validation. **Crucially, ensure that segments from the *same original long recording* do not end up in both train and validation sets if possible, to avoid data leakage.** If splitting randomly, shuffle first.
-   **Beginner-safe default:** If you do not yet have a better strategy, keep validation small but representative. A simple 95/5 split is usually fine for a first pass, as long as you avoid leaking chunks from the same original recording into both sets.
-   **Path warning:** Many frameworks expect manifest audio paths to be relative to a specific working directory, not relative to the manifest file itself. Before training, open one manifest line and verify that the training script can actually resolve that path from the directory where you will run it.
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
        # IMPORTANT: many frameworks want paths relative to the directory where
        # train.py will be launched, not absolute paths and not paths relative
        # to the manifest file itself.
        # Example:
        # relative_audio_path = os.path.relpath(audio_path, start=training_script_dir)
        relative_audio_path = audio_path  # This only works if your framework accepts this exact path style.

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

### Practical Verification Scripts

Here are some practical scripts to help verify your dataset quality:

**Platform note:** the shell example below is for Linux or macOS. On Windows, either run it from Git Bash/WSL or translate the same checks into PowerShell or Python.

#### Check Audio Properties (Sampling Rate, Channels, Duration)

```bash
#!/bin/bash
# verify_audio.sh - Check audio properties across all WAV files
# Usage: ./verify_audio.sh /path/to/audio/directory

AUDIO_DIR="$1"
echo "Checking audio files in $AUDIO_DIR..."

# Check if SoX is installed
if ! command -v soxi &> /dev/null; then
    echo "SoX not found. Please install it first (e.g., 'apt-get install sox' or 'brew install sox')."
    exit 1
fi

# Initialize counters and arrays
total_files=0
non_mono=0
wrong_rate=0
too_short=0
too_long=0
target_rate=22050  # Change this to your target sampling rate
min_duration=1.0   # Minimum duration in seconds
max_duration=15.0  # Maximum duration in seconds

# Process all WAV files
find "$AUDIO_DIR" -name "*.wav" | while read -r file; do
    total_files=$((total_files + 1))
    
    # Get audio properties
    channels=$(soxi -c "$file")
    rate=$(soxi -r "$file")
    duration=$(soxi -d "$file" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    
    # Check properties
    if [ "$channels" -ne 1 ]; then
        echo "WARNING: Non-mono file: $file (channels: $channels)"
        non_mono=$((non_mono + 1))
    fi
    
    if [ "$rate" -ne "$target_rate" ]; then
        echo "WARNING: Wrong sampling rate: $file (rate: $rate Hz, expected: $target_rate Hz)"
        wrong_rate=$((wrong_rate + 1))
    fi
    
    if (( $(echo "$duration < $min_duration" | bc -l) )); then
        echo "WARNING: File too short: $file (duration: ${duration}s, minimum: ${min_duration}s)"
        too_short=$((too_short + 1))
    fi
    
    if (( $(echo "$duration > $max_duration" | bc -l) )); then
        echo "WARNING: File too long: $file (duration: ${duration}s, maximum: ${max_duration}s)"
        too_long=$((too_long + 1))
    fi
    
    # Print progress every 100 files
    if [ $((total_files % 100)) -eq 0 ]; then
        echo "Processed $total_files files..."
    fi
done

# Print summary
echo "===== SUMMARY ====="
echo "Total files checked: $total_files"
echo "Non-mono files: $non_mono"
echo "Files with wrong sampling rate: $wrong_rate"
echo "Files too short (<${min_duration}s): $too_short"
echo "Files too long (>${max_duration}s): $too_long"

if [ $((non_mono + wrong_rate + too_short + too_long)) -eq 0 ]; then
    echo "All files passed basic checks!"
else
    echo "Some issues were found. Please review the warnings above."
fi
```

#### Verify Manifest File Integrity

```python
#!/usr/bin/env python3
# verify_manifest.py - Check that all files in manifest exist and have matching transcripts
# Usage: python verify_manifest.py path/to/manifest.txt

import os
import sys
from pathlib import Path

def verify_manifest(manifest_path):
    """Verify that all audio files and transcripts in the manifest exist and are valid."""
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        return False
    
    print(f"Verifying manifest: {manifest_path}")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    
    # Statistics
    total_entries = 0
    missing_audio = 0
    empty_transcripts = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_entries += 1
            
            # Parse the line (assuming pipe-separated format: audio_path|transcript|speaker_id)
            parts = line.split('|')
            if len(parts) < 2:
                print(f"Line {line_num}: Invalid format. Expected at least 'audio_path|transcript'")
                continue
            
            audio_path = parts[0]
            transcript = parts[1]
            
            # Check if audio path is relative and resolve it
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"Line {line_num}: Audio file not found: {audio_path}")
                missing_audio += 1
            
            # Check if transcript is empty
            if not transcript or transcript.isspace():
                print(f"Line {line_num}: Empty transcript for {audio_path}")
                empty_transcripts += 1
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Total entries: {total_entries}")
    print(f"Missing audio files: {missing_audio}")
    print(f"Empty transcripts: {empty_transcripts}")
    
    if missing_audio == 0 and empty_transcripts == 0:
        print("All manifest entries are valid!")
        return True
    else:
        print("Issues found in manifest. Please fix them before proceeding.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_manifest.py path/to/manifest.txt")
        sys.exit(1)
    
    success = verify_manifest(sys.argv[1])
    sys.exit(0 if success else 1)
```

#### Visualize Audio Spectrograms for Quality Assessment

This script helps you visually inspect the quality of your audio files by generating [mel spectrograms](../glossary.md#glossary-mel-spectrogram), which are image-like views of audio energy over time:

```python
#!/usr/bin/env python3
# generate_spectrograms.py - Create spectrograms for audio quality assessment
# Usage: python generate_spectrograms.py /path/to/audio/directory /path/to/output/directory [num_samples]

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

def generate_spectrograms(audio_dir, output_dir, num_samples=10):
    """Generate spectrograms for a random sample of audio files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all WAV files
    wav_files = list(Path(audio_dir).glob('**/*.wav'))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return False
    
    # Sample files if there are more than requested
    if len(wav_files) > num_samples:
        wav_files = random.sample(wav_files, num_samples)
    
    print(f"Generating spectrograms for {len(wav_files)} files...")
    
    for i, wav_path in enumerate(wav_files):
        try:
            # Load audio file
            y, sr = librosa.load(wav_path, sr=None)
            
            # Create figure with two subplots
            plt.figure(figsize=(12, 8))
            
            # Plot waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {wav_path.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Plot spectrogram
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram')
            
            # Save figure
            output_path = os.path.join(output_dir, f'spectrogram_{i+1}_{wav_path.stem}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            print(f"Generated: {output_path}")
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
    
    print(f"Spectrograms saved to {output_dir}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_spectrograms.py /path/to/audio/directory /path/to/output/directory [num_samples]")
        sys.exit(1)
    
    audio_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    success = generate_spectrograms(audio_dir, output_dir, num_samples)
    sys.exit(0 if success else 1)
```

These scripts provide practical tools to verify your dataset's quality before training, helping you identify and fix issues early in the process.

---

## Data Quality Checklist

Before moving to training setup, do one final quick review of the prepared dataset:

- [ ] Every `.wav` listed in the manifests actually exists.
- [ ] Every audio segment has the correct matching transcript.
- [ ] Most segments stay within your target duration range, with few outliers.
- [ ] Random spot checks sound clean and do not contain strong noise, music, echo, or the wrong speaker.
- [ ] All files use the same sample rate, channel layout, and expected format.
- [ ] Volume levels are reasonably consistent across the dataset.
- [ ] Transcripts are clean: no timestamps, no speaker labels, and consistent punctuation.
- [ ] Manifest files use the correct format and resolve properly from the directory where you will run `train.py`.
- [ ] Your validation set does not overlap with the training set.

**Tip:** Use tools like `soxi` (from SoX) or `ffprobe` to batch-check audio properties (sampling rate, channels, duration). Write small scripts to verify file existence and basic manifest formatting.

Once your dataset passes this quality check, you are ready to proceed to setting up the training environment.
