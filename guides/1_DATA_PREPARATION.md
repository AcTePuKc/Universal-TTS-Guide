# Guide 1: Data Preparation for TTS Training

**Navigation:** [Main README]({{ site.baseurl }}/){: .btn .btn-primary} | [Next Step: Training Setup](./2_TRAINING_SETUP.md){: .btn .btn-primary} | 

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

### 1.2 Advanced Audio Cleaning (Noise/Music Removal) - *Optional but Recommended*

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


### 1.3. Audio Chunking (Splitting into Segments)

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

### 1.4. Volume Normalization

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

### 1.5. Transcription: Creating Text Pairs

-   **Goal:** Obtain an accurate text transcript for *every single normalized audio chunk*. The text should represent *exactly* what is spoken in the audio.
-   **Methods:**
    *   **Automatic Speech Recognition (ASR):** Best for large datasets. Use high-quality ASR models.
    *   **[OpenAI Whisper](https://github.com/openai/whisper):** Excellent multilingual, open-source option. Runs locally (GPU recommended) or via API. *Note: While powerful for word accuracy, Whisper's punctuation and capitalization may require careful review and correction during the cleaning step.* Various community fine-tuned Whisper models (often found on Hugging Face) may offer improvements.
    *   **[Google Gemini Models](https://ai.google.dev/) (e.g., via API or AI Studio):** Models like Gemini Pro or Flash can perform audio transcription. Often requires audio to be in specific formats and may perform best on shorter segments (aligning well with the pre-chunking step). Check current API offerings and potential free tiers.
    *   **Cloud Services:** Google Cloud Speech-to-Text, AWS Transcribe, Azure Speech Service offer robust APIs, often with pay-as-you-go pricing and potentially free tiers initially.
    *   **Other Models:** Explore [Hugging Face Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) for other open-source or fine-tuned ASR models specific to your language.
    *   **Manual Transcription:** Most accurate but very time-consuming. Suitable for small, high-value datasets or for *correcting ASR outputs*.
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

### 1.6. Data Structuring & Manifest File Creation

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

| Aspect                  | Check                                                               | Why Important?                                             | Action if Failed                                                                      |
| :---------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Audio Completeness**  | Do all listed `.wav` files in manifests actually exist?               | Training will crash if files are missing.                | Re-run manifest generation; check file paths; ensure no files were accidentally deleted. |
| **Transcript Match**    | Does each `.wav` have a corresponding, accurate `.txt`/transcript?    | Mismatched pairs teach the model incorrect associations. | Verify filenames; review ASR output; manually correct transcripts.                     |
| **Audio Length**        | Are most segments within the desired range (e.g., 2-15s)? Few outliers? | Very short/long segments can destabilize training.       | Re-run chunking with adjusted parameters; manually filter outliers from manifests.      |
| **Audio Quality**       | Listen to random samples: Low background noise? No music/reverb/echo? | Garbage In, Garbage Out. Model learns the noise.         | Improve source audio; apply noise reduction (carefully!); filter out bad segments.     |
| **Speaker Consistency** | For single-speaker: Is it always the target voice? No other speakers? | Prevents voice dilution or instability.                    | Manually review/filter segments; check chunking boundaries.                         |
| **Format & Specs** | All WAV? **Identical** sampling rate? Mono channels? PCM 16-bit?      | Inconsistencies cause errors or poor performance. | Re-run conversion/resampling steps (Section 1.1). Batch-verify specs using command-line tools like `ffprobe` or `soxi` (part of the [SoX](http://sox.sourceforge.net/) package). Example: `soxi -r *.wav` to check rates. |
| **Volume Levels**       | Listen to random samples: Are volumes relatively consistent?          | Drastic volume shifts can hinder learning.               | Re-run normalization (Section 1.3); check normalization parameters.                 |
| **Transcription Cleanliness** | No timestamps, speaker labels? Fillers handled consistently? Punctuation standard? Numbers/symbols expanded? | Ensures text maps cleanly to speech sounds/prosody.      | Re-run text cleaning scripts; perform manual review and correction.                   |
| **Manifest Format**     | Correct `path|text|speaker_id` structure? Paths valid? No extra lines? | Parser errors will prevent data loading.                 | Check delimiter (`|`); validate paths relative to training script location; check encoding (UTF-8 preferred). |
| **Train/Val Split**     | Are validation files truly unseen during training? No overlap?        | Overlapping data gives misleading validation scores.     | Ensure random shuffle before splitting; check splitting logic.                        |

**Tip:** Use tools like `soxi` (from SoX) or `ffprobe` to batch-check audio properties (sampling rate, channels, duration). Write small scripts to verify file existence and basic manifest formatting.

### 2.1. Practical Data Verification Tools

Here are practical commands and scripts to help verify your dataset meets the quality requirements:

#### Audio Format and Properties Verification

```bash
# Check sampling rate of all WAV files (should all be the same, e.g., 22050)
# Using SoX (install with: apt-get install sox or brew install sox)
soxi -r normalized_chunks/*.wav | sort | uniq -c

# Check number of channels (should all be 1 for mono)
soxi -c normalized_chunks/*.wav | sort | uniq -c

# Check bit depth (should typically be 16-bit)
soxi -b normalized_chunks/*.wav | sort | uniq -c

# Check duration range (helps identify outliers)
for file in normalized_chunks/*.wav; do
  duration=$(soxi -D "$file")
  echo "$file: $duration seconds"
done | sort -k2 -n

# Alternative using ffprobe (part of ffmpeg)
for file in normalized_chunks/*.wav; do
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file"
done
```

#### Manifest Validation Script

This Python script helps validate your manifest files to ensure all audio files exist and have corresponding transcripts:

```python
import os
import sys
import wave

def validate_manifest(manifest_path, base_dir="."):
    """Validate a TTS manifest file for common issues."""
    print(f"Validating manifest: {manifest_path}")
    
    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest file not found: {manifest_path}")
        return False
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total entries in manifest: {len(lines)}")
    
    issues = 0
    audio_durations = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            print(f"WARNING: Line {i} is empty")
            issues += 1
            continue
        
        parts = line.split('|')
        if len(parts) < 2:
            print(f"ERROR: Line {i} doesn't have the expected format (audio_path|text|speaker_id)")
            issues += 1
            continue
        
        audio_path = parts[0]
        text = parts[1]
        
        # Make path absolute if it's relative
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(base_dir, audio_path)
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            print(f"ERROR: Audio file not found: {audio_path} (line {i})")
            issues += 1
            continue
        
        # Check if text is empty
        if not text.strip():
            print(f"WARNING: Empty transcript for {audio_path} (line {i})")
            issues += 1
        
        # Check audio duration
        try:
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
                audio_durations.append((audio_path, duration))
        except Exception as e:
            print(f"ERROR: Cannot read audio file {audio_path}: {e}")
            issues += 1
    
    # Report duration statistics
    if audio_durations:
        durations = [d for _, d in audio_durations]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        print(f"\nAudio duration statistics:")
        print(f"  Average: {avg_duration:.2f} seconds")
        print(f"  Minimum: {min_duration:.2f} seconds")
        print(f"  Maximum: {max_duration:.2f} seconds")
        
        # Report outliers (very short or very long files)
        print("\nPotential outliers:")
        for path, duration in audio_durations:
            if duration < 1.0:
                print(f"  Very short audio: {path} ({duration:.2f}s)")
            elif duration > 15.0:
                print(f"  Very long audio: {path} ({duration:.2f}s)")
    
    print(f"\nValidation complete. Found {issues} issues.")
    return issues == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_manifest.py path/to/manifest.txt [base_directory]")
        sys.exit(1)
    
    manifest_path = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    success = validate_manifest(manifest_path, base_dir)
    sys.exit(0 if success else 1)
```

#### Visual Inspection of Audio

While automated checks are helpful, visual inspection of spectrograms can reveal issues that aren't obvious from listening or basic stats:

```python
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import random

def plot_spectrogram(audio_path, output_dir="spectrograms"):
    """Generate and save a spectrogram visualization of an audio file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)  # Keep original sampling rate
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {os.path.basename(audio_path)}")
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    
    # Save figure
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

# Example usage: Plot spectrograms for 5 random files
audio_dir = "normalized_chunks"
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
sample_files = random.sample(audio_files, min(5, len(audio_files)))

for audio_file in sample_files:
    output_path = plot_spectrogram(audio_file)
    print(f"Generated spectrogram: {output_path}")
```

These tools will help you systematically verify your dataset quality and identify potential issues before training.

---

Once your dataset passes this quality check, you are ready to proceed to setting up the training environment.

**Next Step:** [Training Setup](./2_TRAINING_SETUP.md){: .btn .btn-primary} | 
[Back to Top](#top){: .btn .btn-primary}
