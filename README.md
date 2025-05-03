# Universal TTS Model Training & Dataset Preparation Guide

## Available Languages

- **English** (Current)
- [Español](/languages/es/README.md) (Coming Soon)
- [Français](/languages/fr/README.md) (Coming Soon)

*Want to contribute a translation? See the [Translation Guide](#translation-guide) below.*

## Introduction

Welcome! This comprehensive guide provides a universal process for preparing your own speech datasets and training a custom Text-to-Speech (TTS) model. Whether you have a small dataset (e.g., 10 hours) or a larger one (100+ hours), these steps will help you organize your data correctly and navigate the training process for most modern TTS frameworks.

**Goal:** To empower you to fine-tune or train a TTS model on a specific voice or language using your own audio-text pairs.

**What This Guide Covers:**
This guide is split into several parts, covering the entire workflow from planning to using your trained model:

1.  **Planning:** Initial considerations before starting your project.
2.  **Data Preparation:** Acquiring, processing, and structuring audio and text data.
3.  **Training Setup:** Preparing your environment and configuring the training parameters.
4.  **Model Training:** Launching, monitoring, and fine-tuning the TTS model.
5.  **Inference:** Using your trained model to synthesize speech.
6.  **Packaging & Sharing:** Organizing and documenting your model for future use or distribution.
7.  **Troubleshooting & Resources:** Common issues and helpful tools.

---

## 0. Before You Start: Planning Your Dataset

Before collecting data, consider these crucial points to ensure your project is well-defined and achievable:

1.  **Speaker:** Will it be a single speaker or multiple speakers? Single-speaker datasets are simpler to start with for fine-tuning or initial training. Multi-speaker models require careful data balancing and speaker ID management.
2.  **Data Source:** Where will you get the audio? (Audiobooks, podcasts, radio archives, professionally recorded voice data, your own recordings). **Crucially, ensure you have the necessary rights or licenses to use the data for training models.**
3.  **Audio Quality:** Aim for the highest quality possible. Prioritize clean recordings with minimal background noise, reverb, music, or overlapping speech. Consistency in recording conditions is highly beneficial.
4.  **Language & Domain:** What language(s) will the model speak? What is the speaking style or domain (e.g., narration, conversational, news reading)? The model will perform best on text similar to its training data.
5.  **Target Data Amount:** How much data do you plan to collect or use?
    *   **~1-5 hours:** Might be sufficient for basic voice *cloning* if using a strong pre-trained model, but quality might be limited.
    *   **~5-20 hours:** Generally considered the minimum for decent *fine-tuning* of a specific voice onto a pre-trained model.
    *   **50-100+ hours:** Better for training robust models or training models with less reliance on pre-trained weights, especially for less common languages.
    *   **1000+ hours:** Needed for training high-quality, general-purpose models largely from scratch.
6.  **Sampling Rate:** Decide on a target sampling rate (e.g., 16000 Hz, 22050 Hz, 44100 Hz, 48000 Hz) early on. Higher rates capture more detail but require more storage and compute. **All your training data MUST consistently use the chosen rate.** 22050 Hz is a common balance for many TTS models.

---

## Process Overview & Navigation

This guide is broken down into focused modules. Follow the links below for detailed steps on each phase:

1.  **➡️ [Data Preparation](./guides/1_DATA_PREPARATION.md)**
    *   Covers acquiring, cleaning, segmenting, normalizing audio, transcribing text, and creating the necessary manifest files for training. Includes the crucial data quality checklist.

2.  **➡️ [Training Setup](./guides/2_TRAINING_SETUP.md)**
    *   Guides you through setting up your Python environment, installing dependencies (like PyTorch with CUDA), choosing a TTS framework, and configuring the training parameters in your configuration file.

3.  **➡️ [Model Training](./guides/3_MODEL_TRAINING.md)**
    *   Explains how to launch the training script, monitor its progress (loss, validation), resume interrupted training, and provides specific tips for fine-tuning existing models.

4.  **➡️ [Inference](./guides/4_INFERENCE.md)**
    *   Details how to use your trained model checkpoint to synthesize speech from new text, including single sentence, batch processing, and multi-speaker considerations.

5.  **➡️ [Packaging and Sharing](./guides/5_PACKAGING_AND_SHARING.md)**
    *   Provides best practices for organizing your trained model files (checkpoints, configs, samples), documenting them with a README, versioning, and preparing them for sharing or archival.

6.  **➡️ [Troubleshooting and Resources](./guides/6_TROUBLESHOOTING_AND_RESOURCES.md)** 
    *   Offers solutions for common problems encountered during training and inference, and lists useful external tools, libraries, and communities.

---

## Conclusion

By following these guides, you'll gain a comprehensive understanding of the workflow for preparing data and training your own Text-to-Speech models. Remember that meticulous data preparation is the foundation for a high-quality voice, and the training process often involves iterative refinement.

Now, proceed to the relevant section based on where you are in your project lifecycle. Good luck building your custom voices! 🚀

## Contributing 

Contributions to improve this guide are welcome! Whether you find typos, inaccuracies, have suggestions for clearer explanations, want to add information about specific tools or frameworks, or have ideas for new sections, your input is valuable.

Please feel free to:

*   **Open an Issue:** To report errors, suggest improvements, or discuss potential changes.
*   **Submit a Pull Request:** For concrete fixes or additions. Please try to ensure your changes are clear and align with the guide's overall structure and tone.

We appreciate any effort to make this guide more accurate, comprehensive, and helpful for the community!

## Glossary of Technical Terms

This glossary explains key technical terms used throughout the guides to help newcomers understand the terminology:

- **ASR (Automatic Speech Recognition)**: Technology that converts spoken language into written text; used for transcribing audio data.
- **Batch Size**: The number of training examples processed together in one forward/backward pass; affects training speed and memory usage.
- **Checkpoint**: A saved snapshot of a model's weights during or after training, allowing you to resume training or use the model for inference.
- **CUDA**: NVIDIA's parallel computing platform that enables GPU acceleration for deep learning tasks.
- **dBFS (Decibels relative to Full Scale)**: A unit of measurement for audio levels in digital systems, where 0 dBFS represents the maximum possible level.
- **Diffusion Models**: A class of generative models that gradually add and then remove noise from data; some recent TTS systems use this approach.
- **FFT (Fast Fourier Transform)**: An algorithm that converts time-domain signals to frequency-domain representations; fundamental for audio processing.
- **Fine-tuning**: The process of taking a pre-trained model and further training it on a smaller, specific dataset to adapt it to a new voice or language.
- **LUFS (Loudness Units relative to Full Scale)**: A standardized measurement of perceived loudness, more representative of human hearing than peak measurements.
- **Manifest File**: A text file that lists audio files and their corresponding transcriptions, used to tell the training script where to find the data.
- **Mel Spectrogram**: A visual representation of audio that approximates human auditory perception by using the mel scale; commonly used as an intermediate representation in TTS systems.
- **Overfitting**: When a model learns the training data too well, including its noise and outliers, resulting in poor performance on new data.
- **Sampling Rate**: The number of audio samples per second (measured in Hz); higher rates capture more audio detail but require more storage and processing power.
- **STFT (Short-Time Fourier Transform)**: A technique that determines the frequency content of local sections of a signal as it changes over time.
- **TTS (Text-to-Speech)**: Technology that converts written text into spoken voice output.
- **Validation Loss**: A metric that measures the error of a model on a validation dataset (data not used for training); helps detect overfitting.
- **VRAM (Video RAM)**: Memory on a graphics card; deep learning models and their intermediate calculations are stored here during training.
- **Vocoder**: A component in some TTS systems that converts acoustic features (like mel spectrograms) into waveforms (actual audio).

## Translation Guide

We welcome translations of this guide to make it accessible to a wider audience. If you'd like to contribute a translation, please follow these steps:

1. **Fork the repository** to your own GitHub account
2. **Create the necessary directory structure** for your language:
   ```
   languages/[language_code]/
   ├── README.md
   └── guides/
       ├── 1_DATA_PREPARATION.md
       ├── 2_TRAINING_SETUP.md
       └── ... (all guide files)
   ```
   Where `[language_code]` is the ISO 639-1 two-letter code for your language (e.g., `es` for Spanish)

3. **Translate the content** starting with the README.md and then the individual guide files
   - Maintain the same file structure and Markdown formatting
   - Keep all code examples unchanged (they should remain in English)
   - Translate all explanatory text, headers, and comments

4. **Update navigation links** to point to the correct files within your language directory

5. **Submit a Pull Request** with your translation

**Important Notes for Translators:**
- Technical terms can be challenging to translate. When in doubt, you can keep the English term followed by a brief explanation in your language.
- Try to maintain the same tone and level of technical detail as the original.
- If you find errors or areas for improvement in the original English content while translating, please open a separate issue to address them.

## [Licence](./LICENCE.md)
The content in this guide is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material as long as you provide appropriate credit. The content is also protected under the copyright of 2025 AcTePuKc and any new contributions will be subject to the same license.