# Universal TTS Model Training & Dataset Preparation Guide

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

1.  **➡️ [Data Preparation (`1_DATA_PREPARATION.md`)](./1_DATA_PREPARATION.md)**
    *   Covers acquiring, cleaning, segmenting, normalizing audio, transcribing text, and creating the necessary manifest files for training. Includes the crucial data quality checklist.

2.  **➡️ [Training Setup (`2_TRAINING_SETUP.md`)](./2_TRAINING_SETUP.md)**
    *   Guides you through setting up your Python environment, installing dependencies (like PyTorch with CUDA), choosing a TTS framework, and configuring the training parameters in your configuration file.

3.  **➡️ [Model Training (`3_MODEL_TRAINING.md`)](./3_MODEL_TRAINING.md)**
    *   Explains how to launch the training script, monitor its progress (loss, validation), resume interrupted training, and provides specific tips for fine-tuning existing models.

4.  **➡️ [Inference (`4_INFERENCE.md`)](./4_INFERENCE.md)**
    *   Details how to use your trained model checkpoint to synthesize speech from new text, including single sentence, batch processing, and multi-speaker considerations.

5.  **➡️ [Packaging and Sharing (`5_PACKAGING_AND_SHARING.md`)](./5_PACKAGING_AND_SHARING.md)**
    *   Provides best practices for organizing your trained model files (checkpoints, configs, samples), documenting them with a README, versioning, and preparing them for sharing or archival.

6.  **➡️ [Troubleshooting and Resources (`6_TROUBLESHOOTING_AND_RESOURCES.md`)](./6_TROUBLESHOOTING_AND_RESOURCES.md)**
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

## [Licence](./LICENCE)
The content in this guide is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material as long as you provide appropriate credit. The content is also protected under the copyright of 2025 AcTePuKc and any new contributions will be subject to the same license.