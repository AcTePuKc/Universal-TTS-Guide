# Guide 2: Training Environment Setup & Configuration

**Navigation:** [Main README](/Universal-TTS-Guide/) | [Previous Step: Data Preparation](./1_DATA_PREPARATION.md) | [Next Step: Model Training](./3_MODEL_TRAINING.md)

With your dataset prepared, the next stage involves setting up the necessary software environment and configuring the parameters for your specific training run.

---

## 3. Training Environment Setup

This section covers installing the required software and organizing your project files.

### 3.1. Choose and Clone a TTS Framework

-   **Select a Framework:** Choose a TTS codebase suitable for your goals. Consider factors like:
    *   **Architecture:** VITS, StyleTTS2, Tacotron2+Vocoder, etc. Newer architectures often yield better quality.
    *   **Fine-tuning Support:** Does the framework explicitly support fine-tuning from pre-trained models? This is often easier than training from scratch.
    *   **Language Support:** Check if the model/tokenizer handles your target language well.
    *   **Community & Maintenance:** Is the repository actively maintained? Are there community discussions or support channels?
    *   **Pre-trained Models:** Does the framework provide pre-trained models suitable as a starting point for fine-tuning?
-   **Clone the Repository:** Once chosen, clone the framework's code repository using Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY> # Navigate into the cloned directory
    ```
    *   Example: `git clone https://github.com/some-user/some-tts-framework.git`

### 3.2. Set Up Python Environment & Install Dependencies

-   **Virtual Environment (Recommended):** Create and activate a dedicated Python virtual environment to isolate dependencies and avoid conflicts with other projects or system Python packages.
    *   **Using `venv` (built-in):**
        ```bash
        python -m venv venv_tts  # Create environment named 'venv_tts'
        # Activate it:
        # Windows: .\venv_tts\Scripts\activate
        # Linux/macOS: source venv_tts/bin/activate
        ```
    *   **Using `conda`:**
        ```bash
        conda create --name tts_env python=3.9 # Or desired Python version
        conda activate tts_env
        ```
-   **Install PyTorch with CUDA:** This is critical for GPU acceleration. Visit the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/) and select the options matching your OS, package manager (`pip` or `conda`), compute platform (CUDA version), and desired PyTorch version. **Ensure your installed NVIDIA drivers are compatible with the chosen CUDA version.**
    ```bash
    # Example command using pip for CUDA 11.8 (check PyTorch website for current commands!)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Verify installation:
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    # Should output PyTorch version, True, and your CUDA version if successful.
    ```
-   **Install Framework Requirements:** Most frameworks list their dependencies in a `requirements.txt` file. Install them using `pip` (or `uv`, which is often faster).
    ```bash
    # Navigate to the framework's directory first if you aren't already there
    # Using pip:
    pip install -r requirements.txt

    # Using uv (if installed: pip install uv):
    uv pip install -r requirements.txt
    ```
    *   **Troubleshooting:** Pay attention to any installation errors. They might indicate missing system libraries (like `libsndfile`), incompatible package versions, or issues with your CUDA/PyTorch setup. Check the framework's documentation for specific prerequisites.

### 3.3. Organize Your Project Folder

-   A well-organized folder structure makes managing your project easier. Place your prepared dataset (or create a symbolic link to it) within or alongside the framework's code. A common structure looks like this:

    ```bash
    <YOUR_PROJECT_ROOT>/
    ├── <TTS_REPO_DIRECTORY>/         # The cloned framework code
    │   ├── train.py                 # Main training script (name might vary)
    │   ├── inference.py             # Inference script (name might vary)
    │   ├── configs/                 # Directory for configuration files
    │   │   └── base_config.yaml     # Example framework config
    │   ├── requirements.txt
    │   └── ... (other framework files)
    │
    ├── my_tts_dataset/              # Your prepared dataset from Guide 1
    │   ├── normalized_chunks/       # Final audio files
    │   │   ├── segment_00001.wav
    │   │   └── ...
    │   ├── transcripts/             # Optional: text files if not directly in manifest
    │   ├── train_list.txt           # Training manifest
    │   └── val_list.txt             # Validation manifest
    │
    ├── checkpoints/                 # Create this directory: Where models will be saved
    │   └── my_custom_model/         # Subdirectory for a specific training run
    │
    └── my_configs/                  # Optional: Place your custom configs here
        └── my_training_run_config.yaml
    ```
-   **Paths:** Ensure the paths specified later in your configuration file (for datasets, outputs) are correct relative to where you will *run* the `train.py` script (usually from within the `<TTS_REPO_DIRECTORY>`).

---

## 4. Configuring the Training Run

Before launching the training, you need to create a configuration file that tells the framework *how* to train the model, using *your* specific data.

### 4.1. Find and Copy a Base Configuration

-   **Locate Examples:** Explore the `configs/` directory within the TTS framework. Look for configuration files (`.yaml`, `.json`, or similar) that serve as templates.
-   **Choose Appropriately:** Select a config file that matches your goal:
    *   **Fine-tuning:** Look for names like `config_ft.yaml`, `finetune_*.yaml`. These often assume you'll provide a pre-trained model.
    *   **Training from Scratch:** Look for names like `config_base.yaml`, `train_*.yaml`.
    *   **Dataset Size:** Some frameworks might offer configs tuned for small (`_sm`) or large (`_lg`) datasets.
-   **Copy and Rename:** Copy the chosen template file to a new location (e.g., your own `my_configs/` directory or within the framework's `configs/` directory) and give it a descriptive name for your specific run (e.g., `my_yoruba_voice_ft_config.yaml`).
    ```bash
    # Example: Copying a fine-tuning config
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```

### 4.2. Edit Your Custom Configuration File

-   Open your newly copied configuration file (`my_yoruba_voice_ft_config.yaml`) in a text editor.
-   **Modify Key Parameters:** Carefully review and modify the parameters. Parameter names **will vary significantly** between frameworks, but common categories include:

    ```yaml
    # --- Dataset & Data Loading ---
    # Paths relative to where you run train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt" # Path to your training manifest
    val_filelist_path: "../my_tts_dataset/val_list.txt"   # Path to your validation manifest
    # Some frameworks might need 'data_path' or 'audio_root' pointing to the audio directory instead/additionally.

    # --- Output & Logging ---
    output_directory: "../checkpoints/my_yoruba_voice_run1" # VERY IMPORTANT: Where models, logs, samples are saved. Create this base dir if needed.
    log_interval: 100                  # How often (in steps/batches) to print logs
    validation_interval: 1000          # How often (in steps/batches) to run validation
    save_checkpoint_interval: 5000     # How often (in steps/batches) to save model checkpoints

    # --- Core Training Hyperparameters ---
    epochs: 1000                       # Total number of passes over the training data. Adjust based on dataset size and convergence.
    batch_size: 16                     # Number of samples processed in parallel per GPU. DECREASE if you get CUDA OOM errors. INCREASE for faster training if VRAM allows.
    learning_rate: 1e-4                # Initial learning rate. May need tuning (e.g., lower for fine-tuning: 5e-5 or 1e-5).
    # lr_scheduler: "cosine_decay"     # Learning rate schedule (e.g., step decay, exponential decay) - framework dependent
    # weight_decay: 0.01               # Regularization parameter

    # --- Audio Parameters ---
    sampling_rate: 22050               # CRITICAL: MUST match the sampling rate of your prepared dataset (from Guide 1).
    # Other audio params (often depend on model architecture):
    # filter_length: 1024              # FFT size for STFT
    # hop_length: 256                  # Hop size for STFT
    # win_length: 1024                 # Window size for STFT
    # n_mel_channels: 80               # Number of Mel bands
    # mel_fmin: 0.0                    # Minimum Mel frequency
    # mel_fmax: 8000.0                 # Maximum Mel frequency (often sampling_rate / 2)

    # --- Model Architecture ---
    # model_type: "VITS"               # Type of model architecture
    # hidden_channels: 192             # Size of internal layers
    # num_speakers: 1                  # Set to >1 for multi-speaker datasets (must match data prep)

    # --- Fine-tuning Specifics (If Applicable) ---
    # Set 'True' or provide path when fine-tuning
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth" # Path to the pre-trained checkpoint to start from.
    # Optional: Specify layers to ignore/reinitialize if needed
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```
-   **Read Framework Docs:** Consult the specific documentation of your chosen TTS framework to understand what each parameter in its configuration file does.

### 4.3. Hardware and Dataset Considerations

-   **GPU VRAM:** The `batch_size` is the primary knob to control GPU memory usage. Start with a recommended value (e.g., 16 or 32) and decrease it if you encounter "CUDA out of memory" errors during training startup. Larger batch sizes generally lead to faster convergence but require more VRAM.
-   **Dataset Size vs. Epochs:**
    *   **Small Datasets (< 20h):** May require fewer epochs (e.g., 300-1500) but need careful monitoring via validation loss/samples to avoid overfitting (where the model memorizes the training data but performs poorly on new text). Consider lower learning rates.
    *   **Large Datasets (> 50h):** Can benefit from more epochs (1000+) to fully learn the patterns in the data.
-   **CPU:** While the GPU does the heavy lifting, a decent multi-core CPU is needed for data loading and pre-processing, which can become a bottleneck otherwise.
-   **Storage:** Ensure you have enough disk space for the dataset, the Python environment, the framework code, and especially the saved checkpoints, which can become large (hundreds of MBs to GBs per checkpoint).

### 4.4. Monitoring Tools (TensorBoard)

-   Most modern TTS frameworks integrate with [TensorBoard](https://www.tensorflow.org/tensorboard) for visualizing training progress.
-   The configuration file often has settings related to logging (e.g., `use_tensorboard: True`, `log_directory`).
-   During training, you can typically launch TensorBoard by running `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (e.g., `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) in a separate terminal. This allows you to monitor loss curves, learning rates, and potentially listen to synthesized validation samples in your web browser.

---

With your environment set up and configuration file tailored to your data and goals, you are now ready to start the actual model training process.

**Next Step:** [Model Training (`3_MODEL_TRAINING.md`)](./3_MODEL_TRAINING.md)