# Guide 3: Model Training & Fine-tuning

**Navigation:** [Main README](./README.md) | [Previous Step: Training Setup (`2_TRAINING_SETUP.md`)](./2_TRAINING_SETUP.md) | [Next Step: Inference (`4_INFERENCE.md`)](./4_INFERENCE.md)

You've prepared your data and configured your training environment. Now it's time to actually train (or fine-tune) your Text-to-Speech model. This phase involves running the training script, monitoring its progress, and understanding how to manage the process.

---

## 5. Running the Training

This section details how to launch, monitor, and manage the training process.

### 5.1. Launching the Training Script

-   **Navigate to the Correct Directory:** Open your terminal or command prompt and navigate into the main directory of the cloned TTS framework repository (the directory containing the `train.py` script or its equivalent).
-   **Activate Virtual Environment:** Ensure your dedicated Python virtual environment (e.g., `venv_tts`, `tts_env`) is activated.
    ```bash
    # Example activation (adjust path/name as needed)
    # Windows: ..\venv_tts\Scripts\activate
    # Linux/macOS: source ../venv_tts/bin/activate
    # Conda: conda activate tts_env
    ```
-   **Execute the Training Command:** Run the framework's training script, pointing it to your custom configuration file created in Guide 2. The exact command structure varies between frameworks. Common patterns include:
    ```bash
    # Common Pattern 1: Using --config argument
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Common Pattern 2: Using -c for config and -m for model/output directory name
    # (Check if your output_directory in the config is overridden by -m)
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Common Pattern 3: Specifying checkpoint directory directly
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```
    *   **Multi-GPU Training:** If you have multiple GPUs and the framework supports distributed training (check its documentation), you might use commands involving `torchrun` or `python -m torch.distributed.launch`. Example:
        ```bash
        # Example using torchrun (adjust nproc_per_node to your GPU count)
        torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
        ```

### 5.2. Monitoring Training Progress

-   **Console Output:** The terminal where you launched the training will display progress information. Look for:
    *   **Initialization:** Messages indicating the model is being built, data loaders are being prepared, and potentially a pre-trained model is being loaded (for fine-tuning).
    *   **Epochs/Steps:** Current training progress (e.g., `Epoch: [1/1000]`, `Step: [500/100000]`).
    *   **Loss Values:** Crucial metrics indicating how well the model is learning. Expect to see `train_loss` (loss on the current batch) and, periodically, `validation_loss` (loss on the unseen validation set). Both should generally decrease over time. Specific loss components (like `mel_loss`, `duration_loss`, `kl_loss`) might also be reported depending on the model architecture.
    *   **Learning Rate:** The current learning rate might be printed, especially if a scheduler is reducing it over time.
    *   **Timestamps/Speed:** Time taken per step or epoch.
-   **TensorBoard (Highly Recommended):** If enabled in your config, use TensorBoard for visual monitoring.
    *   **Launch:** Open a *new* terminal (keep the training one running), activate the same virtual environment, and run:
        ```bash
        # Point logdir to the output_directory specified in your config
        tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
        ```
    *   **Access:** Open the URL provided by TensorBoard (usually `http://localhost:6006/`) in your web browser.
    *   **Visualize:** You can see plots of training and validation losses over time, learning rate schedules, and potentially other metrics.
    *   **Listen to Audio Samples:** Many frameworks log synthesized audio samples from the validation set to TensorBoard periodically (check the `AUDIO` tab). Listening to these is the *best* way to qualitatively assess model improvement and identify issues like noise, mispronunciations, or robotic output.
-   **Output Directory:** Check the `output_directory` you specified in your config (`../checkpoints/my_yoruba_voice_run1`). It should contain:
    *   Saved model checkpoints (`.pth`, `.pt`, `.ckpt` files).
    *   Log files (`train.log`, etc.).
    *   Copies of the configuration file used.
    *   TensorBoard event files (usually in a `logs` or `events` subdirectory).
    *   Possibly synthesized audio samples.

### 5.3. Understanding Checkpoints

-   **What they are:** Checkpoints are snapshots of the model's state (all its learned weights and potentially the optimizer state) saved at specific intervals during training.
-   **Why they matter:**
    *   **Resuming Training:** Allow you to continue training if it's interrupted (due to crashes, power outages, or manual stopping).
    *   **Evaluating Progress:** You can use checkpoints from different stages to synthesize audio and see how the model evolved.
    *   **Selecting the Best Model:** Validation loss helps identify good checkpoints, but the *best* model is often chosen by listening to synthesized audio from several promising checkpoints near the lowest validation loss. Sometimes a slightly earlier checkpoint sounds better than the absolute lowest loss one.
-   **Saving Frequency:** Configure the `save_checkpoint_interval` in your config. Saving too often consumes disk space; saving too infrequently risks losing significant progress if a crash occurs. Saving every few thousand steps or once per epoch is common. Many frameworks also save the "best" checkpoint based on validation loss automatically.

### 5.4. Resuming Interrupted Training

-   If your training stops unexpectedly or you stop it manually, you can usually resume from the last saved checkpoint.
-   Find the path to the latest checkpoint file in your output directory (e.g., `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` or `latest_checkpoint.pth`).
-   Use the framework's resume argument when launching the training script again. The argument name varies:
    ```bash
    # Example using --resume_checkpoint
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

    # Example using --restore_path or --resume_path
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
    ```
-   The script should load the model weights and optimizer state from the checkpoint and continue training from that point.

### 5.5. When to Stop Training

-   **Epoch Limit:** Training stops automatically when the maximum number of `epochs` specified in the config is reached.
-   **Early Stopping:** Monitor the `validation_loss`. If it stops decreasing and starts to consistently increase for a prolonged period (e.g., across several validation intervals), the model might be starting to overfit. You might consider stopping training manually around the point where validation loss was lowest.
-   **Qualitative Assessment:** Regularly listen to the validation audio samples generated in TensorBoard or manually synthesize samples using recent checkpoints. Stop training when you are satisfied with the audio quality and stability, even if the loss is still slightly decreasing. Further training might not yield perceptible improvements or could even degrade quality.

---

## 6. Fine-tuning Specifics

Fine-tuning leverages a powerful pre-trained model and adapts it to your specific dataset (speaker, language, style). It's usually faster and requires less data than training from scratch.

### 6.1. The Goal

-   To transfer the general speech synthesis capabilities (like understanding text-to-sound mapping, basic prosody) from the large dataset the base model was trained on, while specializing the voice identity and potentially accent/style to match your smaller, specific dataset.

### 6.2. Key Configuration Differences (Recap from Setup)

-   **`pretrained_model_path`:** You MUST provide the path to the pre-trained model checkpoint file in your configuration.
-   **`fine_tuning: True`:** Ensure any flag indicating fine-tuning mode is enabled if the framework requires it.
-   **Learning Rate:** Start with a *lower* learning rate than typically used for training from scratch (e.g., `1e-5`, `2e-5`, `5e-5`). A high learning rate can destroy the valuable information learned by the pre-trained model.
-   **Batch Size:** Can often be similar to training from scratch, adjust based on VRAM.
-   **Epochs:** The required number of epochs for fine-tuning is usually significantly less than training from scratch, but still depends on dataset size and desired quality. Monitor validation loss and audio samples closely.

### 6.3. Potential Strategies (Framework Dependent)

-   **Full Network Fine-tuning:** The default approach is often to update weights across the entire network, but with a low learning rate.
-   **Freezing Layers:** Some frameworks allow freezing parts of the network (e.g., the text encoder or duration predictor) initially and only training specific components (like speaker embeddings or the decoder). This can sometimes help preserve the base model's strengths while adapting specific aspects. Check your framework's documentation for `--freeze_layers` or similar options.
-   **Ignoring Layers:** When loading the pre-trained model, you might want to `ignore_layers` (or `reinitialize_layers`) like the final output layer or speaker embedding layer, especially if your dataset has a different number of speakers than the pre-trained model.

### 6.4. Monitoring Fine-tuning

-   **Rapid Initial Improvement:** You should see the validation loss drop relatively quickly initially as the model adapts to the target voice.
-   **Focus on Audio Quality:** Pay close attention to the synthesized validation samples. Is the voice identity shifting towards your target speaker? Is the speech clear and stable? Fine-tuning is often more about perceptual quality than hitting the absolute minimum loss value.

---

With training launched and monitored, the next step, after selecting a good checkpoint, is to use the model for generating speech on new text.

**Next Step:** [Inference (`4_INFERENCE.md`)](./4_INFERENCE.md)