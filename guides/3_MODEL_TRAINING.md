# Guide 3: Model Training & Fine-tuning

**Navigation:** [Main README]({{ site.baseurl }}/){: .btn .btn-primary} | [Previous Step: Training Setup](./2_TRAINING_SETUP.md){: .btn .btn-primary} | [Next Step: Inference](./4_INFERENCE.md){: .btn .btn-primary}

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

## 6. Fine-tuning vs. Training from Scratch

### 6.1. Choosing Your Approach

When starting a TTS project, one of the most important decisions is whether to fine-tune an existing model or train a new one from scratch. This table helps you decide which approach is best for your specific situation:

| Factor | Fine-tuning | Training from Scratch |
|:-------|:------------|:----------------------|
| **Dataset Size** | Works well with smaller datasets (5-20 hours)<br>Can produce good results with as little as 1-2 hours for some voices | Typically requires larger datasets (30+ hours)<br>Less than 20 hours often leads to poor quality |
| **Voice Similarity** | Best when your target voice is similar to voices in the pre-trained model's training data | Necessary when your target voice is very unique or significantly different from available pre-trained models |
| **Language** | Works well if fine-tuning within the same language<br>Can work for cross-lingual with careful preparation | Required for languages with no available pre-trained models<br>Better for capturing language-specific phonetics |
| **Training Time** | Much faster (days instead of weeks)<br>Requires fewer epochs to converge | Significantly longer training time<br>May require 2-5x more epochs |
| **Hardware Requirements** | Similar GPU requirements but for less time<br>Can often use smaller batch sizes | Needs sustained GPU access for longer periods<br>May benefit more from multi-GPU setups |
| **Quality Potential** | Can achieve excellent quality quickly<br>May inherit limitations of the base model | Maximum flexibility and potential quality<br>No constraints from previous training |
| **Stability** | Generally more stable training process<br>Less prone to collapse or non-convergence | More sensitive to hyperparameters<br>Higher risk of training instability |

#### When to Choose Fine-tuning

Fine-tuning is generally recommended when:
- You have limited data (less than 20 hours)
- You need faster results
- Your target voice/language is reasonably similar to available pre-trained models
- You have limited computational resources
- You're new to TTS training (fine-tuning is more forgiving)

#### When to Choose Training from Scratch

Training from scratch is better when:
- You have abundant data (30+ hours)
- Your target voice is highly unique or has characteristics not represented in pre-trained models
- You're working with a language that's poorly supported by existing models
- You need maximum control over all aspects of the model
- You have access to significant computational resources
- You're building a foundation model that others will fine-tune

### 6.2. Fine-tuning Specifics

Fine-tuning leverages a powerful pre-trained model and adapts it to your specific dataset (speaker, language, style). It's usually faster and requires less data than training from scratch.

#### The Goal

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

## 7. Comprehensive Troubleshooting Guide

Training TTS models can be challenging, with many potential issues. This section provides solutions for common problems you might encounter.

### 7.1. Common Error Messages and Solutions

| Error Message | Possible Causes | Solutions |
|:--------------|:----------------|:----------|
| `CUDA out of memory` | • Batch size too large<br>• Model too large for GPU<br>• Memory leak | • Reduce batch size<br>• Enable gradient checkpointing<br>• Use mixed precision training<br>• Reduce sequence length |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | • Incorrect data type in dataset<br>• Incompatible tensor types | • Check data preprocessing<br>• Ensure all tensors have correct dtype<br>• Add explicit type conversion |
| `ValueError: too many values to unpack` | • Mismatch between model outputs and loss function expectations<br>• Incorrect data format | • Check model output structure<br>• Verify loss function implementation<br>• Debug data loader outputs |
| `FileNotFoundError: [Errno 2] No such file or directory` | • Incorrect paths in config<br>• Missing data files | • Verify all file paths<br>• Check manifest file integrity<br>• Ensure data is downloaded/extracted |
| `KeyError: 'speaker_id'` | • Missing speaker information<br>• Incorrect dataset format | • Check dataset format<br>• Verify speaker mapping file<br>• Add speaker information to manifest |
| `Loss is NaN` | • Learning rate too high<br>• Unstable initialization<br>• Gradient explosion | • Reduce learning rate<br>• Add gradient clipping<br>• Check for division by zero<br>• Normalize input data |
| `ModuleNotFoundError: No module named 'X'` | • Missing dependency<br>• Environment issue | • Install missing package<br>• Check virtual environment<br>• Verify package versions |
| `RuntimeError: expected scalar type Float but found Double` | • Inconsistent tensor types | • Add `.float()` to tensors<br>• Check data preprocessing<br>• Standardize dtype across model |

### 7.2. Training Quality Issues

| Symptom | Possible Causes | Solutions |
|:--------|:----------------|:----------|
| **Robotic/Buzzy Audio** | • Vocoder issues<br>• Insufficient training<br>• Poor audio preprocessing | • Train vocoder longer<br>• Check audio normalization<br>• Verify sampling rate consistency |
| **Word Skipping/Repetition** | • Attention problems<br>• Unstable training<br>• Insufficient data | • Use guided attention loss<br>• Add more data variety<br>• Reduce learning rate<br>• Check for long silences in data |
| **Incorrect Pronunciation** | • Text normalization issues<br>• Phoneme errors<br>• Language mismatch | • Improve text preprocessing<br>• Use phoneme-based input<br>• Add pronunciation dictionary |
| **Speaker Identity Loss** | • Overfitting to dominant speaker<br>• Weak speaker embeddings<br>• Insufficient speaker data | • Balance speaker data<br>• Increase speaker embedding dim<br>• Use speaker adversarial loss |
| **Slow Convergence** | • Learning rate issues<br>• Poor initialization<br>• Complex dataset | • Try different LR schedules<br>• Use transfer learning<br>• Simplify dataset initially |
| **Unstable Training** | • Batch variance<br>• Outliers in dataset<br>• Optimizer issues | • Use gradient accumulation<br>• Clean outlier samples<br>• Try different optimizers |

### 7.3. Framework-Specific Issues

#### Coqui TTS
```
# Error: "RuntimeError: Error in applying gradient to param_name"
# Solution: Check for NaN values in your dataset or reduce learning rate
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # Run before training to debug

# Error: "ValueError: Tacotron training requires `r` > 1"
# Solution: Set reduction factor correctly in config
# Example fix in config.json:
"r": 2  # Try values between 2-5
```

#### ESPnet
```
# Error: "TypeError: forward() missing 1 required positional argument: 'feats'"
# Solution: Check data formatting and ensure feats are provided
# Debug data loading:
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS/StyleTTS
```
# Error: "RuntimeError: expected scalar type Half but found Float"
# Solution: Ensure consistent precision throughout model
# Add to your training script:
model = model.half()  # If using mixed precision
# OR
model = model.float()  # If not using mixed precision
```

### 7.4. Hardware and Environment Issues

1. **GPU Memory Fragmentation**
   - **Symptom**: OOM errors after training for several hours despite sufficient VRAM
   - **Solution**: Periodically restart training from checkpoint, use smaller batches

2. **CPU Bottlenecks**
   - **Symptom**: GPU utilization fluctuates or stays low
   - **Solution**: Increase num_workers in DataLoader, use faster storage, pre-cache datasets

3. **Disk I/O Bottlenecks**
   - **Symptom**: Training stalls periodically during data loading
   - **Solution**: Use SSD storage, increase prefetch factor, cache dataset in RAM

4. **Environment Conflicts**
   - **Symptom**: Mysterious crashes or import errors
   - **Solution**: Use isolated environments (conda/venv), check CUDA/PyTorch compatibility

### 7.5. Debugging Strategies

1. **Isolate the Problem**
   ```bash
   # Test data loading separately
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"
   
   # Test forward pass with dummy data
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Simplify to Identify Issues**
   - Train on a tiny subset (10-20 samples)
   - Disable data augmentation temporarily
   - Try with a single speaker first

3. **Visualize Intermediate Outputs**
   - Plot attention alignments
   - Visualize mel spectrograms at different stages
   - Monitor gradient norms

4. **Enable Verbose Logging**
   ```bash
   # Add to your training script
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

5. **Use TensorBoard Profiling**
   ```python
   # Add to your training code
   from torch.profiler import profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Your forward pass
   print(prof.key_averages().table())
   ```

---

With training launched and monitored, the next step, after selecting a good checkpoint, is to use the model for generating speech on new text.

**Next Step:** [Inference](./4_INFERENCE.md){: .btn .btn-primary} | 
[Back to Top](#top){: .btn .btn-primary}