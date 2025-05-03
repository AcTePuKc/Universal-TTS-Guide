# Guide 4: Inference - Generating Speech with Your Model

**Navigation:** [Main README]({{ site.baseurl }}/){: .btn .btn-primary} | [Previous Step: Model Training](./3_MODEL_TRAINING.md){: .btn .btn-primary} | [Next Step: Packaging and Sharing](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

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

## 8. Model Evaluation and Deployment

### 8.1. Evaluating TTS Model Quality

While subjective listening tests are the gold standard for TTS evaluation, there are also objective metrics that can help quantify your model's performance:

#### Objective Evaluation Metrics

| Metric | What It Measures | Tools/Implementation | Interpretation |
|:-------|:-----------------|:---------------------|:---------------|
| **MOS (Mean Opinion Score)** | Overall perceived quality | Human evaluators rate samples on a 1-5 scale | Higher is better; industry standard but requires human raters |
| **PESQ (Perceptual Evaluation of Speech Quality)** | Audio quality compared to reference | Available in Python via `pypesq` | Range: -0.5 to 4.5; higher is better |
| **STOI (Short-Time Objective Intelligibility)** | Speech intelligibility | Available in Python via `pystoi` | Range: 0 to 1; higher is better |
| **Character/Word Error Rate (CER/WER)** | Intelligibility via ASR | Run ASR on synthesized speech and compare to input text | Lower is better; measures if words are pronounced correctly |
| **Mel Cepstral Distortion (MCD)** | Spectral distance from reference | Custom implementation using librosa | Lower is better; typically 2-8 for TTS systems |
| **F0 RMSE** | Pitch accuracy | Custom implementation using librosa | Lower is better; measures pitch contour accuracy |
| **Voicing Decision Error** | Voiced/unvoiced accuracy | Custom implementation | Lower is better; measures if speech/silence is correctly placed |

#### Practical Evaluation Approach

1. **Prepare Test Set**: Create a set of diverse test sentences not seen during training
   ```
   # Example test_sentences.txt
   This is a simple declarative sentence.
   Is this an interrogative sentence?
   Wow! This is an exclamatory sentence!
   This sentence contains numbers like 123 and symbols like %.
   This is a much longer sentence that continues for quite some time, testing the model's ability to maintain coherence and proper prosody across longer utterances with multiple clauses and phrases.
   ```

2. **Generate Samples**: Use your model to synthesize speech for all test sentences

3. **Conduct Listening Tests**: Have multiple listeners rate samples on:
   - Naturalness (1-5 scale)
   - Audio quality/artifacts (1-5 scale)
   - Pronunciation accuracy (1-5 scale)
   - Speaker similarity (1-5 scale, if cloning a specific voice)

4. **Implement Objective Metrics**: This Python snippet demonstrates how to calculate some basic metrics:

   ```python
   import numpy as np
   import librosa
   from pesq import pesq
   from pystoi import stoi
   import torch
   from transformers import pipeline

   def evaluate_tts_sample(generated_audio_path, reference_audio_path=None, original_text=None):
       """Evaluate a TTS sample using various metrics."""
       results = {}
       
       # Load generated audio
       y_gen, sr_gen = librosa.load(generated_audio_path, sr=None)
       
       # Basic audio statistics
       results["duration"] = librosa.get_duration(y=y_gen, sr=sr_gen)
       results["rms_energy"] = np.sqrt(np.mean(y_gen**2))
       
       # If reference audio is available, calculate comparison metrics
       if reference_audio_path:
           y_ref, sr_ref = librosa.load(reference_audio_path, sr=sr_gen)  # Match sampling rates
           
           # Ensure same length for comparison
           min_len = min(len(y_gen), len(y_ref))
           y_gen_trim = y_gen[:min_len]
           y_ref_trim = y_ref[:min_len]
           
           # PESQ (requires 16kHz or 8kHz audio)
           if sr_gen in [8000, 16000]:
               try:
                   results["pesq"] = pesq(sr_gen, y_ref_trim, y_gen_trim, 'wb')
               except Exception as e:
                   results["pesq"] = f"Error: {str(e)}"
           else:
               results["pesq"] = "Requires 8kHz or 16kHz audio"
           
           # STOI
           try:
               results["stoi"] = stoi(y_ref_trim, y_gen_trim, sr_gen, extended=False)
           except Exception as e:
               results["stoi"] = f"Error: {str(e)}"
       
       # If original text is available, perform ASR and calculate WER/CER
       if original_text:
           try:
               # Load ASR model (requires transformers and torch)
               asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
               
               # Transcribe generated audio
               transcription = asr(generated_audio_path)["text"].strip().lower()
               original_text = original_text.strip().lower()
               
               results["transcription"] = transcription
               results["original_text"] = original_text
               
               # Simple character error rate calculation
               def cer(ref, hyp):
                   ref, hyp = ref.lower(), hyp.lower()
                   return levenshtein_distance(ref, hyp) / len(ref)
               
               def levenshtein_distance(s1, s2):
                   if len(s1) < len(s2):
                       return levenshtein_distance(s2, s1)
                   if len(s2) == 0:
                       return len(s1)
                   previous_row = range(len(s2) + 1)
                   for i, c1 in enumerate(s1):
                       current_row = [i + 1]
                       for j, c2 in enumerate(s2):
                           insertions = previous_row[j + 1] + 1
                           deletions = current_row[j] + 1
                           substitutions = previous_row[j] + (c1 != c2)
                           current_row.append(min(insertions, deletions, substitutions))
                       previous_row = current_row
                   return previous_row[-1]
               
               results["character_error_rate"] = cer(original_text, transcription)
           except Exception as e:
               results["asr_error"] = str(e)
       
       return results
   ```

### 8.2. Deploying TTS Models

Once you've trained and evaluated your model, you might want to deploy it for practical use. Here are some deployment options:

#### Production Deployment Considerations

When moving from experimentation to production deployment, consider these important factors:

1. **Model Optimization**
   - **Quantization**: Reduce model precision from FP32 to FP16 or INT8 to decrease size and increase inference speed
   - **Pruning**: Remove unnecessary weights to create smaller, faster models
   - **Knowledge Distillation**: Train a smaller "student" model to mimic your larger "teacher" model
   - **ONNX Conversion**: Convert your PyTorch/TensorFlow model to ONNX format for better cross-platform performance

2. **Latency Optimization**
   - **Batch Processing**: For non-real-time applications, process multiple requests in batches
   - **Streaming Synthesis**: For real-time applications, implement chunk-by-chunk processing
   - **Caching**: Cache frequently requested phrases or phoneme sequences
   - **Hardware Acceleration**: Utilize GPU/TPU for parallel processing or specialized hardware like NVIDIA TensorRT

3. **Scalability**
   - **Containerization**: Package your model and dependencies in Docker containers
   - **Kubernetes**: Orchestrate multiple containers for high availability and load balancing
   - **Auto-scaling**: Automatically adjust resources based on demand
   - **Queue Systems**: Implement request queues (RabbitMQ, Kafka) for handling traffic spikes

4. **Monitoring and Maintenance**
   - **Performance Metrics**: Track latency, throughput, error rates, and resource utilization
   - **Quality Monitoring**: Periodically sample and evaluate output quality
   - **A/B Testing**: Compare different model versions in production
   - **Continuous Training**: Set up pipelines to retrain models with new data

#### Sample Production Deployment Architecture

```
[Client Applications] → [Load Balancer] → [API Gateway]
                                             ↓
[Request Validation] → [Rate Limiting] → [Authentication]
                                             ↓
[Request Queue] → [TTS Worker Pods (Kubernetes)] → [Audio Cache]
                         ↓                              ↑
                  [TTS Model Container]                 |
                         ↓                              |
                  [Audio Post-Processing] → [Audio Storage]
```

#### Local Deployment Options

1. **Command-line Interface**: The simplest approach is to create a script that wraps the inference code:

   ```python
   # tts_cli.py
   import argparse
   import os
   import torch
   
   # Import your model-specific modules here
   # from your_tts_framework import load_model, synthesize_text
   
   def main():
       parser = argparse.ArgumentParser(description="Text-to-Speech CLI")
       parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
       parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
       parser.add_argument("--config", type=str, required=True, help="Path to model config")
       parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
       parser.add_argument("--speaker", type=str, default=None, help="Speaker ID for multi-speaker models")
       args = parser.parse_args()
       
       # Load model (implementation depends on your framework)
       model = load_model(args.model, args.config)
       
       # Synthesize speech
       audio = synthesize_text(model, args.text, speaker_id=args.speaker)
       
       # Save audio
       save_audio(audio, args.output)
       print(f"Audio saved to {args.output}")
   
   if __name__ == "__main__":
       main()
   ```

2. **Simple Web UI**: Create a basic web interface using Flask or Gradio:

   ```python
   # app.py (Flask example)
   from flask import Flask, request, send_file, render_template
   import os
   import torch
   import uuid
   
   # Import your model-specific modules here
   # from your_tts_framework import load_model, synthesize_text
   
   app = Flask(__name__)
   
   # Load model at startup (for faster inference)
   MODEL_PATH = "path/to/best_model.pth"
   CONFIG_PATH = "path/to/config.yaml"
   model = load_model(MODEL_PATH, CONFIG_PATH)
   
   @app.route('/')
   def index():
       return render_template('index.html')
   
   @app.route('/synthesize', methods=['POST'])
   def synthesize():
       text = request.form['text']
       speaker_id = request.form.get('speaker_id', None)
       
       # Generate unique filename
       output_file = f"static/audio/{uuid.uuid4()}.wav"
       os.makedirs(os.path.dirname(output_file), exist_ok=True)
       
       # Synthesize speech
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       
       # Save audio
       save_audio(audio, output_file)
       
       return {'audio_path': output_file}
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000, debug=True)
   ```

3. **Gradio Interface** (Even simpler):

   ```python
   import gradio as gr
   import torch
   
   # Import your model-specific modules here
   # from your_tts_framework import load_model, synthesize_text
   
   # Load model
   MODEL_PATH = "path/to/best_model.pth"
   CONFIG_PATH = "path/to/config.yaml"
   model = load_model(MODEL_PATH, CONFIG_PATH)
   
   def tts_function(text, speaker_id=None):
       # Synthesize speech
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       sampling_rate = 22050  # Adjust to your model's rate
       return (sampling_rate, audio)
   
   # Create Gradio interface
   iface = gr.Interface(
       fn=tts_function,
       inputs=[
           gr.Textbox(lines=3, placeholder="Enter text to synthesize..."),
           gr.Dropdown(choices=["speaker1", "speaker2"], label="Speaker", visible=True)  # For multi-speaker models
       ],
       outputs=gr.Audio(type="numpy"),
       title="Text-to-Speech Demo",
       description="Enter text and generate speech using a custom TTS model."
   )
   
   iface.launch(server_name="0.0.0.0", server_port=7860)
   ```

#### Cloud Deployment Options

For production use, consider these options:

1. **Hugging Face Spaces**: Upload your model to Hugging Face and create a Gradio or Streamlit app
2. **REST API**: Wrap your model in a FastAPI or Flask application and deploy to cloud services
3. **Serverless Functions**: For lightweight models, deploy as serverless functions (AWS Lambda, Google Cloud Functions)
4. **Docker Containers**: Package your model and dependencies in a Docker container for consistent deployment

#### Performance Optimization

To improve inference speed and efficiency:

1. **Model Quantization**: Convert model weights to lower precision (FP16 or INT8)
   ```python
   # Example of FP16 conversion with PyTorch
   model = model.half()  # Convert to FP16
   ```

2. **Model Pruning**: Remove unnecessary weights to create smaller models
3. **ONNX Conversion**: Convert PyTorch models to ONNX format for faster inference
   ```python
   # Example ONNX export
   import torch.onnx
   
   # Export the model
   torch.onnx.export(model,               # model being run
                     dummy_input,         # model input (or a tuple for multiple inputs)
                     "model.onnx",        # where to save the model
                     export_params=True,  # store the trained parameter weights inside the model file
                     opset_version=11,    # the ONNX version to export the model to
                     do_constant_folding=True)  # optimization
   ```

4. **Batch Processing**: Process multiple text inputs at once for higher throughput
5. **Caching**: Cache frequently requested outputs to avoid regeneration

Now that you can generate speech using your trained model, the next logical step is to organize your model files properly for future use, sharing, or deployment.

**Next Step:** [Packaging and Sharing](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 
[Back to Top](#top){: .btn .btn-primary}