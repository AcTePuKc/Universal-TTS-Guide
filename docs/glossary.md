# Glossary
<a id="glossary-of-technical-terms"></a>

This glossary explains key technical terms used throughout the guide.

This glossary explains key technical terms used throughout the guides to help newcomers understand the terminology:

- <a id="glossary-asr"></a>**ASR (Automatic Speech Recognition)**: Technology that converts spoken language into written text; used for transcribing audio data.
- <a id="glossary-batch-size"></a>**Batch Size**: The number of training examples processed together in one forward/backward pass; affects training speed and memory usage.
- <a id="glossary-checkpoint"></a>**Checkpoint**: A saved snapshot of a model's weights during or after training, allowing you to resume training or use the model for inference.
- <a id="glossary-cuda"></a>**CUDA**: NVIDIA's parallel computing platform that enables GPU acceleration for deep learning tasks.
- <a id="glossary-dbfs"></a>**dBFS (Decibels relative to Full Scale)**: A unit of measurement for audio levels in digital systems, where 0 dBFS represents the maximum possible level.
- <a id="glossary-diffusion-models"></a>**Diffusion Models**: A class of generative models that gradually add and then remove noise from data; some recent TTS systems use this approach.
- <a id="glossary-fft"></a>**FFT (Fast Fourier Transform)**: An algorithm that converts time-domain signals to frequency-domain representations; fundamental for audio processing.
- <a id="glossary-fine-tuning"></a>**Fine-tuning**: The process of taking a pre-trained model and further training it on a smaller, specific dataset to adapt it to a new voice or language.
- <a id="glossary-lufs"></a>**LUFS (Loudness Units relative to Full Scale)**: A standardized measurement of perceived loudness, more representative of human hearing than peak measurements.
- <a id="glossary-manifest-file"></a>**Manifest File**: A text file that lists audio files and their corresponding transcriptions, used to tell the training script where to find the data.
- <a id="glossary-mel-spectrogram"></a>**Mel Spectrogram**: A visual representation of audio that approximates human auditory perception by using the mel scale; commonly used as an intermediate representation in TTS systems.
- <a id="glossary-overfitting"></a>**Overfitting**: When a model learns the training data too well, including its noise and outliers, resulting in poor performance on new data.
- <a id="glossary-sampling-rate"></a>**Sampling Rate**: The number of audio samples per second (measured in Hz); higher rates capture more audio detail but require more storage and processing power.
- <a id="glossary-stft"></a>**STFT (Short-Time Fourier Transform)**: A technique that determines the frequency content of local sections of a signal as it changes over time.
- <a id="glossary-tts"></a>**TTS (Text-to-Speech)**: Technology that converts written text into spoken voice output.
- <a id="glossary-validation-loss"></a>**Validation Loss**: A metric that measures the error of a model on a validation dataset (data not used for training); helps detect overfitting.
- <a id="glossary-vram"></a>**VRAM (Video RAM)**: Memory on a graphics card; deep learning models and their intermediate calculations are stored here during training.
- <a id="glossary-vocoder"></a>**Vocoder**: A component in some TTS systems that converts acoustic features (like mel spectrograms) into waveforms (actual audio).

<a id="translation-guide"></a>
