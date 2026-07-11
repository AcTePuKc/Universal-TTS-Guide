<a id="glossary-of-technical-terms"></a>
# Glosario de Términos Técnicos

Este glosario explica los términos técnicos clave usados a lo largo de las guías para ayudar a los recién llegados a entender la terminología:

- **ASR (Automatic Speech Recognition)**: Tecnología que convierte el lenguaje hablado en texto escrito; se utiliza para transcribir datos de audio (Reconocimiento Automático del Habla).
- **Batch Size**: El número de ejemplos de entrenamiento que se procesan juntos en un mismo paso hacia adelante/atrás (forward/backward); afecta a la velocidad de entrenamiento y al uso de memoria (tamaño de lote).
<a id="glossary-checkpoint"></a>
- **Checkpoint**: Una instantánea guardada de los pesos de un modelo durante o después del entrenamiento, que permite reanudar el entrenamiento o usar el modelo para inferencia.
<a id="glossary-cuda"></a>
- **CUDA**: La plataforma de computación paralela de NVIDIA que permite la aceleración por GPU para tareas de deep learning.
- **dBFS (Decibels relative to Full Scale)**: Una unidad de medida para los niveles de audio en sistemas digitales, donde 0 dBFS representa el nivel máximo posible (decibelios relativos a la escala completa).
- **Diffusion Models**: Una clase de modelos generativos que gradualmente añaden y luego eliminan ruido de los datos; algunos sistemas TTS recientes usan este enfoque (modelos de difusión).
- **FFT (Fast Fourier Transform)**: Un algoritmo que convierte señales del dominio del tiempo a representaciones del dominio de la frecuencia; fundamental para el procesamiento de audio (Transformada Rápida de Fourier).
- **Fine-tuning**: El proceso de tomar un modelo preentrenado y seguir entrenándolo con un dataset más pequeño y específico para adaptarlo a una nueva voz o idioma (ajuste fino).
- **LUFS (Loudness Units relative to Full Scale)**: Una medida estandarizada de la sonoridad percibida, más representativa de la audición humana que las mediciones de pico.
<a id="glossary-manifest-file"></a>
- **Manifest File**: Un archivo de texto que enumera los archivos de audio y sus transcripciones correspondientes, usado para indicarle al script de entrenamiento dónde encontrar los datos (archivo manifest).
- **Mel Spectrogram**: Una representación visual del audio que aproxima la percepción auditiva humana usando la escala mel; comúnmente usado como representación intermedia en sistemas TTS (espectrograma mel).
- <a id="glossary-overfitting"></a>**Overfitting**: Cuando un modelo aprende demasiado bien los datos de entrenamiento, incluyendo su ruido y valores atípicos, lo que resulta en un mal rendimiento con datos nuevos (sobreajuste).
- <a id="glossary-sampling-rate"></a>**Sampling Rate**: El número de muestras de audio por segundo (medido en Hz); las frecuencias más altas capturan más detalle del audio pero requieren más almacenamiento y potencia de procesamiento (frecuencia de muestreo).
- **STFT (Short-Time Fourier Transform)**: Una técnica que determina el contenido de frecuencia de secciones locales de una señal a medida que cambia en el tiempo (Transformada de Fourier de Tiempo Corto).
- **TTS (Text-to-Speech)**: Tecnología que convierte texto escrito en salida de voz hablada (Texto a Voz).
- <a id="glossary-validation-loss"></a>**Validation Loss**: Una métrica que mide el error de un modelo sobre un dataset de validación (datos no usados para el entrenamiento); ayuda a detectar el overfitting (pérdida de validación).
<a id="glossary-vram"></a>
- **VRAM (Video RAM)**: Memoria de una tarjeta gráfica; los modelos de deep learning y sus cálculos intermedios se almacenan aquí durante el entrenamiento.
- **Vocoder**: Un componente de algunos sistemas TTS que convierte características acústicas (como los espectrogramas mel) en formas de onda (audio real).
