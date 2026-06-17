# Guía 6: Resolución de Problemas y Recursos

**Navegación:** [README Principal]({{ site.baseurl }}/languages/es/){: .btn .btn-primary} | [Paso Anterior: Empaquetado y Compartición](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

Esta guía proporciona soluciones para los problemas comunes que surgen durante el proceso de preparación de datos, entrenamiento e inferencia de TTS, junto con una lista de herramientas y recursos útiles.

---

## 8. Resolución de Problemas Comunes

Consulta esta tabla cuando encuentres problemas. Los problemas suelen tener su origen en la calidad de los datos o en los ajustes de configuración.

| Categoría del Problema         | Problema Específico                                      | Causas Posibles y Soluciones                                                                                                                                                              | Guía(s) Relevante(s)                                                                 |
| :----------------------- | :-------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Preparación de Datos**   | Errores de script durante la segmentación/normalización       | Rutas de archivo incorrectas; formato de audio inicialmente no soportado; dependencias faltantes (`ffmpeg`, `pydub`); audio extremadamente ruidoso/silencioso que confunde la detección de silencio. **Comprueba las rutas del script, instala las dependencias, ajusta los parámetros de silencio.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md)                                  |
|                          | La generación del manifest omite muchos archivos                | Nombres de archivo no coincidentes entre el audio y las transcripciones; archivos de transcripción vacíos; rutas incorrectas especificadas en el script; codificación no UTF-8 en los archivos de texto. **Verifica los nombres, comprueba las rutas, asegúrate de que los archivos de texto tengan contenido y codificación UTF-8.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md)                                  |
| **Configuración del Entrenamiento**     | `pip install` falla                               | Librerías del sistema faltantes (p. ej., `libsndfile-dev`); versión de Python incompatible; problemas de red; conflictos entre paquetes. **Lee los mensajes de error con atención, instala las librerías del sistema, usa un entorno virtual, consulta la documentación del framework para los prerrequisitos.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md)                                    |
|                          | PyTorch `cuda is not available`                   | Versión de PyTorch incorrecta instalada (solo CPU); versión incompatible del driver de NVIDIA/toolkit de CUDA; GPU no detectada por el SO. **Reinstala PyTorch con la versión correcta de CUDA desde el sitio oficial, actualiza los drivers.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md)                                    |
| **Ejecución del Entrenamiento** | Error CUDA Out-of-Memory (OOM) al inicio/durante el entrenamiento | `batch_size` demasiado grande para la VRAM de la GPU; arquitectura del modelo demasiado compleja; fuga de memoria (memory leak) en el framework/código personalizado. **Reduce el `batch_size` en la configuración; habilita la Precisión Mixta Automática (AMP/FP16) si está disponible; comprueba si hay actualizaciones del framework.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | El Training Loss es `NaN` o diverge (explota)     | Tasa de aprendizaje demasiado alta; gradientes inestables; batch de datos defectuoso (p. ej., audio/texto corrupto); problemas de precisión numérica. **Reduce la tasa de aprendizaje; comprueba la calidad de los datos; usa recorte de gradiente (gradient clipping) (a menudo habilitado por defecto); prueba FP32 si usas AMP/FP16.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | El Training Loss se estanca (no disminuye)        | Tasa de aprendizaje demasiado baja; mala calidad/variedad de datos; modelo atascado en un mínimo local; configuración incorrecta del modelo. **Aumenta ligeramente la tasa de aprendizaje; mejora/aumenta los datos; comprueba la configuración (esp. los parámetros de audio); prueba un optimizador diferente.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | El Validation Loss aumenta mientras el Training Loss disminuye (Overfitting) | El modelo memoriza los datos de entrenamiento; conjunto de validación insuficiente/no representativo; entrenamiento durante demasiado tiempo. **Detén el entrenamiento de forma temprana (basándote en el mejor val loss); añade más datos de entrenamiento diversos; usa regularización (weight decay, dropout - comprueba la configuración); mejora el conjunto de validación.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
| **Calidad de Inferencia**  | La salida suena robótica/monótona                   | Entrenamiento insuficiente; mala prosodia en los datos de entrenamiento; limitaciones de la arquitectura del modelo; problemas de normalización del texto. **Entrena durante más tiempo; mejora la variedad/calidad de los datos; prueba una arquitectura de modelo diferente; asegúrate de que el texto esté bien puntuado/normalizado.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md), [4_INFERENCE.md](./4_INFERENCE.md) |
|                          | La salida es ruidosa/distorsionada/ininteligible            | Mala calidad de los datos (ruido incorporado); el modelo no convergió; desajuste entre la configuración de entrenamiento y la configuración/checkpoint de inferencia; sampling rate incorrecto usado en la inferencia. **Limpia rigurosamente los datos de entrenamiento; entrena durante más tiempo; asegura una coincidencia EXACTA de configuración/checkpoint; verifica los parámetros de audio.** | Todas las Guías                                                                        |
|                          | La salida suena como el locutor incorrecto (fine-tuning) | Modelo preentrenado no cargado correctamente; tasa de aprendizaje demasiado alta inicialmente; datos/pasos de fine-tuning insuficientes; desajuste del speaker ID. **Verifica `pretrained_model_path` e `ignore_layers` en la configuración; usa una LR más baja para el fine-tuning; entrena durante más tiempo; comprueba el speaker ID.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md), [4_INFERENCE.md](./4_INFERENCE.md) |
|                          | La inferencia se corta antes de tiempo o habla demasiado rápido/lento  | Limitación del modelo (predicción de duración); ajuste de inferencia que limita la longitud máxima de salida; parámetro de length scale/velocidad incorrecto. **Consulta la documentación del framework para los ajustes de pasos máximos del decodificador / longitud máxima; ajusta los parámetros de control de velocidad.** | [4_INFERENCE.md](./4_INFERENCE.md)                                                |
| **Uso del Modelo**        | No se puede cargar el archivo de checkpoint                       | Descarga/archivo corrupto; uso de un checkpoint con una versión del framework o archivo de configuración incompatible; ruta de archivo incorrecta. **Vuelve a descargar/verifica la integridad del archivo; usa la configuración correcta; asegúrate de que la versión del framework coincida con la usada para el entrenamiento; comprueba la ruta.** | [5_PACKAGING_AND_SHARING.md](./5_PACKAGING_AND_SHARING.md), [4_INFERENCE.md](./4_INFERENCE.md) |

---

## 10. Recursos y Herramientas Útiles

Esta lista incluye software, librerías y comunidades útiles para proyectos de TTS.

### Procesamiento y Análisis de Audio:

*   **[Audacity](https://www.audacityteam.org/):** Editor de audio gratuito, de código abierto y multiplataforma. Excelente para la inspección manual, limpieza, etiquetado y procesamiento básico de archivos de audio.
*   **[FFmpeg](https://ffmpeg.org/):** La herramienta de línea de comandos multiusos (navaja suiza) para conversión de audio/vídeo, remuestreo, manipulación de canales, cambios de formato y mucho más. Esencial para automatizar operaciones por lotes.
*   **[SoX (Sound eXchange Compiled)](http://sox.sourceforge.net/) o [Sox - Source Code](https://codeberg.org/sox_ng/sox_ng/):** Utilidad de línea de comandos para el procesamiento de audio. Útil para efectos, conversión de formato y obtención de información del audio (comando `soxi`).
*   **[pydub](https://github.com/jiaaro/pydub):** Librería de Python para una fácil manipulación de audio (corte, conversión de formato, ajuste de volumen, detección de silencio). Usa el backend de ffmpeg/libav.
*   **[librosa](https://librosa.org/doc/latest/index.html):** Librería de Python para análisis avanzado de audio, extracción de características (como los espectrogramas mel) y visualización. A menudo usada internamente por los frameworks de TTS.
*   **[soundfile](https://python-soundfile.readthedocs.io/en/latest/):** Librería de Python para leer/escribir archivos de audio, basada en libsndfile. Soporta muchos formatos.
*   **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm):** Librería de Python para la normalización de sonoridad (LUFS), generalmente preferida sobre la simple normalización de pico para una consistencia percibida.

### Transcripción (ASR):

*   **[OpenAI Whisper](https://github.com/openai/whisper):** Modelo ASR de código abierto y alta calidad, soporta muchos idiomas. Buena base de referencia, pero la puntuación a menudo necesita revisión. Puede ejecutarse localmente (se recomienda GPU) o vía API. Existen varias implementaciones de la comunidad.
*   **[Modelos Google Gemini (vía API/AI Studio)](https://ai.google.dev/):** Modelos capaces para la transcripción, a menudo rinden bien con audio claro, potencialmente mejor en segmentos previamente segmentados. Comprueba la API/Studio para conocer las capacidades actuales y los precios/niveles gratuitos.
*   **Servicios de ASR en la Nube:**
    *   [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
    *   [AWS Transcribe](https://aws.amazon.com/transcribe/)
    *   [Azure Speech Service](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/)
    *   *A menudo fiables, de pago por uso, pueden tener cuotas gratuitas iniciales.*
*   **[Hugging Face Transformers - Modelos ASR](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition):** Hub para muchos modelos ASR preentrenados, incluyendo versiones con fine-tuning de Whisper y otros. Explora modelos con fine-tuning para idiomas específicos o mejora de la puntuación.
*   **[ElevenLabs Speech To Text (Scribe)](https://elevenlabs.io/speech-to-text):** *Servicio Comercial.* Conocido por una precisión muy alta tanto en la transcripción como en la puntuación, pero es un servicio de pago y puede ser relativamente caro en comparación con otros. Vale la pena considerarlo si el presupuesto lo permite y se requiere la máxima precisión lista para usar.

### Frameworks y Bases de Código de TTS (Ejemplos - Busca forks/sucesores activos):

*   **[StyleTTS2 (Repo de Investigación)](https://github.com/yl4579/StyleTTS2):** Trabajo influyente sobre el control de estilo. Busca forks mantenidos activamente que hayan implementado pipelines de entrenamiento/inferencia.
*   **[VITS (Repo de Investigación)](https://github.com/jaywalnut310/vits):** Arquitectura popular de extremo a extremo. Existen muchos forks e implementaciones.
*   **[Coqui TTS (Archivado)](https://github.com/coqui-ai/TTS):** Fue una librería muy popular y completa. Aunque está archivada, su base de código y sus conceptos siguen siendo influyentes. Es posible que existan muchos forks activos.
*   **[ESPnet](https://github.com/espnet/espnet):** Kit de herramientas de procesamiento de voz de extremo a extremo, que incluye recetas de TTS para varios modelos. Más orientado a la investigación.
*   **Busca en GitHub:** Usa palabras clave como "TTS", "VITS training", "StyleTTS2 training", "PyTorch TTS" para encontrar proyectos actuales.

### Entorno de Python y Deep Learning:

*   **[Python](https://www.python.org/):** El lenguaje de programación central.
*   **[PyTorch](https://pytorch.org/):** La principal librería de deep learning usada por la mayoría de los frameworks modernos de TTS.
*   **[TensorBoard](https://www.tensorflow.org/tensorboard):** Esencial para visualizar el progreso del entrenamiento (también funciona con PyTorch).
*   **[pip](https://pip.pypa.io/en/stable/) / [uv](https://github.com/astral-sh/uv):** Instaladores de paquetes de Python. `uv` es una alternativa más nueva y a menudo mucho más rápida.
*   **[conda](https://docs.conda.io/en/latest/) / [venv](https://docs.python.org/3/library/venv.html):** Herramientas para crear entornos de Python aislados.
*   **[Git](https://git-scm.com/):** Sistema de control de versiones, esencial para clonar repositorios y gestionar código.
*   **[Hugging Face Hub](https://huggingface.co/):** Plataforma para compartir modelos (incluyendo TTS), datasets y código.

### Comunidades:

*   **GitHub Discussions/Issues de los Frameworks de TTS:** Consulta el repositorio específico que estés usando para preguntas y respuestas de la comunidad.
*   **Servidores de Discord:** Muchas comunidades de IA/ML (como LAION, EleutherAI, servidores de frameworks específicos) tienen canales dedicados a TTS.
*   **Reddit:** Subreddits como r/SpeechSynthesis, r/MachineLearning.

---

Esto concluye la serie principal de guías. Recuerda que construir buenos modelos TTS a menudo implica iteración: revisar la preparación de datos o ajustar los parámetros de entrenamiento según los resultados es una práctica común. ¡Mucha suerte!

---
**Navegación:** [README Principal]({{ site.baseurl }}/languages/es/){: .btn .btn-primary} | [Paso Anterior: Empaquetado y Compartición](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | [Volver Arriba](#top){: .btn .btn-primary}
