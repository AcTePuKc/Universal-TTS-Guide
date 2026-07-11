# Guía de Resolución de Problemas y Recursos para TTS


Esta guía proporciona soluciones para los problemas comunes que surgen durante el proceso de preparación de datos, entrenamiento e inferencia de TTS, junto con una lista de herramientas y recursos útiles.

Si algún término de esta guía no te resulta familiar, consulta el [glosario](../glossary.md#glossary-of-technical-terms). Resolver problemas es más rápido cuando puedes distinguir con seguridad entre [checkpoint](../glossary.md#glossary-checkpoint), [archivo manifest](../glossary.md#glossary-manifest-file), [CUDA](../glossary.md#glossary-cuda) y [VRAM](../glossary.md#glossary-vram) sin tener que adivinar.

---

## Resolución de Problemas Comunes

Consulta esta tabla cuando encuentres problemas. Los problemas suelen tener su origen en la calidad de los datos o en los ajustes de configuración.

Antes de cambiar cinco ajustes a la vez, revisa lo básico en este orden:

1. confirma las rutas, los nombres de archivo y la estructura de carpetas
2. confirma que el checkpoint y la configuración realmente pertenecen a la misma ejecución de entrenamiento
3. confirma que los ajustes de audio coinciden con el entrenamiento, especialmente el sampling rate
4. confirma que el entorno es el que crees: entorno de Python correcto, versiones de dependencias y visibilidad de CUDA

Ese orden detecta una gran parte de los errores de principiantes e intermedios antes de empezar una depuración más profunda.

| Categoría del Problema | Problema Específico | Causas Posibles y Soluciones | Guía(s) Relevante(s) |
| :--- | :--- | :--- | :--- |
| **Preparación de Datos** | Errores de script durante la segmentación o normalización | Rutas de archivo incorrectas; formato de audio inicialmente no soportado; dependencias faltantes (`ffmpeg`, `pydub`); audio extremadamente ruidoso o silencioso que confunde la detección de silencio. **Comprueba las rutas del script, instala las dependencias y ajusta los parámetros de silencio.** | [1_DATA_PREPARATION.md](./1-data-preparation.md) |
| **Preparación de Datos** | La generación del manifest omite muchos archivos | Nombres de archivo no coincidentes entre el audio y las transcripciones; archivos de transcripción vacíos; rutas incorrectas especificadas en el script; archivos de texto sin codificación UTF-8. **Verifica los nombres, comprueba las rutas y asegúrate de que los archivos de texto tengan contenido y codificación UTF-8.** | [1_DATA_PREPARATION.md](./1-data-preparation.md) |
| **Configuración del Entrenamiento** | `pip install` falla | Librerías del sistema faltantes como `libsndfile-dev`; versión de Python incompatible; problemas de red; conflictos entre paquetes. **Lee los mensajes de error con atención, instala las librerías del sistema, usa un entorno virtual y consulta la documentación del framework para los prerrequisitos.** | [2_TRAINING_SETUP.md](./2-training-setup.md) |
| **Configuración del Entrenamiento** | PyTorch `cuda is not available` | Versión de PyTorch incorrecta instalada (solo CPU); versión incompatible del driver de NVIDIA o del toolkit de CUDA; GPU no detectada por el sistema operativo. **Reinstala PyTorch con la versión correcta de CUDA desde el sitio oficial y actualiza los drivers.** | [2_TRAINING_SETUP.md](./2-training-setup.md) |
| **Ejecución del Entrenamiento** | Error CUDA Out-of-Memory (OOM) al inicio o durante el entrenamiento | `batch_size` demasiado grande para la VRAM de la GPU; arquitectura del modelo demasiado compleja; fuga de memoria en el framework o en código personalizado. **Reduce `batch_size` en la configuración, habilita AMP/FP16 si está disponible y comprueba si hay actualizaciones del framework.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Ejecución del Entrenamiento** | El Training Loss es `NaN` o diverge | Tasa de aprendizaje demasiado alta; gradientes inestables; lote de datos defectuoso; problemas de precisión numérica. **Reduce la tasa de aprendizaje, comprueba la calidad de los datos, usa recorte de gradiente y prueba FP32 si usas AMP/FP16.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Ejecución del Entrenamiento** | El Training Loss se estanca | Tasa de aprendizaje demasiado baja; mala calidad o variedad de datos; modelo atascado en un mínimo local; configuración incorrecta. **Aumenta ligeramente la tasa de aprendizaje, mejora o amplía los datos, revisa la configuración y prueba un optimizador diferente.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Ejecución del Entrenamiento** | El Validation Loss aumenta mientras el Training Loss disminuye | El modelo memoriza los datos de entrenamiento; el conjunto de validación es insuficiente o poco representativo; el entrenamiento dura demasiado. **Detén el entrenamiento antes, añade datos más diversos, usa regularización y mejora el conjunto de validación.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Calidad de Inferencia** | La salida suena robótica o monótona | Entrenamiento insuficiente; mala prosodia en los datos; limitaciones de la arquitectura del modelo; problemas de normalización del texto. **Entrena durante más tiempo, mejora la variedad y calidad de los datos, prueba otra arquitectura y asegúrate de que el texto esté bien puntuado y normalizado.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [3_MODEL_TRAINING.md](./3-model-training.md), [4_INFERENCE.md](./4-inference.md) |
| **Calidad de Inferencia** | La salida es ruidosa, distorsionada o ininteligible | Mala calidad de los datos; el modelo no convergió; desajuste entre la configuración de entrenamiento y la de inferencia; `sampling rate` incorrecto en la inferencia. **Limpia rigurosamente los datos, entrena durante más tiempo, asegura una coincidencia exacta entre configuración y checkpoint y verifica los parámetros de audio.** | Todas las Guías |
| **Calidad de Inferencia** | La salida suena como el locutor incorrecto en fine-tuning | Modelo preentrenado no cargado correctamente; tasa de aprendizaje demasiado alta al inicio; datos o pasos de fine-tuning insuficientes; desajuste del `speaker_id`. **Verifica `pretrained_model_path` e `ignore_layers`, usa una tasa de aprendizaje más baja y comprueba el `speaker_id`.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md), [4_INFERENCE.md](./4-inference.md) |
| **Calidad de Inferencia** | La inferencia se corta antes de tiempo o habla demasiado rápido o lento | Limitación del modelo; ajuste de inferencia que limita la longitud máxima de salida; parámetro de velocidad o `length scale` incorrecto. **Consulta la documentación del framework para longitud máxima y pasos del decodificador y ajusta los parámetros de velocidad.** | [4_INFERENCE.md](./4-inference.md) |
| **Uso del Modelo** | No se puede cargar el archivo de checkpoint | Descarga o archivo corrupto; checkpoint con versión incompatible del framework o configuración; ruta de archivo incorrecta. **Vuelve a descargar el archivo, verifica su integridad, usa la configuración correcta y comprueba la ruta.** | [5_PACKAGING_AND_SHARING.md](./5-packaging-and-sharing.md), [4_INFERENCE.md](./4-inference.md) |

---

## Si Aún Necesitas Ayuda

Cuando escribas en un issue tracker, Discord o foro, incluye suficientes detalles para que otra persona pueda reproducir el problema:

- el nombre del framework, la rama o release, y las versiones de Python y PyTorch
- tu modelo de GPU, la cantidad de VRAM y si estás trabajando con CUDA o CPU
- el comando exacto que ejecutaste y el mensaje de error exacto
- si el problema ocurre durante la preparación de datos, el arranque del entrenamiento, la carga del checkpoint o la inferencia
- un ejemplo pequeño de la configuración afectada, una línea del manifest o el texto de entrada, si aplica

Los buenos reportes de errores reciben respuestas útiles mucho más rápido que un vago "no funciona".

Si es posible, reduce el problema a un comando corto, una entrada pequeña y un mensaje de error exacto. La gente casi siempre puede ayudar más rápido cuando no tiene que reconstruir primero todo tu proyecto.

## Recursos y Herramientas Útiles

Esta lista incluye software, librerías y comunidades útiles para proyectos de TTS.

Toma esta sección como un mapa inicial, no como una lista fija de recomendaciones. Los repositorios TTS, los forks mantenidos, las herramientas cloud y los modelos de precios cambian con regularidad, así que verifica la actividad y la documentación actuales antes de comprometerte con un flujo de trabajo.

### Procesamiento y Análisis de Audio:

*   **[Audacity](https://www.audacityteam.org/):** Editor de audio gratuito, de código abierto y multiplataforma. Excelente para la inspección manual, limpieza, etiquetado y procesamiento básico de archivos de audio.
*   **[FFmpeg](https://ffmpeg.org/):** La herramienta de línea de comandos multiusos para conversión de audio y vídeo, remuestreo, manipulación de canales, cambios de formato y mucho más. Esencial para automatizar operaciones por lotes.
*   **[SoX (Sound eXchange Compiled)](http://sox.sourceforge.net/) o [Sox - Source Code](https://codeberg.org/sox_ng/sox_ng/):** Utilidad de línea de comandos para el procesamiento de audio. Útil para efectos, conversión de formato y obtención de información del audio con `soxi`.
*   **[pydub](https://github.com/jiaaro/pydub):** Librería de Python para una manipulación sencilla de audio. Usa el backend de ffmpeg/libav.
*   **[librosa](https://librosa.org/doc/latest/index.html):** Librería de Python para análisis avanzado de audio, extracción de características y visualización.
*   **[soundfile](https://python-soundfile.readthedocs.io/en/latest/):** Librería de Python para leer y escribir archivos de audio.
*   **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm):** Librería de Python para normalización de sonoridad (LUFS).

### Transcripción (ASR):

*   **[OpenAI Whisper](https://github.com/openai/whisper):** Modelo ASR de código abierto y alta calidad, compatible con muchos idiomas. Buena base de referencia, aunque la puntuación suele necesitar revisión.
*   **[Herramientas y APIs de Google para transcripción de audio](https://ai.google.dev/):** Google puede ofrecer servicios o modelos útiles para transcripción. Los nombres de producto, límites y niveles gratuitos cambian con el tiempo, así que revisa la documentación actual antes de elegir un flujo de trabajo concreto.
*   **Servicios de ASR en la Nube:**
    *   [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
    *   [AWS Transcribe](https://aws.amazon.com/transcribe/)
    *   [Azure Speech Service](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/)
    *   *A menudo fiables, de pago por uso, y pueden tener cuotas gratuitas iniciales.*
*   **[Hugging Face Transformers - Modelos ASR](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition):** Hub para muchos modelos ASR preentrenados, incluyendo versiones con fine-tuning de Whisper y otros.
*   **[ElevenLabs Speech To Text (Scribe)](https://elevenlabs.io/speech-to-text):** *Servicio Comercial.* Conocido por una precisión muy alta, pero es un servicio de pago y puede ser relativamente caro.

### Frameworks y Bases de Código de TTS (Ejemplos - Busca forks o sucesores activos):

*   **[StyleTTS2 (Repo de Investigación)](https://github.com/yl4579/StyleTTS2):** Trabajo influyente sobre el control de estilo. Busca forks mantenidos activamente con pipelines completos.
*   **[VITS (Repo de Investigación)](https://github.com/jaywalnut310/vits):** Arquitectura popular de extremo a extremo. Existen muchos forks e implementaciones.
*   **[Coqui TTS (Archivado)](https://github.com/coqui-ai/TTS):** Referencia histórica. Fue muy influyente, pero para flujos nuevos conviene que los principiantes prioricen proyectos activos o forks realmente mantenidos.
*   **[ESPnet](https://github.com/espnet/espnet):** Kit de herramientas de procesamiento de voz de extremo a extremo, con recetas TTS para varios modelos.
*   **Busca en GitHub:** Usa palabras clave como "TTS", "VITS training", "StyleTTS2 training" o "PyTorch TTS" para encontrar proyectos actuales.

### Entorno de Python y Deep Learning:

*   **[Python](https://www.python.org/):** El lenguaje de programación central.
*   **[PyTorch](https://pytorch.org/):** La principal librería de deep learning usada por la mayoría de los frameworks modernos de TTS.
*   **[TensorBoard](https://www.tensorflow.org/tensorboard):** Esencial para visualizar el progreso del entrenamiento.
*   **[pip](https://pip.pypa.io/en/stable/) / [uv](https://github.com/astral-sh/uv):** Instaladores de paquetes de Python. `uv` es una alternativa más nueva y a menudo mucho más rápida.
*   **[conda](https://docs.conda.io/en/latest/) / [venv](https://docs.python.org/3/library/venv.html):** Herramientas para crear entornos de Python aislados.
*   **[Git](https://git-scm.com/):** Sistema de control de versiones, esencial para clonar repositorios y gestionar código.
*   **[Hugging Face Hub](https://huggingface.co/):** Plataforma para compartir modelos, datasets y código.

### Comunidades:

*   **GitHub Discussions/Issues de los Frameworks de TTS:** Consulta el repositorio específico que estés usando.
*   **Servidores de Discord:** Muchas comunidades de IA y ML tienen canales dedicados a TTS.
*   **Reddit:** Subreddits como `r/SpeechSynthesis` y `r/MachineLearning`.

---

Esto concluye la serie principal de guías. Recuerda que construir buenos modelos TTS a menudo implica iteración: revisar la preparación de datos o ajustar los parámetros de entrenamiento según los resultados es una práctica común.

---
