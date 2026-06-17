# Guía 2: Configuración y Ajuste del Entorno de Entrenamiento

**Navegación:** [README Principal]({{ site.baseurl }}/languages/es/){: .btn .btn-primary} | [Paso Anterior: Preparación de Datos](./1_DATA_PREPARATION.md){: .btn .btn-primary} | [Siguiente Paso: Entrenamiento del Modelo](./3_MODEL_TRAINING.md){: .btn .btn-primary}

Con tu dataset preparado, la siguiente etapa consiste en configurar el entorno de software necesario y ajustar los parámetros para tu ejecución de entrenamiento específica.

---

## 3. Configuración del Entorno de Entrenamiento

Esta sección cubre la instalación del software requerido y la organización de los archivos de tu proyecto.

### 3.1. Elegir y Clonar un Framework de TTS

-   **Selecciona un Framework:** Elige una base de código de TTS adecuada para tus objetivos. Considera factores como:
    *   **Arquitectura:** VITS, StyleTTS2, Tacotron2+Vocoder, etc. Las arquitecturas más nuevas a menudo producen mejor calidad.
    *   **Soporte para Fine-tuning:** ¿El framework soporta explícitamente el fine-tuning a partir de modelos preentrenados? Esto suele ser más fácil que entrenar desde cero.
    *   **Soporte de Idioma:** Comprueba si el modelo/tokenizador gestiona bien tu idioma objetivo.
    *   **Comunidad y Mantenimiento:** ¿Se mantiene el repositorio de forma activa? ¿Hay debates de la comunidad o canales de soporte?
    *   **Modelos Preentrenados:** ¿El framework proporciona modelos preentrenados adecuados como punto de partida para el fine-tuning?

#### Comparación de Arquitecturas de TTS

Al seleccionar una arquitectura de TTS, considera estas opciones populares y sus características:

| Arquitectura | Ventajas | Desventajas | Mejor Para | Requisitos de Hardware |
|:-------------|:-----|:-----|:---------|:---------------------|
| **VITS** | • De extremo a extremo (sin vocoder separado)<br>• Audio de alta calidad<br>• Inferencia rápida<br>• Buena para fine-tuning | • Compleja de entender<br>• Puede ser inestable durante el entrenamiento<br>• Requiere un ajuste cuidadoso de hiperparámetros | • Clonación de voz de un solo locutor<br>• Proyectos que necesitan una salida de alta calidad<br>• Cuando tienes más de 5 horas de datos | • Entrenamiento: 8GB+ VRAM<br>• Inferencia: 4GB+ VRAM |
| **StyleTTS2** | • Excelente control de voz y estilo<br>• Calidad de última generación<br>• Buena para emoción/prosodia | • Más nueva, implementaciones potencialmente menos estables<br>• Arquitectura más compleja<br>• Menos recursos de la comunidad | • Proyectos que requieren control de estilo<br>• Síntesis de habla expresiva<br>• Multi-locutor con transferencia de estilo | • Entrenamiento: 12GB+ VRAM<br>• Inferencia: 6GB+ VRAM |
| **Tacotron2 + HiFi-GAN** | • Bien establecida, estable<br>• Más fácil de entender<br>• Más tutoriales disponibles<br>• Componentes separados para una depuración más fácil | • Pipeline de dos etapas (más lento)<br>• Generalmente menor calidad que los modelos más nuevos<br>• Más propensa a fallos de atención en textos largos | • Proyectos educativos<br>• Cuando se prioriza la estabilidad sobre la calidad<br>• Entornos de bajos recursos | • Entrenamiento: 6GB+ VRAM<br>• Inferencia: 2GB+ VRAM |
| **FastSpeech2** | • No autorregresiva (inferencia más rápida)<br>• Más estable que Tacotron2<br>• Buena documentación | • Requiere alineaciones de fonemas<br>• Preprocesamiento más complejo<br>• Calidad no tan alta como VITS/StyleTTS2 | • Aplicaciones en tiempo real<br>• Cuando la velocidad de inferencia es crítica<br>• Salida más controlada | • Entrenamiento: 8GB+ VRAM<br>• Inferencia: 2GB+ VRAM |
| **YourTTS (variante de VITS)** | • Soporte multilingüe<br>• Clonación de voz zero-shot<br>• Buena para transferencia de idioma | • Configuración de entrenamiento compleja<br>• Requiere una preparación cuidadosa de los datos<br>• Puede necesitar datasets más grandes | • Proyectos multilingües<br>• Clonación de voz interlingüe<br>• Cuando se necesita flexibilidad de idioma | • Entrenamiento: 10GB+ VRAM<br>• Inferencia: 4GB+ VRAM |
| **TTS basado en Difusión** | • El mayor potencial de calidad<br>• Prosodia más natural<br>• Mejor manejo de palabras raras | • Inferencia muy lenta<br>• Entrenamiento extremadamente intensivo en cómputo<br>• Más nueva, menos establecida | • Generación offline<br>• Cuando la calidad supera a la velocidad<br>• Proyectos de investigación | • Entrenamiento: 16GB+ VRAM<br>• Inferencia: 8GB+ VRAM |

**Nota sobre los Requisitos de Hardware:**
- Estos son mínimos aproximados; tamaños de batch más grandes o configuraciones de modelo más grandes requerirán más VRAM
- Los tiempos de entrenamiento varían significativamente: VITS/StyleTTS2 normalmente necesitan más epochs que Tacotron2
- La inferencia por CPU es posible para todos los modelos, pero será significativamente más lenta

### 1.3. Requisitos de Hardware Detallados

Elegir el hardware adecuado es crítico para un entrenamiento exitoso de modelos TTS. Aquí tienes un desglose detallado de los requisitos para diferentes escenarios:

#### Requisitos de GPU por Tipo de Modelo y Tamaño del Dataset

| Tipo de Modelo | Dataset Pequeño (<10h) | Dataset Mediano (10-50h) | Dataset Grande (>50h) | Modelos de GPU Recomendados |
|:-----------|:---------------------|:------------------------|:---------------------|:-----------------------|
| **Tacotron2 + HiFi-GAN** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **FastSpeech2** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **VITS** | 12GB VRAM | 16GB VRAM | 24GB+ VRAM | RTX 3080, RTX 3090, A5000 |
| **StyleTTS2** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |
| **XTTS-v2** | 24GB VRAM | 32GB VRAM | 40GB+ VRAM | RTX 4090, A100, A6000 |
| **TTS basado en Difusión** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |

#### CPU y Memoria del Sistema

| Escala de Entrenamiento | Requisitos de CPU | RAM del Sistema | Almacenamiento |
|:---------------|:-----------------|:-----------|:--------|
| **Aficionado/Personal** | 4+ núcleos, 2.5GHz+ | 16GB | 50GB SSD |
| **Investigación** | 8+ núcleos, 3.0GHz+ | 32GB | 100GB+ SSD |
| **Producción** | 16+ núcleos, 3.5GHz+ | 64GB+ | 500GB+ NVMe SSD |

#### Opciones de GPU en la Nube y Costes Aproximados

| Proveedor de Nube | Opción de GPU | VRAM | Coste Aprox./Hora | Mejor Para |
|:---------------|:-----------|:-----|:------------------|:---------|
| **Google Colab** | T4/P100 (Gratis)<br>V100/A100 (Pro) | 16GB<br>16-40GB | Gratis<br>$10-$15 | Experimentación, datasets pequeños |
| **Kaggle** | P100/T4 | 16GB | Gratis (horas limitadas) | Datasets pequeños-medianos |
| **AWS** | g4dn.xlarge (T4)<br>p3.2xlarge (V100)<br>p4d.24xlarge (A100) | 16GB<br>16GB<br>40GB | $0.50-$0.75<br>$3.00-$3.50<br>$20.00-$32.00 | Cualquier escala, producción |
| **GCP** | n1-standard-8 + T4<br>a2-highgpu-1g (A100) | 16GB<br>40GB | $0.35-$0.50<br>$3.80-$4.50 | Cualquier escala, producción |
| **Azure** | NC6s_v3 (V100)<br>NC24ads_A100_v4 | 16GB<br>80GB | $3.00-$3.50<br>$16.00-$24.00 | Cualquier escala, producción |
| **Lambda Labs** | 1x RTX 3090<br>1x A100 | 24GB<br>40GB | $1.10<br>$1.99 | Investigación, datasets medianos |
| **Vast.ai** | Varias GPUs de consumo | 8-24GB | $0.20-$1.00 | Entrenamiento con presupuesto ajustado |

#### Estimaciones de Tiempo de Entrenamiento

| Modelo | Tamaño del Dataset | GPU | Tiempo de Entrenamiento Aproximado | Epochs hasta Convergencia |
|:------|:-------------|:----|:--------------------------|:----------------------|
| **Tacotron2 + HiFi-GAN** | 10 horas | RTX 3080 | 2-3 días | 50-100K pasos |
| **FastSpeech2** | 10 horas | RTX 3080 | 2-3 días | 150-200K pasos |
| **VITS** | 10 horas | RTX 3090 | 3-5 días | 300-500K pasos |
| **StyleTTS2** | 10 horas | RTX 3090 | 4-7 días | 500-800K pasos |
| **XTTS-v2** | 10 horas | RTX 4090 | 5-10 días | 1M+ pasos |

#### Consejos de Optimización para Reducir los Requisitos de Hardware

1. **Gradient Accumulation (acumulación de gradientes)**: Simula tamaños de batch más grandes acumulando gradientes a lo largo de múltiples pasos hacia adelante/atrás (forward/backward)
2. **Mixed Precision Training (entrenamiento de precisión mixta)**: Usa FP16 en lugar de FP32 para reducir el uso de VRAM hasta en un 50%
3. **Gradient Checkpointing**: Intercambia cómputo por memoria recalculando las activaciones durante el paso hacia atrás (backward pass)
4. **Model Parallelism (paralelismo de modelo)**: Divide modelos grandes entre múltiples GPUs
5. **Progressive Training (entrenamiento progresivo)**: Empieza con modelos/configuraciones más pequeños y aumenta gradualmente la complejidad

Estos requisitos deberían ayudarte a planificar tus necesidades de hardware según los objetivos específicos de tu proyecto y tus restricciones de presupuesto.
-   **Clona el Repositorio:** Una vez elegido, clona el repositorio de código del framework usando Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY> # Navega dentro del directorio clonado
    ```
    *   Ejemplo: `git clone https://github.com/some-user/some-tts-framework.git`

### 3.2. Configurar el Entorno de Python e Instalar Dependencias

-   **Entorno Virtual (Recomendado):** Crea y activa un entorno virtual de Python dedicado para aislar las dependencias y evitar conflictos con otros proyectos o con los paquetes de Python del sistema.
    *   **Usando `venv` (integrado):**
        ```bash
        python -m venv venv_tts  # Crea un entorno llamado 'venv_tts'
        # Actívalo:
        # Windows: .\venv_tts\Scripts\activate
        # Linux/macOS: source venv_tts/bin/activate
        ```
    *   **Usando `conda`:**
        ```bash
        conda create --name tts_env python=3.9 # O la versión de Python deseada
        conda activate tts_env
        ```
-   **Instalar PyTorch con CUDA:** Esto es crítico para la aceleración por GPU. Visita la [Guía Oficial de Instalación de PyTorch](https://pytorch.org/get-started/locally/) y selecciona las opciones que coincidan con tu sistema operativo, gestor de paquetes (`pip` o `conda`), plataforma de cómputo (versión de CUDA) y la versión de PyTorch deseada. **Asegúrate de que los drivers de NVIDIA instalados sean compatibles con la versión de CUDA elegida.**
    ```bash
    # Comando de ejemplo usando pip para CUDA 11.8 (¡consulta el sitio web de PyTorch para los comandos actuales!)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Verifica la instalación:
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    # Debería mostrar la versión de PyTorch, True, y tu versión de CUDA si tiene éxito.
    ```
-   **Instalar los Requisitos del Framework:** La mayoría de los frameworks listan sus dependencias en un archivo `requirements.txt`. Instálalas usando `pip` (o `uv`, que suele ser más rápido).
    ```bash
    # Navega primero al directorio del framework si no estás ya allí
    # Usando pip:
    pip install -r requirements.txt

    # Usando uv (si está instalado: pip install uv):
    uv pip install -r requirements.txt
    ```
    *   **Resolución de Problemas:** Presta atención a cualquier error de instalación. Podría indicar la falta de librerías del sistema (como `libsndfile`), versiones de paquetes incompatibles, o problemas con tu configuración de CUDA/PyTorch. Consulta la documentación del framework para conocer los prerrequisitos específicos.

### 3.3. Organizar la Carpeta de tu Proyecto

-   Una estructura de carpetas bien organizada facilita la gestión de tu proyecto. Coloca tu dataset preparado (o crea un enlace simbólico a él) dentro o junto al código del framework. Una estructura común se ve así:

    ```bash
    <YOUR_PROJECT_ROOT>/
    ├── <TTS_REPO_DIRECTORY>/         # El código del framework clonado
    │   ├── train.py                 # Script principal de entrenamiento (el nombre puede variar)
    │   ├── inference.py             # Script de inferencia (el nombre puede variar)
    │   ├── configs/                 # Directorio para los archivos de configuración
    │   │   └── base_config.yaml     # Configuración de ejemplo del framework
    │   ├── requirements.txt
    │   └── ... (otros archivos del framework)
    │
    ├── my_tts_dataset/              # Tu dataset preparado de la Guía 1
    │   ├── normalized_chunks/       # Archivos de audio finales
    │   │   ├── segment_00001.wav
    │   │   └── ...
    │   ├── transcripts/             # Opcional: archivos de texto si no están directamente en el manifest
    │   ├── train_list.txt           # Manifest de entrenamiento
    │   └── val_list.txt             # Manifest de validación
    │
    ├── checkpoints/                 # Crea este directorio: donde se guardarán los modelos
    │   └── my_custom_model/         # Subdirectorio para una ejecución de entrenamiento específica
    │
    └── my_configs/                  # Opcional: coloca aquí tus configuraciones personalizadas
        └── my_training_run_config.yaml
    ```
-   **Rutas:** Asegúrate de que las rutas especificadas más adelante en tu archivo de configuración (para datasets, salidas) sean correctas en relación con el lugar desde donde *ejecutarás* el script `train.py` (normalmente desde dentro del `<TTS_REPO_DIRECTORY>`).

---

## 4. Configurando la Ejecución del Entrenamiento

Antes de lanzar el entrenamiento, necesitas crear un archivo de configuración que le indique al framework *cómo* entrenar el modelo, usando *tus* datos específicos.

### 4.1. Encontrar y Copiar una Configuración Base

-   **Localiza Ejemplos:** Explora el directorio `configs/` dentro del framework de TTS. Busca archivos de configuración (`.yaml`, `.json`, o similares) que sirvan como plantillas.
-   **Elige Apropiadamente:** Selecciona un archivo de configuración que coincida con tu objetivo:
    *   **Fine-tuning:** Busca nombres como `config_ft.yaml`, `finetune_*.yaml`. Estos a menudo asumen que proporcionarás un modelo preentrenado.
    *   **Entrenamiento desde Cero:** Busca nombres como `config_base.yaml`, `train_*.yaml`.
    *   **Tamaño del Dataset:** Algunos frameworks podrían ofrecer configuraciones ajustadas para datasets pequeños (`_sm`) o grandes (`_lg`).
-   **Copia y Renombra:** Copia el archivo de plantilla elegido a una nueva ubicación (por ejemplo, tu propio directorio `my_configs/` o dentro del directorio `configs/` del framework) y dale un nombre descriptivo para tu ejecución específica (por ejemplo, `my_yoruba_voice_ft_config.yaml`).
    ```bash
    # Ejemplo: copiando una configuración de fine-tuning
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```

### 4.2. Editar tu Archivo de Configuración Personalizado

-   Abre tu archivo de configuración recién copiado (`my_yoruba_voice_ft_config.yaml`) en un editor de texto.
-   **Modifica los Parámetros Clave:** Revisa y modifica cuidadosamente los parámetros. Los nombres de los parámetros **variarán significativamente** entre frameworks, pero las categorías comunes incluyen:

    ```yaml
    # --- Dataset y Carga de Datos ---
    # Rutas relativas al lugar desde donde ejecutas train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt" # Ruta a tu manifest de entrenamiento
    val_filelist_path: "../my_tts_dataset/val_list.txt"   # Ruta a tu manifest de validación
    # Algunos frameworks podrían necesitar 'data_path' o 'audio_root' apuntando al directorio de audio en su lugar/adicionalmente.

    # --- Salida y Registro (Logging) ---
    output_directory: "../checkpoints/my_yoruba_voice_run1" # MUY IMPORTANTE: donde se guardan modelos, logs, muestras. Crea este directorio base si es necesario.
    log_interval: 100                  # Cada cuánto (en pasos/batches) imprimir logs
    validation_interval: 1000          # Cada cuánto (en pasos/batches) ejecutar la validación
    save_checkpoint_interval: 5000     # Cada cuánto (en pasos/batches) guardar los checkpoints del modelo

    # --- Hiperparámetros Centrales del Entrenamiento ---
    epochs: 1000                       # Número total de pasadas sobre los datos de entrenamiento. Ajusta según el tamaño del dataset y la convergencia.
    batch_size: 16                     # Número de muestras procesadas en paralelo por GPU. REDUCE si obtienes errores de CUDA OOM. AUMENTA para un entrenamiento más rápido si la VRAM lo permite.
    learning_rate: 1e-4                # Tasa de aprendizaje inicial. Puede necesitar ajuste (p. ej., más baja para fine-tuning: 5e-5 o 1e-5).
    # lr_scheduler: "cosine_decay"     # Programación de la tasa de aprendizaje (p. ej., step decay, exponential decay) - depende del framework
    # weight_decay: 0.01               # Parámetro de regularización

    # --- Parámetros de Audio ---
    sampling_rate: 22050               # CRÍTICO: DEBE coincidir con el sampling rate de tu dataset preparado (de la Guía 1).
    # Otros parámetros de audio (a menudo dependen de la arquitectura del modelo):
    # filter_length: 1024              # Tamaño de FFT para STFT
    # hop_length: 256                  # Tamaño de salto (hop) para STFT
    # win_length: 1024                 # Tamaño de la ventana para STFT
    # n_mel_channels: 80               # Número de bandas Mel
    # mel_fmin: 0.0                    # Frecuencia Mel mínima
    # mel_fmax: 8000.0                 # Frecuencia Mel máxima (a menudo sampling_rate / 2)

    # --- Arquitectura del Modelo ---
    # model_type: "VITS"               # Tipo de arquitectura del modelo
    # hidden_channels: 192             # Tamaño de las capas internas
    # num_speakers: 1                  # Establece >1 para datasets multi-locutor (debe coincidir con la preparación de datos)

    # --- Aspectos Específicos del Fine-tuning (Si Aplica) ---
    # Establece 'True' o proporciona la ruta al hacer fine-tuning
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth" # Ruta al checkpoint preentrenado del que partir.
    # Opcional: especifica las capas a ignorar/reinicializar si es necesario
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```
-   **Lee la Documentación del Framework:** Consulta la documentación específica de tu framework de TTS elegido para entender qué hace cada parámetro de su archivo de configuración.

### 4.3. Consideraciones de Hardware y Dataset

-   **VRAM de la GPU:** El `batch_size` es el principal control para gestionar el uso de memoria de la GPU. Empieza con un valor recomendado (por ejemplo, 16 o 32) y redúcelo si encuentras errores de "CUDA out of memory" durante el inicio del entrenamiento. Los tamaños de batch más grandes generalmente conducen a una convergencia más rápida, pero requieren más VRAM.
-   **Tamaño del Dataset vs. Epochs:**
    *   **Datasets Pequeños (< 20h):** Pueden requerir menos epochs (por ejemplo, 300-1500), pero necesitan una monitorización cuidadosa a través del validation loss/muestras para evitar el overfitting (cuando el modelo memoriza los datos de entrenamiento pero rinde mal con texto nuevo). Considera tasas de aprendizaje más bajas.
    *   **Datasets Grandes (> 50h):** Pueden beneficiarse de más epochs (1000+) para aprender por completo los patrones de los datos.
-   **CPU:** Aunque la GPU hace el trabajo pesado, se necesita una CPU multinúcleo decente para la carga y el preprocesamiento de datos, que de lo contrario puede convertirse en un cuello de botella.
-   **Almacenamiento:** Asegúrate de tener suficiente espacio en disco para el dataset, el entorno de Python, el código del framework y, especialmente, los checkpoints guardados, que pueden llegar a ser grandes (de cientos de MB a GB por checkpoint).

### 4.4. Herramientas de Monitorización (TensorBoard)

-   La mayoría de los frameworks modernos de TTS se integran con [TensorBoard](https://www.tensorflow.org/tensorboard) para visualizar el progreso del entrenamiento.
-   El archivo de configuración a menudo tiene ajustes relacionados con el registro (logging) (por ejemplo, `use_tensorboard: True`, `log_directory`).
-   Durante el entrenamiento, normalmente puedes lanzar TensorBoard ejecutando `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (por ejemplo, `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) en una terminal separada. Esto te permite monitorizar las curvas de loss, las tasas de aprendizaje y, potencialmente, escuchar muestras de validación sintetizadas en tu navegador web.

---

Con tu entorno configurado y el archivo de configuración adaptado a tus datos y objetivos, ahora estás listo para comenzar el proceso real de entrenamiento del modelo.

**Siguiente Paso:** [Entrenamiento del Modelo](./3_MODEL_TRAINING.md){: .btn .btn-primary} | 
[Volver Arriba](#top){: .btn .btn-primary}
