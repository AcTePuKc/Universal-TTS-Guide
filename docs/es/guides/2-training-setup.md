# Guía de Configuración de Entrenamiento TTS


Con tu dataset preparado, la siguiente etapa consiste en configurar el entorno de software y la primera configuración utilizable para el entrenamiento o fine-tuning del modelo TTS.

Si algún término de hardware o entrenamiento no está claro, consulta el [glosario](../glossary.md#glossary-of-technical-terms). Esta página solo explica los términos que afectan directamente a las decisiones de configuración.

---

## Configuración del entorno de entrenamiento

Esta sección cubre la instalación del software necesario y la organización de los archivos del proyecto.

### Elegir y clonar un framework de TTS

Si eres principiante, elige un framework que se mantenga activamente, tenga instrucciones claras de instalación y ya soporte el tipo de entrenamiento que necesitas. No optimices primero por "la arquitectura más avanzada". Optimiza por "puedo instalarlo, ejecutarlo y depurarlo".

-   **Selecciona un framework:** Elige una base de código adecuada para tus objetivos. Ten en cuenta:
    *   **Arquitectura:** VITS, StyleTTS2, Tacotron2 + vocoder, etc.
    *   **Soporte para fine-tuning:** Si permite ajustar modelos preentrenados de forma clara.
    *   **Soporte de idioma:** Si el tokenizador y la normalización funcionan bien con tu idioma.
    *   **Comunidad y mantenimiento:** Si el repositorio sigue activo y tiene soporte práctico.
    *   **Modelos preentrenados:** Si tienes un buen punto de partida para no entrenar desde cero.

### Comparación de arquitecturas TTS

| Arquitectura | Ventajas | Desventajas | Mejor para | Requisitos de hardware |
|:-------------|:---------|:------------|:-----------|:-----------------------|
| **VITS** | End-to-end, audio de alta calidad, inferencia rápida, buena para fine-tuning | Más compleja, puede ser inestable, requiere ajuste fino | Clonado de voz de un solo locutor y proyectos orientados a calidad | Entrenamiento: 8GB+ VRAM, inferencia: 4GB+ VRAM |
| **StyleTTS2** | Muy buen control de estilo y voz, calidad muy alta | Más nueva, más compleja, menos recursos comunitarios | Voz expresiva y control de estilo | Entrenamiento: 12GB+ VRAM, inferencia: 6GB+ VRAM |
| **Tacotron2 + HiFi-GAN** | Más estable, más fácil de entender, más tutoriales | Pipeline de dos etapas, calidad menor que modelos más nuevos | Proyectos educativos o más predecibles | Entrenamiento: 6GB+ VRAM, inferencia: 2GB+ VRAM |
| **FastSpeech2** | Inferencia rápida, más estable que Tacotron2, buena documentación | Requiere alineaciones fonémicas y preprocesamiento más complejo | Aplicaciones rápidas y salida más controlada | Entrenamiento: 8GB+ VRAM, inferencia: 2GB+ VRAM |
| **YourTTS / XTTS** | Soporte multilingüe, zero-shot, flexibilidad entre idiomas | Configuración compleja, exige más cuidado en los datos | Proyectos multilingües y cross-lingual | Entrenamiento: 10GB+ VRAM, inferencia: 4GB+ VRAM |
| **TTS basado en difusión** | Gran potencial de calidad, prosodia más natural | Inferencia lenta y entrenamiento muy costoso | Generación offline e investigación | Entrenamiento: 16GB+ VRAM, inferencia: 8GB+ VRAM |

**Nota sobre el hardware:**
- Son mínimos aproximados.
- Batch sizes mayores y configuraciones más grandes requerirán más VRAM.
- La inferencia por CPU es posible, pero será bastante más lenta.

**Atajo práctico:** si estás eligiendo tu primer framework para un proyecto real, en lugar de comparar arquitecturas, elige el que tenga la documentación de instalación más clara, el gestor de incidencias más activo y un ejemplo de fine-tuning cercano a tu caso de uso.

### Requisitos de hardware según la escala del proyecto

#### Requisitos de GPU por modelo y tamaño del dataset

| Tipo de modelo | Dataset pequeño (<10h) | Dataset medio (10-50h) | Dataset grande (>50h) | GPUs recomendadas |
|:---------------|:-----------------------|:------------------------|:----------------------|:------------------|
| **Tacotron2 + HiFi-GAN** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **FastSpeech2** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **VITS** | 12GB VRAM | 16GB VRAM | 24GB+ VRAM | RTX 3080, RTX 3090, A5000 |
| **StyleTTS2** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |
| **XTTS-v2** | 24GB VRAM | 32GB VRAM | 40GB+ VRAM | RTX 4090, A100, A6000 |
| **TTS basado en difusión** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |

#### CPU, RAM y almacenamiento

| Escala | CPU | RAM | Almacenamiento |
|:-------|:----|:----|:---------------|
| **Personal** | 4+ núcleos, 2.5GHz+ | 16GB | 50GB SSD |
| **Investigación** | 8+ núcleos, 3.0GHz+ | 32GB | 100GB+ SSD |
| **Producción** | 16+ núcleos, 3.5GHz+ | 64GB+ | 500GB+ NVMe SSD |

#### Opciones orientativas de GPU en la nube*

**\*Nota de actualidad:** los proveedores y ejemplos de GPU siguientes reflejan el panorama de la nube en el momento en que se escribió esta guía. Las ofertas, la disponibilidad y los precios cambian con frecuencia según la región, los descuentos y el spot pricing. Usa la tabla solo como orientación y verifica las opciones y precios actuales en el sitio del proveedor antes de presupuestar el entrenamiento.

| Proveedor | Opción GPU | VRAM | Coste relativo | Mejor para |
|:----------|:-----------|:-----|:---------------|:-----------|
| **Google Colab** | T4/P100 (los niveles gratuitos pueden variar)<br>V100/A100 (los niveles de pago pueden variar) | 16GB<br>16-40GB | Bajo a medio | Pruebas y datasets pequeños |
| **Kaggle** | P100/T4 | 16GB | Bajo | Datasets pequeños y medianos |
| **AWS** | g4dn.xlarge (T4)<br>p3.2xlarge (V100)<br>p4d.24xlarge (A100) | 16GB<br>16GB<br>40GB | Medio a muy alto | Cualquier escala |
| **GCP** | Instancias T4<br>Instancias A100 | 16GB<br>40GB | Medio a muy alto | Cualquier escala |
| **Azure** | Instancias clase V100 o A100 | 16GB+ | Medio a muy alto | Cualquier escala |
| **Lambda Labs** | 1x RTX 3090<br>1x A100 | 24GB<br>40GB | Medio | Investigación y datasets medianos |
| **Vast.ai** | Varias GPU de consumo | 8-24GB | Bajo a medio | Entrenamiento con presupuesto ajustado |

#### Rangos muy aproximados de tiempo de entrenamiento

**Nota sobre tiempos:** estos rangos cambian mucho según la implementación, el batch size, la limpieza del dataset, el tokenizador, el checkpoint de origen y si haces fine-tuning o entrenamiento desde cero. Úsalos como orden de magnitud, no como promesa.

| Modelo | Tamaño del dataset | GPU | Tiempo aproximado | Pasos hasta convergencia |
|:-------|:-------------------|:----|:------------------|:-------------------------|
| **Tacotron2 + HiFi-GAN** | 10 horas | RTX 3080 | 2-3 días | 50-100K pasos |
| **FastSpeech2** | 10 horas | RTX 3080 | 2-3 días | 150-200K pasos |
| **VITS** | 10 horas | RTX 3090 | 3-5 días | 300-500K pasos |
| **StyleTTS2** | 10 horas | RTX 3090 | 4-7 días | 500-800K pasos |
| **XTTS-v2** | 10 horas | RTX 4090 | 5-10 días | 1M+ pasos |

#### Consejos para reducir los requisitos de hardware

1. **Gradient accumulation:** simula batch sizes más grandes acumulando gradientes durante varios pasos forward/backward.
2. **Mixed precision training:** usa FP16 en lugar de FP32 para reducir el uso de VRAM hasta un 50%.
3. **Gradient checkpointing:** intercambia memoria por cómputo recalculando activaciones durante el paso backward.
4. **Model parallelism:** divide modelos grandes entre varias GPU.
5. **Entrenamiento progresivo:** empieza con modelos o configuraciones más pequeños y aumenta la complejidad gradualmente.

Estos requisitos deberían ayudarte a planificar las necesidades de hardware según los objetivos y el presupuesto de tu proyecto.

-   **Clona el repositorio:** Una vez elegido, clona el framework con Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY>
    ```
    *   Ejemplo: `git clone https://github.com/some-user/some-tts-framework.git`

### Configurar el entorno de Python e instalar dependencias

-   **Entorno virtual:** Es recomendable usar un entorno virtual dedicado.
    *   **Con `venv`:**
        ```bash
        python -m venv venv_tts
        # Windows: .\venv_tts\Scripts\activate
        # Linux/macOS: source venv_tts/bin/activate
        ```
    *   **Con `conda`:**
        ```bash
        conda create --name tts_env python=3.9
        conda activate tts_env
        ```
-   **Instalar PyTorch con CUDA:** Usa el [instalador oficial de PyTorch](https://pytorch.org/get-started/locally/) para hacer coincidir versión de CUDA, drivers y gestor de paquetes.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    ```
-   **Instalar dependencias del framework:**
    ```bash
    pip install -r requirements.txt

    # Si usas uv:
    uv pip install -r requirements.txt
    ```
    *   **Si hay errores:** Revisa bibliotecas del sistema faltantes, incompatibilidades de versión o problemas entre CUDA y PyTorch.

#### Prueba mínima de sanidad del entorno

Antes de editar una gran configuración o iniciar un entrenamiento largo, confirma que estos comandos funcionan:

```bash
python --version
ffmpeg -version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__)"
```

Si alguno falla, detente y corrige el entorno primero. Es mucho más barato que depurar una ejecución rota después de horas de espera.

### Organizar la carpeta del proyecto

-   Una estructura clara evita confusión con manifests, checkpoints y configuraciones.

    ```mermaid
    flowchart TD
        root["Project Root"] --> repo["TTS Repo Directory"]
        repo --> scripts["Scripts principales"]
        scripts --> train["train.py"]
        scripts --> inference["inference.py"]
        repo --> config["configs/base_config.yaml"]
        repo --> requirements["requirements.txt"]
        repo --> repoMore["otros archivos del framework"]
    ```

    ```mermaid
    flowchart TD
        root["Project Root"] --> dataset["my_tts_dataset"]
        dataset --> audio["normalized_chunks"]
        audio --> wav1["segment_00001.wav"]
        audio --> wavMore["más archivos .wav"]
        dataset --> transcripts["transcripts (opcional)"]
        dataset --> trainList["train_list.txt"]
        dataset --> valList["val_list.txt"]
    ```

    ```mermaid
    flowchart TD
        root["Project Root"] --> checkpoints["checkpoints"]
        checkpoints --> runDir["my_custom_model"]
        root --> configs["my_configs"]
        configs --> trainingConfig["my_training_run_config.yaml"]
    ```

-   **Rutas:** Asegúrate de que las rutas de tu configuración sean correctas en relación con el lugar desde el que realmente ejecutarás `train.py`, normalmente dentro de `<TTS_REPO_DIRECTORY>`.

---

## Configurar la ejecución de entrenamiento

Antes de lanzar el entrenamiento, necesitas un archivo de configuración que le indique al framework cómo entrenar el modelo con tus datos.

### 1. Encontrar y copiar una configuración base

-   **Busca ejemplos:** Revisa la carpeta `configs/` del framework elegido.
-   **Elige según tu objetivo:**
    *   **Fine-tuning:** Busca nombres como `config_ft.yaml` o `finetune_*.yaml`; normalmente esperan un modelo preentrenado.
    *   **Entrenamiento desde cero:** Busca `config_base.yaml` o `train_*.yaml`.
    *   **Tamaño del dataset:** Algunas implementaciones ofrecen variantes para datasets pequeños (`_sm`) o grandes (`_lg`).
-   **Copia y renombra:** Copia la plantilla a tu propia carpeta y dale un nombre descriptivo para esa ejecución, por ejemplo `my_yoruba_voice_ft_config.yaml`.
    ```bash
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```
    En Windows PowerShell usa `Copy-Item` si no estás trabajando en Git Bash.

**Consejo para principiantes:** parte del ejemplo funcional más cercano que ya proporcione el framework. No construyas tu primera configuración desde cero.

### 2. Editar tu archivo de configuración personalizado

-   Abre la copia de la configuración en un editor de texto.
-   **Modifica los parámetros clave:** Los nombres concretos variarán entre frameworks, pero las categorías suelen ser parecidas.

    ```yaml
    # --- Dataset y carga de datos ---
    # Rutas relativas al lugar desde el que ejecutas train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt"
    val_filelist_path: "../my_tts_dataset/val_list.txt"
    # Algunos frameworks pueden necesitar data_path o audio_root hacia el directorio de audio.

    # --- Salida y registros ---
    output_directory: "../checkpoints/my_yoruba_voice_run1"
    log_interval: 100
    validation_interval: 1000
    save_checkpoint_interval: 5000

    # --- Hiperparámetros principales ---
    epochs: 1000
    batch_size: 16                     # Reduce si aparece un error CUDA OOM.
    learning_rate: 1e-4                # Puede requerir ajuste; suele ser menor en fine-tuning.
    # lr_scheduler: "cosine_decay"
    # weight_decay: 0.01

    # --- Parámetros de audio ---
    sampling_rate: 22050               # DEBE coincidir con la frecuencia de muestreo del dataset preparado (de la Guía 1).
    # filter_length: 1024
    # hop_length: 256
    # win_length: 1024
    # n_mel_channels: 80
    # mel_fmin: 0.0
    # mel_fmax: 8000.0

    # --- Arquitectura del modelo ---
    # model_type: "VITS"
    # hidden_channels: 192
    # num_speakers: 1

    # --- Detalles de fine-tuning (si aplica) ---
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth"
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```

-   **Lee la documentación del framework:** Ahí está la referencia exacta de cada parámetro.
-   **Nota sobre los términos:** en esta configuración, un [checkpoint](../glossary.md#glossary-checkpoint) es una instantánea guardada del modelo, y el [sampling rate](../glossary.md#glossary-sampling-rate) debe coincidir exactamente con el dataset preparado.
-   **Trampa común al empezar:** en tu primera ejecución cambia solo lo obligatorio: rutas del dataset, directorio de salida, sampling rate, batch size y checkpoint de fine-tuning si aplica. No toques diez ajustes a la vez antes de confirmar que el pipeline funciona una vez.

### 3. Consideraciones de hardware y dataset

-   **VRAM de GPU:** [VRAM](../glossary.md#glossary-vram) es la memoria de la tarjeta gráfica. `batch_size` es el control principal para la memoria. Empieza con un valor recomendado y redúcelo si aparece un error «CUDA out of memory».
-   **Tamaño del dataset frente a epochs:**
    *   **Datasets pequeños (<20h):** pueden requerir menos épocas (por ejemplo, 300-1500), pero necesitan seguimiento cuidadoso de la [validation loss](../glossary.md#glossary-validation-loss) y de las muestras para evitar el [overfitting](../glossary.md#glossary-overfitting). Considera learning rates más bajos.
    *   **Datasets grandes (>50h):** pueden beneficiarse de más épocas (1000+) para aprender completamente los patrones de los datos.
-   **CPU:** Aunque la GPU hace la mayor parte del trabajo, se necesita una CPU multinúcleo decente para cargar y preprocesar los datos.
-   **Almacenamiento:** Reserva espacio para el dataset, el entorno de Python, el código del framework y especialmente los checkpoints, que pueden ocupar cientos de MB o varios GB.

### 4. Herramientas de monitorización (TensorBoard)

-   La mayoría de los frameworks modernos de TTS se integran con [TensorBoard](https://www.tensorflow.org/tensorboard).
-   En la configuración suele haber opciones como `use_tensorboard: True` o `log_directory`.
-   Durante el entrenamiento normalmente puedes ejecutar `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (por ejemplo, `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) en otra terminal para supervisar las curvas de pérdida, el learning rate y las muestras de validación sintetizadas.
-   Si TensorBoard aparece vacío, primero comprueba que el framework realmente esté escribiendo archivos de eventos en la ruta esperada. Un panel vacío muchas veces es solo un log path incorrecto.

---

Con el entorno preparado y la configuración adaptada a tus datos, ya estás listo para pasar al entrenamiento real del modelo.

## Antes de continuar

- [ ] Tu entorno Python está activado y las dependencias del framework se instalan sin errores.
- [ ] `torch.cuda.is_available()` devuelve `True` si vas a entrenar con GPU.
- [ ] `ffmpeg` y cualquier biblioteca del sistema necesaria están instalados y visibles en PATH.
- [ ] Las rutas de tu configuración apuntan a manifests, checkpoints y carpetas de salida reales.
- [ ] El `sampling_rate` de la configuración coincide exactamente con el dataset preparado.
