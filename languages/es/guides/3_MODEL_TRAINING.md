# Guía 3: Entrenamiento y Fine-tuning del Modelo

**Navegación:** [README Principal]({{ site.baseurl }}/languages/es/){: .btn .btn-primary} | [Paso Anterior: Configuración del Entrenamiento](./2_TRAINING_SETUP.md){: .btn .btn-primary} | [Siguiente Paso: Inferencia](./4_INFERENCE.md){: .btn .btn-primary}

Has preparado tus datos y configurado tu entorno de entrenamiento. Ahora es el momento de entrenar (o hacer fine-tuning de) tu modelo de Texto a Voz de verdad. Esta fase implica ejecutar el script de entrenamiento, monitorizar su progreso y entender cómo gestionar el proceso.

---

## 5. Ejecutando el Entrenamiento

Esta sección detalla cómo lanzar, monitorizar y gestionar el proceso de entrenamiento.

### 5.1. Lanzando el Script de Entrenamiento

-   **Navega al Directorio Correcto:** Abre tu terminal o línea de comandos y navega al directorio principal del repositorio del framework de TTS clonado (el directorio que contiene el script `train.py` o su equivalente).
-   **Activa el Entorno Virtual:** Asegúrate de que tu entorno virtual de Python dedicado (por ejemplo, `venv_tts`, `tts_env`) esté activado.
    ```bash
    # Activación de ejemplo (ajusta la ruta/nombre según sea necesario)
    # Windows: ..\venv_tts\Scripts\activate
    # Linux/macOS: source ../venv_tts/bin/activate
    # Conda: conda activate tts_env
    ```
-   **Ejecuta el Comando de Entrenamiento:** Ejecuta el script de entrenamiento del framework, apuntándolo a tu archivo de configuración personalizado creado en la Guía 2. La estructura exacta del comando varía entre frameworks. Los patrones comunes incluyen:
    ```bash
    # Patrón Común 1: usando el argumento --config
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Patrón Común 2: usando -c para la configuración y -m para el nombre del directorio de modelo/salida
    # (Comprueba si tu output_directory en la configuración es sobrescrito por -m)
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Patrón Común 3: especificando directamente el directorio de checkpoints
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```
    *   **Entrenamiento Multi-GPU:** Si tienes múltiples GPUs y el framework soporta entrenamiento distribuido (consulta su documentación), podrías usar comandos que involucren `torchrun` o `python -m torch.distributed.launch`. Ejemplo:
        ```bash
        # Ejemplo usando torchrun (ajusta nproc_per_node a tu número de GPUs)
        torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
        ```

### 5.2. Monitorizando el Progreso del Entrenamiento

-   **Salida en Consola:** La terminal donde lanzaste el entrenamiento mostrará información sobre el progreso. Busca:
    *   **Inicialización:** Mensajes que indican que se está construyendo el modelo, que se preparan los cargadores de datos (data loaders) y, potencialmente, que se está cargando un modelo preentrenado (para fine-tuning).
    *   **Epochs/Pasos:** El progreso actual del entrenamiento (por ejemplo, `Epoch: [1/1000]`, `Step: [500/100000]`).
    *   **Valores de Loss:** Métricas cruciales que indican qué tan bien está aprendiendo el modelo. Espera ver `train_loss` (loss del batch actual) y, periódicamente, `validation_loss` (loss sobre el conjunto de validación no visto). Ambos deberían generalmente disminuir con el tiempo. También podrían reportarse componentes de loss específicos (como `mel_loss`, `duration_loss`, `kl_loss`) según la arquitectura del modelo.
    *   **Tasa de Aprendizaje:** La tasa de aprendizaje actual podría imprimirse, especialmente si un scheduler la está reduciendo con el tiempo.
    *   **Marcas de Tiempo/Velocidad:** El tiempo que toma cada paso o epoch.
-   **TensorBoard (Muy Recomendado):** Si está habilitado en tu configuración, usa TensorBoard para una monitorización visual.
    *   **Lanzamiento:** Abre una *nueva* terminal (mantén la del entrenamiento ejecutándose), activa el mismo entorno virtual y ejecuta:
        ```bash
        # Apunta logdir al output_directory especificado en tu configuración
        tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
        ```
    *   **Acceso:** Abre la URL proporcionada por TensorBoard (normalmente `http://localhost:6006/`) en tu navegador web.
    *   **Visualiza:** Puedes ver gráficos de los losses de entrenamiento y validación a lo largo del tiempo, las programaciones de la tasa de aprendizaje y, potencialmente, otras métricas.
    *   **Escucha Muestras de Audio:** Muchos frameworks registran muestras de audio sintetizadas del conjunto de validación en TensorBoard de forma periódica (consulta la pestaña `AUDIO`). Escuchar estas muestras es la *mejor* manera de evaluar cualitativamente la mejora del modelo e identificar problemas como ruido, errores de pronunciación o salida robótica.
-   **Directorio de Salida:** Comprueba el `output_directory` que especificaste en tu configuración (`../checkpoints/my_yoruba_voice_run1`). Debería contener:
    *   Checkpoints del modelo guardados (archivos `.pth`, `.pt`, `.ckpt`).
    *   Archivos de registro (logs) (`train.log`, etc.).
    *   Copias del archivo de configuración utilizado.
    *   Archivos de eventos de TensorBoard (normalmente en un subdirectorio `logs` o `events`).
    *   Posiblemente muestras de audio sintetizadas.

### 5.3. Entendiendo los Checkpoints

-   **Qué son:** Los checkpoints son instantáneas del estado del modelo (todos sus pesos aprendidos y, potencialmente, el estado del optimizador) guardadas en intervalos específicos durante el entrenamiento.
-   **Por qué importan:**
    *   **Reanudar el Entrenamiento:** Te permiten continuar el entrenamiento si se interrumpe (debido a fallos, cortes de energía o detenciones manuales).
    *   **Evaluar el Progreso:** Puedes usar checkpoints de diferentes etapas para sintetizar audio y ver cómo evolucionó el modelo.
    *   **Seleccionar el Mejor Modelo:** El validation loss ayuda a identificar buenos checkpoints, pero el *mejor* modelo a menudo se elige escuchando el audio sintetizado de varios checkpoints prometedores cercanos al validation loss más bajo. A veces, un checkpoint ligeramente anterior suena mejor que el del loss más bajo en términos absolutos.
-   **Frecuencia de Guardado:** Configura el `save_checkpoint_interval` en tu configuración. Guardar con demasiada frecuencia consume espacio en disco; guardar con muy poca frecuencia arriesga perder un progreso significativo si ocurre un fallo. Guardar cada pocos miles de pasos o una vez por epoch es común. Muchos frameworks también guardan automáticamente el "mejor" checkpoint según el validation loss.

### 5.4. Reanudando un Entrenamiento Interrumpido

-   Si tu entrenamiento se detiene inesperadamente o lo detienes manualmente, normalmente puedes reanudarlo desde el último checkpoint guardado.
-   Encuentra la ruta al archivo de checkpoint más reciente en tu directorio de salida (por ejemplo, `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` o `latest_checkpoint.pth`).
-   Usa el argumento de reanudación del framework al lanzar el script de entrenamiento de nuevo. El nombre del argumento varía:
    ```bash
    # Ejemplo usando --resume_checkpoint
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

    # Ejemplo usando --restore_path o --resume_path
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
    ```
-   El script debería cargar los pesos del modelo y el estado del optimizador desde el checkpoint y continuar el entrenamiento desde ese punto.

### 5.5. Cuándo Detener el Entrenamiento

-   **Límite de Epochs:** El entrenamiento se detiene automáticamente cuando se alcanza el número máximo de `epochs` especificado en la configuración.
-   **Early Stopping (parada temprana):** Monitoriza el `validation_loss`. Si deja de disminuir y comienza a aumentar de forma consistente durante un período prolongado (por ejemplo, a lo largo de varios intervalos de validación), el modelo podría estar empezando a sobreajustar (overfit). Podrías considerar detener el entrenamiento manualmente alrededor del punto donde el validation loss fue más bajo.
-   **Evaluación Cualitativa:** Escucha regularmente las muestras de audio de validación generadas en TensorBoard o sintetiza muestras manualmente usando checkpoints recientes. Detén el entrenamiento cuando estés satisfecho con la calidad y la estabilidad del audio, incluso si el loss aún está disminuyendo ligeramente. Un entrenamiento adicional podría no producir mejoras perceptibles o incluso degradar la calidad.

---

## 6. Fine-tuning vs. Entrenamiento desde Cero

### 6.1. Eligiendo tu Enfoque

Al iniciar un proyecto de TTS, una de las decisiones más importantes es si hacer fine-tuning de un modelo existente o entrenar uno nuevo desde cero. Esta tabla te ayuda a decidir qué enfoque es mejor para tu situación específica:

| Factor | Fine-tuning | Entrenamiento desde Cero |
|:-------|:------------|:----------------------|
| **Tamaño del Dataset** | Funciona bien con datasets más pequeños (5-20 horas)<br>Puede producir buenos resultados con tan solo 1-2 horas para algunas voces | Normalmente requiere datasets más grandes (30+ horas)<br>Menos de 20 horas a menudo conduce a una calidad pobre |
| **Similitud de Voz** | Mejor cuando tu voz objetivo es similar a las voces de los datos de entrenamiento del modelo preentrenado | Necesario cuando tu voz objetivo es muy única o significativamente diferente de los modelos preentrenados disponibles |
| **Idioma** | Funciona bien si se hace fine-tuning dentro del mismo idioma<br>Puede funcionar para casos interlingües con una preparación cuidadosa | Requerido para idiomas sin modelos preentrenados disponibles<br>Mejor para capturar la fonética específica del idioma |
| **Tiempo de Entrenamiento** | Mucho más rápido (días en lugar de semanas)<br>Requiere menos epochs para converger | Tiempo de entrenamiento significativamente más largo<br>Puede requerir de 2 a 5 veces más epochs |
| **Requisitos de Hardware** | Requisitos de GPU similares pero durante menos tiempo<br>A menudo puede usar tamaños de batch más pequeños | Necesita acceso sostenido a la GPU durante períodos más largos<br>Puede beneficiarse más de configuraciones multi-GPU |
| **Potencial de Calidad** | Puede alcanzar una calidad excelente rápidamente<br>Puede heredar las limitaciones del modelo base | Máxima flexibilidad y potencial de calidad<br>Sin restricciones de un entrenamiento previo |
| **Estabilidad** | Proceso de entrenamiento generalmente más estable<br>Menos propenso al colapso o a la no convergencia | Más sensible a los hiperparámetros<br>Mayor riesgo de inestabilidad en el entrenamiento |

#### Cuándo Elegir Fine-tuning

El fine-tuning se recomienda generalmente cuando:
- Tienes datos limitados (menos de 20 horas)
- Necesitas resultados más rápidos
- Tu voz/idioma objetivo es razonablemente similar a los modelos preentrenados disponibles
- Tienes recursos computacionales limitados
- Eres nuevo en el entrenamiento de TTS (el fine-tuning es más indulgente)

#### Cuándo Elegir Entrenamiento desde Cero

El entrenamiento desde cero es mejor cuando:
- Tienes abundantes datos (30+ horas)
- Tu voz objetivo es muy única o tiene características no representadas en los modelos preentrenados
- Trabajas con un idioma que está pobremente soportado por los modelos existentes
- Necesitas el máximo control sobre todos los aspectos del modelo
- Tienes acceso a recursos computacionales significativos
- Estás construyendo un modelo base (foundation model) que otros harán fine-tuning

### 6.2. Aspectos Específicos del Fine-tuning

El fine-tuning aprovecha un potente modelo preentrenado y lo adapta a tu dataset específico (locutor, idioma, estilo). Normalmente es más rápido y requiere menos datos que el entrenamiento desde cero.

#### El Objetivo

-   Transferir las capacidades generales de síntesis de voz (como entender el mapeo de texto a sonido, la prosodia básica) del gran dataset con el que se entrenó el modelo base, mientras se especializa la identidad de la voz y, potencialmente, el acento/estilo para que coincidan con tu dataset más pequeño y específico.

### 6.2. Diferencias Clave en la Configuración (Resumen de la Configuración)

-   **`pretrained_model_path`:** DEBES proporcionar la ruta al archivo de checkpoint del modelo preentrenado en tu configuración.
-   **`fine_tuning: True`:** Asegúrate de que cualquier indicador (flag) que señale el modo de fine-tuning esté habilitado si el framework lo requiere.
-   **Tasa de Aprendizaje (Learning Rate):** Empieza con una tasa de aprendizaje *más baja* que la usada típicamente para el entrenamiento desde cero (por ejemplo, `1e-5`, `2e-5`, `5e-5`). Una tasa de aprendizaje alta puede destruir la valiosa información aprendida por el modelo preentrenado.
-   **Batch Size:** A menudo puede ser similar al del entrenamiento desde cero, ajústalo según la VRAM.
-   **Epochs:** El número de epochs requerido para el fine-tuning suele ser significativamente menor que para el entrenamiento desde cero, pero aún depende del tamaño del dataset y de la calidad deseada. Monitoriza de cerca el validation loss y las muestras de audio.

### 6.3. Estrategias Potenciales (Dependientes del Framework)

-   **Fine-tuning de Red Completa (Full Network):** El enfoque por defecto suele ser actualizar los pesos de toda la red, pero con una tasa de aprendizaje baja.
-   **Congelación de Capas (Freezing Layers):** Algunos frameworks permiten congelar partes de la red (por ejemplo, el codificador de texto o el predictor de duración) inicialmente y entrenar solo componentes específicos (como los speaker embeddings o el decodificador). Esto a veces puede ayudar a preservar las fortalezas del modelo base mientras se adaptan aspectos específicos. Consulta la documentación de tu framework para ver opciones como `--freeze_layers` o similares.
-   **Ignorar Capas (Ignoring Layers):** Al cargar el modelo preentrenado, podrías querer `ignore_layers` (o `reinitialize_layers`) como la capa de salida final o la capa de speaker embedding, especialmente si tu dataset tiene un número diferente de locutores que el modelo preentrenado.

### 6.4. Monitorizando el Fine-tuning

-   **Mejora Inicial Rápida:** Deberías ver el validation loss caer relativamente rápido al principio a medida que el modelo se adapta a la voz objetivo.
-   **Enfócate en la Calidad del Audio:** Presta mucha atención a las muestras de validación sintetizadas. ¿Se está desplazando la identidad de la voz hacia tu locutor objetivo? ¿Es el habla clara y estable? El fine-tuning a menudo trata más sobre la calidad perceptiva que sobre alcanzar el valor de loss mínimo absoluto.

---

## 7. Guía Completa de Resolución de Problemas

Entrenar modelos TTS puede ser un reto, con muchos problemas potenciales. Esta sección proporciona soluciones para los problemas comunes que podrías encontrar.

### 7.1. Mensajes de Error Comunes y Soluciones

| Mensaje de Error | Causas Posibles | Soluciones |
|:--------------|:----------------|:----------|
| `CUDA out of memory` | • Batch size demasiado grande<br>• Modelo demasiado grande para la GPU<br>• Fuga de memoria (memory leak) | • Reduce el batch size<br>• Habilita gradient checkpointing<br>• Usa mixed precision training<br>• Reduce la longitud de la secuencia |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | • Tipo de dato incorrecto en el dataset<br>• Tipos de tensor incompatibles | • Comprueba el preprocesamiento de datos<br>• Asegúrate de que todos los tensores tengan el dtype correcto<br>• Añade una conversión de tipo explícita |
| `ValueError: too many values to unpack` | • Desajuste entre las salidas del modelo y las expectativas de la función de loss<br>• Formato de datos incorrecto | • Comprueba la estructura de salida del modelo<br>• Verifica la implementación de la función de loss<br>• Depura las salidas del data loader |
| `FileNotFoundError: [Errno 2] No such file or directory` | • Rutas incorrectas en la configuración<br>• Archivos de datos faltantes | • Verifica todas las rutas de archivos<br>• Comprueba la integridad del archivo manifest<br>• Asegúrate de que los datos estén descargados/extraídos |
| `KeyError: 'speaker_id'` | • Información del locutor faltante<br>• Formato de dataset incorrecto | • Comprueba el formato del dataset<br>• Verifica el archivo de mapeo de locutores<br>• Añade la información del locutor al manifest |
| `Loss is NaN` | • Tasa de aprendizaje demasiado alta<br>• Inicialización inestable<br>• Explosión de gradientes | • Reduce la tasa de aprendizaje<br>• Añade recorte de gradiente (gradient clipping)<br>• Comprueba si hay división por cero<br>• Normaliza los datos de entrada |
| `ModuleNotFoundError: No module named 'X'` | • Dependencia faltante<br>• Problema con el entorno | • Instala el paquete faltante<br>• Comprueba el entorno virtual<br>• Verifica las versiones de los paquetes |
| `RuntimeError: expected scalar type Float but found Double` | • Tipos de tensor inconsistentes | • Añade `.float()` a los tensores<br>• Comprueba el preprocesamiento de datos<br>• Estandariza el dtype en todo el modelo |

### 7.2. Problemas de Calidad del Entrenamiento

| Síntoma | Causas Posibles | Soluciones |
|:--------|:----------------|:----------|
| **Audio Robótico/Zumbante** | • Problemas con el vocoder<br>• Entrenamiento insuficiente<br>• Mal preprocesamiento del audio | • Entrena el vocoder durante más tiempo<br>• Comprueba la normalización del audio<br>• Verifica la consistencia del sampling rate |
| **Omisión/Repetición de Palabras** | • Problemas de atención<br>• Entrenamiento inestable<br>• Datos insuficientes | • Usa guided attention loss<br>• Añade más variedad de datos<br>• Reduce la tasa de aprendizaje<br>• Comprueba si hay silencios largos en los datos |
| **Pronunciación Incorrecta** | • Problemas de normalización de texto<br>• Errores de fonemas<br>• Desajuste de idioma | • Mejora el preprocesamiento de texto<br>• Usa entrada basada en fonemas<br>• Añade un diccionario de pronunciación |
| **Pérdida de Identidad del Locutor** | • Overfitting al locutor dominante<br>• Speaker embeddings débiles<br>• Datos de locutor insuficientes | • Equilibra los datos de los locutores<br>• Aumenta la dimensión del speaker embedding<br>• Usa speaker adversarial loss |
| **Convergencia Lenta** | • Problemas con la tasa de aprendizaje<br>• Mala inicialización<br>• Dataset complejo | • Prueba diferentes programaciones de LR<br>• Usa transfer learning<br>• Simplifica el dataset inicialmente |
| **Entrenamiento Inestable** | • Varianza entre batches<br>• Valores atípicos en el dataset<br>• Problemas con el optimizador | • Usa gradient accumulation<br>• Limpia las muestras atípicas<br>• Prueba diferentes optimizadores |

### 7.3. Problemas Específicos del Framework

#### Coqui TTS
```
# Error: "RuntimeError: Error in applying gradient to param_name"
# Solución: comprueba si hay valores NaN en tu dataset o reduce la tasa de aprendizaje
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # Ejecuta antes del entrenamiento para depurar

# Error: "ValueError: Tacotron training requires `r` > 1"
# Solución: establece el factor de reducción correctamente en la configuración
# Ejemplo de corrección en config.json:
"r": 2  # Prueba valores entre 2-5
```

#### ESPnet
```
# Error: "TypeError: forward() missing 1 required positional argument: 'feats'"
# Solución: comprueba el formato de los datos y asegúrate de que se proporcionan los feats
# Depura la carga de datos:
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS/StyleTTS
```
# Error: "RuntimeError: expected scalar type Half but found Float"
# Solución: asegura una precisión consistente en todo el modelo
# Añade a tu script de entrenamiento:
model = model.half()  # Si usas mixed precision
# O
model = model.float()  # Si no usas mixed precision
```

### 7.4. Problemas de Hardware y Entorno

1. **Fragmentación de la Memoria de la GPU**
   - **Síntoma**: errores OOM tras varias horas de entrenamiento a pesar de tener suficiente VRAM
   - **Solución**: reinicia periódicamente el entrenamiento desde un checkpoint, usa batches más pequeños

2. **Cuellos de Botella de la CPU**
   - **Síntoma**: la utilización de la GPU fluctúa o se mantiene baja
   - **Solución**: aumenta num_workers en el DataLoader, usa almacenamiento más rápido, precachea los datasets

3. **Cuellos de Botella de E/S de Disco (Disk I/O)**
   - **Síntoma**: el entrenamiento se detiene periódicamente durante la carga de datos
   - **Solución**: usa almacenamiento SSD, aumenta el factor de prefetch, cachea el dataset en RAM

4. **Conflictos de Entorno**
   - **Síntoma**: fallos misteriosos o errores de importación
   - **Solución**: usa entornos aislados (conda/venv), comprueba la compatibilidad de CUDA/PyTorch

### 7.5. Estrategias de Depuración

1. **Aísla el Problema**
   ```bash
   # Prueba la carga de datos por separado
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"
   
   # Prueba el paso hacia adelante (forward pass) con datos ficticios
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Simplifica para Identificar Problemas**
   - Entrena con un subconjunto diminuto (10-20 muestras)
   - Desactiva temporalmente el aumento de datos (data augmentation)
   - Prueba primero con un solo locutor

3. **Visualiza las Salidas Intermedias**
   - Grafica las alineaciones de atención (attention alignments)
   - Visualiza los espectrogramas mel en diferentes etapas
   - Monitoriza las normas de los gradientes

4. **Habilita el Registro Detallado (Verbose Logging)**
   ```bash
   # Añade a tu script de entrenamiento
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

5. **Usa el Profiling de TensorBoard**
   ```python
   # Añade a tu código de entrenamiento
   from torch.profiler import profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Tu paso hacia adelante (forward pass)
   print(prof.key_averages().table())
   ```

---

Con el entrenamiento lanzado y monitorizado, el siguiente paso, tras seleccionar un buen checkpoint, es usar el modelo para generar voz a partir de texto nuevo.

**Siguiente Paso:** [Inferencia](./4_INFERENCE.md){: .btn .btn-primary} | 
[Volver Arriba](#top){: .btn .btn-primary}
