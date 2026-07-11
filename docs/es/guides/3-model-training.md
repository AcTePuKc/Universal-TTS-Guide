# Guía de Entrenamiento y Ajuste Fino de Modelos TTS


Ya tienes los datos preparados y el entorno configurado. Ahora toca entrenar o hacer fine-tuning de tu modelo TTS, seguir el progreso y manejar los checkpoints de forma segura.

Si algún término relacionado con el entrenamiento no está claro, consulta el [glosario](../glossary.md#glossary-of-technical-terms). Esta página solo explica los términos que afectan directamente al lanzamiento, la supervisión o la depuración del entrenamiento.

---

## Ejecutar el entrenamiento

Esta sección cubre cómo lanzar, supervisar y gestionar el entrenamiento.

### Lanzar el script de entrenamiento

-   **Ve al directorio correcto:** Abre tu terminal y entra en la carpeta principal del framework TTS donde está `train.py` o su equivalente.
-   **Activa el entorno virtual:** Asegúrate de tener activa la misma instalación de Python usada en la preparación.

    ```bash
    # Ejemplo de activación
    # Windows: ..\venv_tts\Scripts\activate
    # Linux/macOS: source ../venv_tts/bin/activate
    # Conda: conda activate tts_env
    ```

-   **Ejecuta el entrenamiento usando tu config personalizada:** La estructura exacta del comando depende del framework. Estos son patrones habituales:

    ```bash
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Variante con nombre de run
    # Comprueba si -m sustituye output_directory en tu config.
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Variante indicando directamente la carpeta de checkpoints
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```

-   **Entrenamiento multi-GPU:** Si tienes varias GPU y tu framework admite entrenamiento distribuido, consulta su documentación y puedes usar `torchrun`.

    ```bash
    # Ejemplo con torchrun; ajusta nproc_per_node al número de GPU
    torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
    ```

#### Primera comprobación del arranque completo

Antes de dejar un entrenamiento corriendo durante horas o días, verifica en los primeros minutos:

- el script supera el arranque y empieza a cargar batches reales
- la carpeta de salida empieza a recibir logs, checkpoints o archivos de eventos
- las loss aparecen como números normales, no como `NaN` o `inf`
- el uso de memoria GPU se estabiliza en lugar de crecer hasta caer inmediatamente

Si el trabajo falla aquí, corrige eso primero. Unos primeros cinco minutos rotos suelen convertirse en un día entero perdido.

### Supervisar el progreso

-   **Salida en consola:** Observa:
    *   la inicialización del modelo y de los data loaders
    *   epoch o step actual
    *   `train_loss` y `validation_loss`
    *   learning rate
    *   tiempo por step o epoch
-   **TensorBoard:** Si está activado en tu config, ejecútalo en otra terminal.

    ```bash
    tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
    ```

    Abre la URL que muestre TensorBoard (normalmente `http://localhost:6006/`). Allí podrás seguir las curvas de loss, el learning rate y, si el framework los registra, los samples de validación sintetizados.

-   **Directorio de salida:** Debería contener checkpoints (`.pth`, `.pt` o `.ckpt`), logs, una copia del config, archivos de eventos de TensorBoard y, a veces, samples de audio sintetizados.

#### Qué suele parecer un buen progreso temprano

En un primer run sano normalmente quieres ver:

- entrenamiento sin caídas inmediatas, `NaN` o memoria descontrolada
- `train_loss` y `validation_loss` bajando con el tiempo en lugar de explotar
- muestras de validación cada vez más claras y consistentes
- checkpoints posteriores que suenan mejor que los muy tempranos, aunque aún no sean perfectos

No te obsesiones con un solo número de loss. En TTS, escuchar importa tanto como las métricas.

### Entender los checkpoints

-   Los checkpoints son instantáneas del estado del modelo —sus pesos aprendidos y, a menudo, el estado del optimizador— guardadas durante el entrenamiento.
-   Son importantes para:
    *   **reanudar un entrenamiento interrumpido**
    *   **comparar diferentes momentos del entrenamiento**
    *   **elegir el mejor modelo**
-   **Frecuencia de guardado:** Ajusta un `save_checkpoint_interval` razonable. Guardar demasiado a menudo consume disco; guardar demasiado poco aumenta el riesgo de perder progreso valioso. Muchos frameworks también guardan automáticamente el checkpoint «best» según la validation loss.

### Reanudar un entrenamiento interrumpido

Si el entrenamiento se detiene de forma inesperada, normalmente puedes retomarlo desde el último checkpoint:

- Busca el checkpoint más reciente en la carpeta de salida, por ejemplo `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` o `latest_checkpoint.pth`.
- Usa el argumento de reanudación del framework; el nombre puede variar:

```bash
python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
```

El script debería cargar los pesos y el estado del optimizador y continuar desde el mismo punto.

### Cuándo detener el entrenamiento

-   **Límite de epochs:** el entrenamiento se detiene al alcanzar el máximo de `epochs` indicado en el config.
-   **Parada temprana:** si validation loss deja de bajar y empieza a subir de forma constante durante varios intervalos de validación, el modelo puede estar empezando a sufrir overfitting.
-   **Evaluación mediante escucha:** escucha con regularidad los samples de validación y detén el entrenamiento cuando la calidad y la estabilidad sean suficientes para tu objetivo, aunque la loss siga bajando ligeramente.

A veces un checkpoint ligeramente anterior suena mejor que el de loss mínima absoluta.

---

## Fine-tuning frente a entrenamiento desde cero

### Elegir el enfoque

Cuando empiezas un proyecto TTS, una de las decisiones más importantes es si hacer fine-tuning sobre un modelo existente o entrenar uno nuevo desde cero. Esta tabla te ayuda a elegir el enfoque más adecuado para tu caso:

| Factor | Fine-tuning | Entrenamiento desde cero |
|:-------|:------------|:-------------------------|
| **Tamaño del dataset** | Funciona bien con datasets más pequeños (5-20 horas)<br>En algunas voces puede dar resultados útiles incluso con 1-2 horas | Normalmente requiere datasets más grandes (30+ horas)<br>Con menos de 20 horas suele dar peores resultados |
| **Similitud de voz** | Va mejor cuando la voz objetivo se parece a las voces del modelo base | Es preferible cuando la voz objetivo es muy particular o muy distinta a los modelos disponibles |
| **Idioma** | Funciona bien si haces fine-tuning dentro del mismo idioma<br>Puede funcionar entre idiomas con una preparación cuidadosa | Es necesario cuando no existen buenos modelos base para ese idioma<br>Captura mejor fonética específica del idioma |
| **Tiempo de entrenamiento** | Mucho más rápido (días en lugar de semanas)<br>Necesita menos epochs para converger | Requiere bastante más tiempo<br>Puede necesitar entre 2 y 5 veces más epochs |
| **Requisitos de hardware** | Necesidades similares de GPU, pero durante menos tiempo<br>A menudo tolera batch sizes más pequeños | Requiere acceso sostenido a GPU durante más tiempo<br>Se beneficia más de configuraciones multi-GPU |
| **Potencial de calidad** | Puede alcanzar buena calidad rápidamente<br>Puede heredar límites del modelo base | Ofrece máxima flexibilidad y potencial de calidad<br>No depende de restricciones de un entrenamiento previo |
| **Estabilidad** | Suele ser más estable<br>Es menos propenso a colapsar o no converger | Es más sensible a hiperparámetros<br>Tiene más riesgo de inestabilidad |

#### Cuándo elegir fine-tuning

- Tienes datos limitados
- Necesitas resultados más rápido
- Tu voz o idioma se parece a un modelo preentrenado disponible
- Tienes recursos computacionales limitados
- Eres nuevo en entrenamiento TTS, porque el fine-tuning suele ser más tolerante

#### Cuándo elegir entrenamiento desde cero

- Tienes muchos datos (30+ horas)
- La voz objetivo es muy distinta o tiene rasgos poco representados en modelos base
- No existe un buen modelo base para tu idioma
- Necesitas el máximo control sobre todos los aspectos del modelo
- Tienes acceso a recursos computacionales importantes
- Estás construyendo un modelo base que otros podrían luego ajustar

### Particularidades del fine-tuning

El fine-tuning aprovecha un modelo base potente y lo adapta a tu dataset específico, ya sea una voz, un idioma o un estilo. Normalmente es más rápido y requiere menos datos que entrenar desde cero.

#### El objetivo

- Transferir las capacidades generales de síntesis del modelo base, como la relación texto-audio y una prosodia razonable, mientras adaptas la identidad de la voz y posiblemente el acento o el estilo a tu dataset más pequeño.

### Diferencias clave en la configuración

- **`pretrained_model_path`:** Debes indicar la ruta al checkpoint del modelo preentrenado dentro de la configuración.
- **`fine_tuning: True`:** Activa cualquier bandera de modo fine-tuning si tu framework la requiere.
- **Learning rate:** Empieza con un learning rate más bajo que el usado para entrenar desde cero, por ejemplo `1e-5`, `2e-5` o `5e-5`. Un learning rate alto puede destruir información valiosa del modelo base.
- **Batch size:** A menudo puede ser parecido al de entrenamiento desde cero, ajustado según la VRAM disponible.
- **Epochs:** Suelen ser menos que en entrenamiento desde cero, pero siguen dependiendo del tamaño del dataset y de la calidad buscada. Observa de cerca la validation loss y las muestras de audio.

### Estrategias de fine-tuning

- **Fine-tuning completo de la red:** La opción por defecto suele ser actualizar toda la red con un learning rate bajo.
- **Congelar capas:** Algunos frameworks permiten congelar partes de la red al principio, como el encoder de texto o el predictor de duración, y entrenar solo componentes concretos. Consulta la documentación para opciones como `--freeze_layers`.
- **Ignorar o reinicializar capas:** Al cargar el modelo preentrenado, puede ser útil usar `ignore_layers` o `reinitialize_layers` para capas como la salida final o los speaker embeddings, especialmente si tu dataset tiene un número de speakers diferente.

### Qué vigilar durante el fine-tuning

- **Mejora rápida al principio:** La validation loss debería bajar con relativa rapidez al comienzo.
- **Calidad perceptiva:** Escucha las muestras generadas. La voz debería acercarse al hablante objetivo sin perder claridad ni estabilidad.
- **Estabilidad:** Vigila si aparecen artefactos extraños, repeticiones o degradación al continuar entrenando.

El fine-tuning suele depender más de la calidad percibida que de alcanzar la loss mínima absoluta.

---

## Guía de problemas durante el entrenamiento

Entrenar modelos TTS puede ser complicado y producir distintos problemas. Las secciones siguientes ofrecen orientación para los casos más habituales.

### Errores comunes y qué suelen indicar

| Error | Posibles causas | Soluciones |
|:------|:----------------|:-----------|
| `CUDA out of memory` | Batch size demasiado grande<br>Modelo demasiado pesado para la GPU<br>Fuga o presión de memoria | Reduce batch size<br>Activa gradient checkpointing<br>Usa mixed precision<br>Reduce la longitud de secuencia |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | Tipo de dato incorrecto en el dataset<br>Tensores incompatibles | Revisa el preprocessing<br>Asegúrate de que todos los tensores tengan el dtype correcto<br>Añade conversiones explícitas de tipo |
| `ValueError: too many values to unpack` | Desajuste entre salidas del modelo y la loss<br>Formato de datos incorrecto | Revisa la estructura de salida del modelo<br>Verifica la implementación de la loss<br>Depura la salida del data loader |
| `FileNotFoundError: [Errno 2] No such file or directory` | Rutas incorrectas en el config<br>Faltan archivos de datos | Verifica todas las rutas<br>Comprueba la integridad de los manifests<br>Asegúrate de haber descargado o extraído todos los datos |
| `KeyError: 'speaker_id'` | Falta información del locutor<br>Formato de dataset incorrecto | Revisa el formato del dataset<br>Verifica el archivo de mapeo de speakers<br>Añade la información de speaker al manifest |
| `Loss is NaN` | Learning rate demasiado alto<br>Inicialización inestable<br>Explosión de gradientes | Baja el learning rate<br>Añade gradient clipping<br>Comprueba divisiones por cero<br>Normaliza los datos de entrada |
| `ModuleNotFoundError: No module named 'X'` | Falta una dependencia<br>Problema de entorno | Instala el paquete faltante<br>Revisa el entorno virtual<br>Verifica versiones de paquetes |
| `RuntimeError: expected scalar type Float but found Double` | Tipos de tensor inconsistentes | Añade `.float()` a los tensores<br>Revisa el preprocessing<br>Unifica el dtype en todo el modelo |

### Problemas de calidad

| Síntoma | Posibles causas | Soluciones |
|:--------|:----------------|:-----------|
| **Audio robótico o zumbante** | Problemas del vocoder<br>Entrenamiento insuficiente<br>Preprocessing de audio pobre | Entrena más el vocoder<br>Revisa la normalización de audio<br>Verifica la coherencia del sampling rate |
| **Palabras omitidas o repetidas** | Problemas de atención<br>Entrenamiento inestable<br>Datos insuficientes | Usa guided attention loss<br>Añade más variedad de datos<br>Baja el learning rate<br>Busca silencios largos en el dataset |
| **Pronunciación incorrecta** | Problemas de normalización de texto<br>Errores de fonemas<br>Desajuste de idioma | Mejora el preprocessing de texto<br>Usa entrada basada en fonemas<br>Añade un diccionario de pronunciación |
| **Pérdida de identidad del locutor** | Sobreajuste al speaker dominante<br>Speaker embeddings débiles<br>Pocos datos por speaker | Equilibra los datos de speakers<br>Aumenta la dimensión de speaker embedding<br>Revisa la estrategia multi-speaker |
| **Convergencia lenta** | Problemas de learning rate<br>Mala inicialización<br>Dataset complejo | Prueba otra estrategia de learning rate<br>Usa transfer learning<br>Simplifica el dataset al principio |
| **Entrenamiento inestable** | Alta varianza entre batches<br>Outliers en el dataset<br>Problemas del optimizador | Usa gradient accumulation<br>Limpia muestras atípicas<br>Prueba otro optimizador |

### Problemas de framework y entorno

#### Coqui TTS

```bash
# Error: "RuntimeError: Error in applying gradient to param_name"
# Solución: Busca valores NaN en tu dataset o reduce el learning rate
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # Ejecuta esto antes del entrenamiento para depurar
```

```bash
# Error: "ValueError: Tacotron training requires `r` > 1"
# Solución: Ajusta correctamente el reduction factor en el config
# Ejemplo de corrección en config.json:
"r": 2  # Prueba valores entre 2 y 5
```

#### ESPnet

```bash
# Error: "TypeError: forward() missing 1 required positional argument: 'feats'"
# Solución: Revisa el formato de los datos y asegúrate de proporcionar feats
# Depurar carga de datos:
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS / StyleTTS

```python
# Error: "RuntimeError: expected scalar type Half but found Float"
# Solución: Mantén una precisión consistente en todo el modelo
# Añade a tu script de entrenamiento:
model = model.half()  # Si usas mixed precision
# O
model = model.float()  # Si no usas mixed precision
```

### Problemas de hardware y entorno

1. **Fragmentación de memoria GPU**
   - **Síntoma:** errores OOM tras varias horas aunque la VRAM debería bastar
   - **Solución:** reinicia el entrenamiento desde un checkpoint periódicamente y prueba batches más pequeños

2. **Cuello de botella de CPU**
   - **Síntoma:** uso de GPU bajo o muy irregular
   - **Solución:** aumenta `num_workers` en el DataLoader, usa almacenamiento más rápido y precachea datasets si es posible

3. **Cuello de botella de I/O de disco**
   - **Síntoma:** pausas periódicas durante la carga de datos
   - **Solución:** usa SSD, aumenta el prefetch factor o cachea el dataset en RAM

4. **Conflictos de entorno**
   - **Síntoma:** cierres extraños o errores de importación difíciles de explicar
   - **Solución:** usa entornos aislados, revisa compatibilidad entre CUDA y PyTorch, y evita mezclar instalaciones antiguas

5. **Activa el logging detallado:** Si necesitas más información, añade al script de entrenamiento:

   ```python
   # Añade esto al script de entrenamiento
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

6. **Usa el profiling de TensorBoard:** Cuando sea necesario, perfila el tiempo de CPU/GPU para localizar cuellos de botella:

   ```python
   # Añade esto al código de entrenamiento
   from torch.profiler import profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Tu forward pass
           pass
   print(prof.key_averages().table())
   ```

### Estrategias de depuración

1. **Aísla el problema**

   ```bash
   # Probar la carga de datos por separado
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"

   # Probar un forward pass con datos de ejemplo
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Simplifica para identificar el fallo**
   - Entrena con un subconjunto muy pequeño y limpio
   - Desactiva augmentation temporalmente
   - Usa una configuración más pequeña si el framework lo permite

3. **Revisa artefactos intermedios**
   - Mira alignments de attention, mel spectrograms, logs y muestras de validación
   - Comprueba si el problema aparece desde el principio o tras varios checkpoints

4. **Añade más visibilidad**
   - Activa logging más detallado si existe
   - Guarda más muestras intermedias
   - Usa `torch.autograd.set_detect_anomaly(True)` solo mientras depuras, no como ajuste permanente de entrenamiento

---

Una vez que el entrenamiento esté en marcha y correctamente supervisado, el siguiente paso es elegir un buen checkpoint y usar el modelo para generar voz a partir de texto nuevo.
