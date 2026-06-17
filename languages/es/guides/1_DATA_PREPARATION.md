# Guía 1: Preparación de Datos para el Entrenamiento de TTS

**Navegación:** [README Principal]({{ site.baseurl }}/languages/es/){: .btn .btn-primary} | [Siguiente Paso: Configuración del Entrenamiento](./2_TRAINING_SETUP.md){: .btn .btn-primary} | 

Esta guía cubre la primera fase crítica de cualquier proyecto de TTS: preparar datos de audio y texto de alta calidad y con el formato correcto. La calidad de tu dataset impacta directamente en la calidad de tu modelo TTS final.

---

## 1. Pasos de Preparación del Dataset

Sigue estos pasos de forma sistemática para transformar el audio en bruto en un dataset listo para entrenar.

### 1.1. Adquisición de Audio y Procesamiento Inicial

-   **Reunir Audio:** Recopila tus archivos de audio en bruto (los formatos comunes incluyen WAV, MP3, FLAC, OGG, M4A). Asegúrate de tener los derechos para usar este audio.
-   **Convertir a WAV:** La mayoría de los frameworks de TTS esperan el formato WAV. Usa herramientas como `ffmpeg` o librerías de audio (`pydub`, `soundfile`) para convertir tu audio. Apunta a una codificación WAV estándar como PCM de 16 bits.
    ```bash
    # Ejemplo usando ffmpeg para convertir MP3 a WAV
    ffmpeg -i input_audio.mp3 output_audio.wav
    ```
-   **Estandarizar Canales (Mono):** Los modelos TTS normalmente se entrenan con audio de un solo canal (mono). Convierte las pistas estéreo a mono.
    ```bash
    # Ejemplo usando ffmpeg para convertir WAV estéreo a WAV mono
    ffmpeg -i stereo_input.wav -ac 1 mono_output.wav
    ```
    *   `-ac 1`: Establece el número de canales de audio en 1.
-   **Remuestrear Audio:** Asegúrate de que todos los archivos de audio tengan **exactamente el mismo sampling rate**. Elige tu frecuencia objetivo según los objetivos de tu proyecto y la compatibilidad con el framework (por ejemplo, 16000 Hz, 22050 Hz, 48000 Hz). 22050 Hz es común para muchos modelos.
    ```bash
    # Ejemplo usando ffmpeg para remuestrear a 22050 Hz
    ffmpeg -i input.wav -ar 22050 resampled_output.wav
    ```
    *   `-ar 22050`: Establece el sampling rate del audio (muestras por segundo).

### 1.2 Limpieza Avanzada del Audio (Eliminación de Ruido/Música) - *Opcional pero Recomendado*

-   **Objetivo:** Eliminar sonidos de fondo no deseados como ruido (zumbidos, siseos, ventiladores), música, reverberación u otras voces que interfieran de tu audio fuente, aislando la voz del locutor objetivo en la medida de lo posible. Este paso es crucial si tu audio fuente no tiene calidad de estudio.
-   **¿Por qué?** Los modelos TTS aprenden del audio que se les proporciona. Si el audio contiene ruido de fondo o música, es probable que la voz TTS resultante herede estas características, sonando ruidosa o "turbia". Un audio más limpio conduce a una voz TTS más limpia.

-   **Herramientas y Técnicas:**
    *   **Herramientas de Separación de Fuentes por IA (Recomendadas para Música/Voz):** Estas herramientas usan modelos de IA para separar el audio en diferentes pistas (stems) (voces, música, batería, bajo, otros).
        *   **[Ultimate Vocal Remover (UVR)](https://ultimatevocalremover.com/)**: Una aplicación con interfaz gráfica (GUI) popular, gratuita y de código abierto que proporciona acceso a varios modelos de separación por IA de última generación. Es excelente para eliminar música de fondo o aislar diálogos.
            *   **Modelos (como los mencionados):** UVR te permite usar diferentes modelos de IA. `MDX-Inst-HQ3` es uno de esos modelos, a menudo bueno para separar voces de instrumentos (de ahí "Inst"). Otros modelos MDX, los modelos Demucs (como `htdemucs`) y, potencialmente, modelos como Mel-Roformer (si está integrado o disponible de forma independiente) están diseñados para tareas similares, cada uno con fortalezas y debilidades ligeramente diferentes. La experimentación es clave. Elige modelos enfocados en el **aislamiento de voz**.
        *   **Otras Herramientas:** Los servicios en línea (por ejemplo, Lalal.ai) u otro software independiente podrían usar modelos subyacentes similares (a menudo variantes de Demucs o Spleeter).
    *   **Herramientas Tradicionales de Reducción de Ruido:** A menudo se encuentran en estaciones de trabajo de audio digital (DAWs) o editores de audio.
        *   **[Audacity](https://www.audacityteam.org/):** Contiene efectos integrados de reducción de ruido (requiere muestrear un perfil de ruido). Puede ser eficaz para el ruido de fondo constante (como siseos o zumbidos).
        *   **Plugins Comerciales (por ejemplo, Izotope RX, Waves Clarity):** Ofrecen herramientas de aislamiento de voz, ruido y reverberación más sofisticadas y basadas en IA, pero tienen un coste.
    *   **Edición Espectral:** Eliminar manualmente sonidos no deseados en un editor espectral (como Adobe Audition, Izotope RX, Acon Digital Acoustica). Potente, pero muy laborioso.

-   **Consideraciones sobre el Flujo de Trabajo:**
    *   **Cuándo Aplicarlo:** Generalmente se recomienda aplicar la limpieza a tus **archivos de audio más largos *antes* de la segmentación (Paso 1.3 a continuación)**. Esto permite que los modelos de IA trabajen con más contexto y puede ser más eficiente que procesar miles de segmentos pequeños. Sin embargo, si la limpieza introduce demasiados artefactos en los archivos largos, podrías intentar limpiar individualmente los segmentos problemáticos más adelante.
    *   **Proceso:**
        1.  Carga tu archivo WAV estandarizado (del Paso 1.1) en la herramienta elegida (por ejemplo, UVR).
        2.  Selecciona un modelo de aislamiento de voz apropiado (por ejemplo, un modelo de voz MDX o Demucs).
        3.  Procesa el audio para generar una pista de "solo voces".
        4.  **Escucha con Atención:** Evalúa críticamente la pista de voz separada. Comprueba si hay:
            *   **Artefactos:** La separación por IA a veces puede introducir sonidos "acuosos", fallos (glitches), o partes de la voz que se eliminan por error.
            *   **Ruido/Música Restantes:** ¿Con qué eficacia se eliminó el sonido no deseado?
        5.  **Itera:** Es posible que necesites probar diferentes modelos, ajustar la configuración dentro de la herramienta, o incluso aplicar una segunda pasada de reducción de ruido (por ejemplo, usando la reducción de ruido de Audacity sobre las voces separadas por IA) para obtener los mejores resultados.
    *   **Guardar la Salida:** Guarda la pista de voz limpia como un nuevo archivo WAV (por ejemplo, `original_file_cleaned.wav`). Usa estos archivos limpios como entrada para el *siguiente* paso (Segmentación).

-   **Advertencias:**
    *   **Los Artefactos son Posibles:** Una limpieza agresiva puede degradar la naturalidad de la voz objetivo. Busca un equilibrio entre eliminar el ruido y preservar la calidad de la voz.
    *   **Coste Computacional:** Los modelos de separación por IA pueden ser computacionalmente intensivos y pueden requerir un tiempo significativo, especialmente en archivos de audio largos y sin una GPU potente.


### 1.3. Segmentación del Audio (División en Segmentos)

-   **Objetivo:** Dividir archivos de audio largos (como capítulos de un audiolibro o episodios de podcast) en segmentos más cortos y manejables. La duración ideal de un segmento suele estar entre **2 y 15 segundos**.
-   **¿Por qué Segmentar?**
    *   Alinea la duración del audio con las longitudes típicas de las frases.
    *   Hace factible la transcripción (transcribir archivos de horas de duración es difícil).
    *   Ayuda a gestionar la memoria durante el entrenamiento.
    *   Permite filtrar segmentos no aptos (por ejemplo, silencio puro, ruido, música).
-   **Método:** Usa herramientas que detecten el silencio para dividir el audio. `pydub` es una librería de Python popular para esto.

    ```python
    # Ejemplo usando pydub para la división basada en silencio
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import os

    input_file = "resampled_mono_audio.wav" # Usa la salida del paso 1.1
    output_dir = "audio_chunks"             # Crea este directorio
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading audio file: {input_file}")
    sound = AudioSegment.from_wav(input_file)
    print("Audio loaded. Splitting based on silence...")

    chunks = split_on_silence(
        sound,
        min_silence_len=500,    # Duración mínima del silencio en milisegundos para activar una división. Ajusta según sea necesario.
        silence_thresh=-40,     # Umbral de silencio en dBFS (decibelios relativos a la escala completa). Valores más bajos (p. ej., -50) detectan silencios más tenues. Ajusta según el nivel de ruido de fondo de tu audio.
        keep_silence=200        # Opcional: cantidad de silencio (en ms) a dejar al inicio/final de cada segmento. Ayuda a evitar cortes bruscos.
    )

    print(f"Found {len(chunks)} potential chunks before duration filtering.")

    # --- Filtrado y Exportación ---
    min_duration_sec = 2.0  # Duración mínima del segmento en segundos
    max_duration_sec = 15.0 # Duración máxima del segmento en segundos
    target_sr = 22050       # Asegura que los segmentos conserven el sampling rate correcto (pydub suele encargarse de esto)

    exported_count = 0
    for i, chunk in enumerate(chunks):
        duration_sec = len(chunk) / 1000.0
        if min_duration_sec <= duration_sec <= max_duration_sec:
            # Asegura que el segmento use el sampling rate objetivo si es necesario (pydub intenta preservarlo)
            # chunk = chunk.set_frame_rate(target_sr) # Normalmente no es necesario si la fuente se muestreó correctamente
            
            chunk_filename = f"segment_{exported_count:05d}.wav" # Usa relleno (padding) para facilitar la ordenación
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            print(f"Exporting chunk {i} ({duration_sec:.2f}s) to {chunk_path}")
            chunk.export(chunk_path, format="wav")
            exported_count += 1
        else:
             print(f"Skipping chunk {i} due to duration: {duration_sec:.2f}s")


    print(f"\nExported {exported_count} chunks meeting duration criteria ({min_duration_sec}-{max_duration_sec}s) to '{output_dir}'.")
    ```
-   **Revisión:** Escucha una muestra de los segmentos generados. ¿Son lógicas las divisiones? ¿Se corta el habla? Ajusta `min_silence_len` y `silence_thresh` y vuelve a ejecutar si es necesario. Para audio complicado podría ser necesario dividir o refinar las divisiones manualmente en un editor de audio (como Audacity).

### 1.4. Normalización de Volumen

-   **Objetivo:** Asegurar que todos los segmentos de audio tengan un nivel de volumen consistente. Esto evita que los segmentos silenciosos o ruidosos afecten de forma desproporcionada al entrenamiento.
-   **Métodos:**
    *   **Normalización de Pico (Peak Normalization):** Ajusta el audio para que el punto más alto alcance un nivel específico (por ejemplo, -3.0 dBFS). Simple, pero no garantiza una sonoridad *percibida* consistente.
    *   **Normalización de Sonoridad (LUFS):** Ajusta el audio para alcanzar un nivel de sonoridad percibida objetivo (por ejemplo, -23 LUFS es común para radiodifusión). Generalmente preferida, ya que refleja mejor la audición humana. Requiere librerías como `pyloudnorm`.
-   **Aplicar de Forma Consistente:** Aplica el método de normalización elegido a *todos* los segmentos creados en el paso anterior. Guarda los archivos normalizados en un **nuevo directorio** (por ejemplo, `normalized_chunks`) para mantener los originales intactos.

    ```python
    # Ejemplo usando pydub para la normalización de PICO (PEAK)
    from pydub import AudioSegment
    import os
    import glob

    input_chunk_dir = "audio_chunks"
    output_norm_dir = "normalized_chunks"
    os.makedirs(output_norm_dir, exist_ok=True)
    
    target_dBFS = -3.0 # Amplitud de pico objetivo

    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    print(f"Normalizing chunks from '{input_chunk_dir}' to '{output_norm_dir}' with target peak {target_dBFS} dBFS.")
    
    wav_files = glob.glob(os.path.join(input_chunk_dir, "*.wav"))
    
    for i, wav_file in enumerate(wav_files):
        filename = os.path.basename(wav_file)
        output_path = os.path.join(output_norm_dir, filename)
        
        try:
            sound = AudioSegment.from_wav(wav_file)
            # Aplica ganancia solo si el sonido no es silencio (dBFS no es -inf)
            if sound.dBFS > -float('inf'):
              normalized_sound = match_target_amplitude(sound, target_dBFS)
              normalized_sound.export(output_path, format="wav")
            else:
              print(f"Skipping silent file: {filename}")
              # Opcionalmente copia los archivos silenciosos o gestiónalos según sea necesario
              # shutil.copy(wav_file, output_path) 
            
            if (i + 1) % 50 == 0: # Imprime el progreso
                 print(f"Processed {i+1}/{len(wav_files)} files...")
                 
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nNormalization complete. Normalized files saved in '{output_norm_dir}'.")
    ```
    *   **Nota:** Para la normalización LUFS, usarías una librería como `pyloudnorm`, iterando por los archivos de forma similar.

### 1.5. Transcripción: Creación de Pares de Texto

-   **Objetivo:** Obtener una transcripción de texto precisa para *cada uno de los segmentos de audio normalizados*. El texto debe representar *exactamente* lo que se dice en el audio.
-   **Métodos:**
    *   **Reconocimiento Automático del Habla (ASR):** Lo mejor para datasets grandes. Usa modelos ASR de alta calidad.
    *   **[OpenAI Whisper](https://github.com/openai/whisper):** Una excelente opción multilingüe y de código abierto. Se ejecuta localmente (se recomienda GPU) o a través de API. *Nota: Aunque es potente para la precisión de las palabras, la puntuación y el uso de mayúsculas de Whisper pueden requerir una revisión y corrección cuidadosas durante el paso de limpieza.* Varios modelos de Whisper con fine-tuning de la comunidad (a menudo encontrados en Hugging Face) pueden ofrecer mejoras.
    *   **[Modelos Google Gemini](https://ai.google.dev/) (por ejemplo, vía API o AI Studio):** Modelos como Gemini Pro o Flash pueden realizar la transcripción de audio. A menudo requieren que el audio esté en formatos específicos y pueden rendir mejor en segmentos más cortos (alineándose bien con el paso previo de segmentación). Comprueba las ofertas actuales de la API y los posibles niveles gratuitos.
    *   **Servicios en la Nube:** Google Cloud Speech-to-Text, AWS Transcribe, Azure Speech Service ofrecen APIs robustas, a menudo con precios de pago por uso y posibles niveles gratuitos iniciales.
    *   **Otros Modelos:** Explora [Hugging Face Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) para encontrar otros modelos ASR de código abierto o con fine-tuning específicos para tu idioma.
    *   **Transcripción Manual:** La más precisa, pero muy laboriosa. Adecuada para datasets pequeños y de alto valor o para *corregir las salidas de ASR*.
    *   **Transcripciones Existentes:** Si tu audio fuente viene con transcripciones alineadas (por ejemplo, algunos audiolibros, archivos de radiodifusión), es posible que necesites scripts para analizarlas y alinearlas con tus segmentos.
-   **Formato de Salida:** Crea un archivo `.txt` por cada archivo `.wav` correspondiente en tu directorio `normalized_chunks`. Los nombres de los archivos deben coincidir exactamente (por ejemplo, `normalized_chunks/segment_00001.wav` necesita `transcripts/segment_00001.txt`).
-   **Limpieza y Normalización del Texto:** **¡Esto es crucial!**
    *   **Eliminar Elementos que no son Habla:** Elimina marcas de tiempo (como `[00:01:05]`), etiquetas de locutor ("SPEAKER A:", "John Doe:"), etiquetas de eventos sonoros (`[laughter]`, `[music]`), comentarios de transcripción.
    *   **Gestionar Muletillas (Filler Words):** Decide si conservar o eliminar las muletillas comunes ("uh", "um", "ah"). Conservarlas puede hacer que el TTS suene más natural, pero también puede introducir vacilaciones no deseadas. Eliminarlas conduce a un habla más limpia y directa. La consistencia es clave.
    *   **Puntuación:** Asegura una puntuación consistente y apropiada. Las comas, puntos y signos de interrogación ayudan al modelo a aprender la prosodia. Evita la puntuación excesiva o no estándar.
    *   **Números, Acrónimos, Símbolos:** Expándelos a palabras (por ejemplo, "101" -> "ciento uno", "EE. UU." -> "E E U U" o "Estados Unidos de América", "%" -> "por ciento"). La forma de expandirlos depende de cómo quieras que el TTS los pronuncie. Crea un diccionario/conjunto de reglas de normalización si es necesario.
    *   **Mayúsculas/Minúsculas:** Normalmente convierte el texto a un formato consistente (por ejemplo, minúsculas), a menos que tu framework/tokenizador de TTS gestione las mayúsculas adecuadamente. Consulta la documentación del framework.
    *   **Caracteres Especiales:** Elimina o reemplaza los caracteres que podrían confundir al tokenizador (por ejemplo, emojis, caracteres de control).

    ```
    # Ejemplo de estructura:
    my_tts_dataset/
    ├── normalized_chunks/
    │   ├── segment_00001.wav
    │   ├── segment_00002.wav
    │   └── ...
    └── transcripts/
        ├── segment_00001.txt  # Contiene "Hola mundo."
        ├── segment_00002.txt  # Contiene "Esta es una frase de prueba."
        └── ...
    ```

### 1.6. Estructuración de los Datos y Creación del Archivo Manifest

-   **Objetivo:** Crear archivos de índice (manifests) que le indiquen al script de entrenamiento de TTS dónde encontrar los archivos de audio y sus transcripciones correspondientes.
-   **Formato del Manifest:** El formato más común es un archivo de texto plano donde cada línea representa un par de audio-texto, separados por un delimitador (normalmente una barra vertical `|`).
    ```
    path/to/audio_chunk.wav|El texto de transcripción correspondiente|speaker_id
    ```
    *   `path/to/audio_chunk.wav`: Ruta relativa al archivo de audio normalizado desde el directorio donde se ejecutará el script de entrenamiento.
    *   `El texto de transcripción correspondiente`: El texto limpio y normalizado del archivo `.txt`.
    *   `speaker_id`: Un identificador para el locutor (por ejemplo, `speaker0`, `mary_smith`). Para datasets de un solo locutor, usa el mismo ID para todas las líneas. Para datasets multi-locutor, usa IDs únicos para cada locutor distinto.
-   **División de los Datos (Entrenamiento/Validación):** Divide tus datos en un conjunto de entrenamiento (usado para actualizar los pesos del modelo) y un conjunto de validación (usado para monitorizar el rendimiento sobre datos no vistos y prevenir el overfitting). Una división común es del 90-98% para entrenamiento y del 2-10% para validación. **De forma crucial, asegúrate de que los segmentos de una *misma grabación larga original* no acaben en ambos conjuntos (entrenamiento y validación) si es posible, para evitar fugas de datos (data leakage).** Si divides de forma aleatoria, mezcla (shuffle) primero.
-   **Script para Generar Manifests:**

    ```python
    import os
    import random

    # --- Configuración ---
    dataset_name = "my_tts_dataset"
    normalized_audio_dir = os.path.join(dataset_name, "normalized_chunks")
    transcripts_dir = os.path.join(dataset_name, "transcripts")
    output_dir = dataset_name # Donde se guardarán los archivos manifest

    train_manifest_path = os.path.join(output_dir, "train_list.txt")
    val_manifest_path = os.path.join(output_dir, "val_list.txt")

    speaker_id = "main_speaker" # Usa un ID consistente para datasets de un solo locutor
                                # Para multi-locutor, determina el ID según el nombre de archivo o la fuente
    val_split_ratio = 0.05    # 5% para el conjunto de validación
    random_seed = 42          # Para divisiones reproducibles
    # ---------------------

    manifest_entries = []
    print("Reading audio and transcript files...")

    # Itera por los archivos de audio normalizados
    wav_files = sorted([f for f in os.listdir(normalized_audio_dir) if f.endswith(".wav")])

    for wav_filename in wav_files:
        base_filename = os.path.splitext(wav_filename)[0]
        txt_filename = base_filename + ".txt"
        
        audio_path = os.path.join(normalized_audio_dir, wav_filename)
        # Usa os.path.relpath si tu script de entrenamiento se ejecuta desde una raíz diferente
        # relative_audio_path = os.path.relpath(audio_path, start=training_script_dir) 
        relative_audio_path = audio_path # Asumiendo que el script se ejecuta desde la raíz que contiene 'my_tts_dataset'

        transcript_path = os.path.join(transcripts_dir, txt_filename)

        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                # Limpieza básica: elimina caracteres de barra vertical, recorta espacios sobrantes
                transcript = transcript.replace('|', ' ').strip()
                transcript = ' '.join(transcript.split()) # Normaliza los espacios en blanco

                if transcript: # Asegura que la transcripción no esté vacía tras la limpieza
                    manifest_entries.append(f"{relative_audio_path}|{transcript}|{speaker_id}")
                else:
                    print(f"Warning: Empty transcript for {wav_filename}. Skipping.")
            except Exception as e:
                print(f"Error reading or processing transcript {txt_filename}: {e}. Skipping.")
        else:
            print(f"Warning: Missing transcript file {txt_filename} for {wav_filename}. Skipping.")

    print(f"Found {len(manifest_entries)} valid audio-transcript pairs.")

    # Mezcla y divide
    random.seed(random_seed)
    random.shuffle(manifest_entries)

    split_idx = int(len(manifest_entries) * (1 - val_split_ratio))
    train_entries = manifest_entries[:split_idx]
    val_entries = manifest_entries[split_idx:]

    # Escribe los archivos manifest
    try:
        with open(train_manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_entries))
        print(f"Successfully wrote {len(train_entries)} entries to {train_manifest_path}")

        with open(val_manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_entries))
        print(f"Successfully wrote {len(val_entries)} entries to {val_manifest_path}")
    except Exception as e:
        print(f"Error writing manifest files: {e}")

    ```

---

## 2. Checklist de Calidad de los Datos

Antes de pasar a la configuración del entrenamiento, revisa rigurosamente tu dataset preparado usando esta checklist. Solucionar los problemas ahora ahorra mucho tiempo después.

| Aspecto                  | Comprobación                                                               | ¿Por qué es Importante?                                             | Acción si Falla                                                                      |
| :---------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Completitud del Audio**  | ¿Existen realmente todos los archivos `.wav` listados en los manifests?               | El entrenamiento se bloqueará si faltan archivos.                | Vuelve a ejecutar la generación de manifests; comprueba las rutas de los archivos; asegúrate de que ningún archivo se haya borrado por accidente. |
| **Coincidencia de Transcripción**    | ¿Tiene cada `.wav` un `.txt`/transcripción correspondiente y preciso?    | Los pares mal emparejados enseñan al modelo asociaciones incorrectas. | Verifica los nombres de archivo; revisa la salida del ASR; corrige las transcripciones manualmente.                     |
| **Duración del Audio**        | ¿Está la mayoría de los segmentos dentro del rango deseado (p. ej., 2-15s)? ¿Pocos valores atípicos? | Los segmentos muy cortos/largos pueden desestabilizar el entrenamiento.       | Vuelve a ejecutar la segmentación con parámetros ajustados; filtra manualmente los valores atípicos de los manifests.      |
| **Calidad del Audio**       | Escucha muestras aleatorias: ¿Poco ruido de fondo? ¿Sin música/reverberación/eco? | Basura entra, basura sale. El modelo aprende el ruido.         | Mejora el audio fuente; aplica reducción de ruido (¡con cuidado!); filtra los segmentos malos.     |
| **Consistencia del Locutor** | Para un solo locutor: ¿Es siempre la voz objetivo? ¿Sin otros locutores? | Previene la dilución o inestabilidad de la voz.                    | Revisa/filtra los segmentos manualmente; comprueba los límites de la segmentación.                         |
| **Formato y Especificaciones** | ¿Todo WAV? ¿Sampling rate **idéntico**? ¿Canales mono? ¿PCM de 16 bits?      | Las inconsistencias causan errores o un rendimiento pobre. | Vuelve a ejecutar los pasos de conversión/remuestreo (Sección 1.1). Verifica las especificaciones por lotes usando herramientas de línea de comandos como `ffprobe` o `soxi` (parte del paquete [SoX](http://sox.sourceforge.net/)). Ejemplo: `soxi -r *.wav` para comprobar las frecuencias. |
| **Niveles de Volumen**       | Escucha muestras aleatorias: ¿Son los volúmenes relativamente consistentes?          | Los cambios drásticos de volumen pueden dificultar el aprendizaje.               | Vuelve a ejecutar la normalización (Sección 1.3); comprueba los parámetros de normalización.                 |
| **Limpieza de la Transcripción** | ¿Sin marcas de tiempo, etiquetas de locutor? ¿Muletillas gestionadas de forma consistente? ¿Puntuación estándar? ¿Números/símbolos expandidos? | Asegura que el texto se mapee limpiamente a los sonidos del habla/prosodia.      | Vuelve a ejecutar los scripts de limpieza de texto; realiza una revisión y corrección manuales.                   |
| **Formato del Manifest**     | ¿Estructura `path|text|speaker_id` correcta? ¿Rutas válidas? ¿Sin líneas extra? | Los errores del parser impedirán la carga de los datos.                 | Comprueba el delimitador (`|`); valida las rutas relativas a la ubicación del script de entrenamiento; comprueba la codificación (se prefiere UTF-8). |
| **División Entrenamiento/Validación**     | ¿Son los archivos de validación realmente no vistos durante el entrenamiento? ¿Sin solapamiento?        | Los datos solapados dan puntuaciones de validación engañosas.     | Asegura una mezcla (shuffle) aleatoria antes de dividir; comprueba la lógica de división.                        |

**Consejo:** Usa herramientas como `soxi` (de SoX) o `ffprobe` para comprobar por lotes las propiedades del audio (sampling rate, canales, duración). Escribe pequeños scripts para verificar la existencia de los archivos y el formato básico del manifest.

### 2.1. Scripts Prácticos de Verificación

Aquí tienes algunos scripts prácticos para ayudarte a verificar la calidad de tu dataset:

#### Comprobar las Propiedades del Audio (Sampling Rate, Canales, Duración)

```bash
#!/bin/bash
# verify_audio.sh - Comprueba las propiedades del audio en todos los archivos WAV
# Uso: ./verify_audio.sh /path/to/audio/directory

AUDIO_DIR="$1"
echo "Checking audio files in $AUDIO_DIR..."

# Comprueba si SoX está instalado
if ! command -v soxi &> /dev/null; then
    echo "SoX not found. Please install it first (e.g., 'apt-get install sox' or 'brew install sox')."
    exit 1
fi

# Inicializa contadores y arrays
total_files=0
non_mono=0
wrong_rate=0
too_short=0
too_long=0
target_rate=22050  # Cambia esto por tu sampling rate objetivo
min_duration=1.0   # Duración mínima en segundos
max_duration=15.0  # Duración máxima en segundos

# Procesa todos los archivos WAV
find "$AUDIO_DIR" -name "*.wav" | while read -r file; do
    total_files=$((total_files + 1))
    
    # Obtiene las propiedades del audio
    channels=$(soxi -c "$file")
    rate=$(soxi -r "$file")
    duration=$(soxi -d "$file" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    
    # Comprueba las propiedades
    if [ "$channels" -ne 1 ]; then
        echo "WARNING: Non-mono file: $file (channels: $channels)"
        non_mono=$((non_mono + 1))
    fi
    
    if [ "$rate" -ne "$target_rate" ]; then
        echo "WARNING: Wrong sampling rate: $file (rate: $rate Hz, expected: $target_rate Hz)"
        wrong_rate=$((wrong_rate + 1))
    fi
    
    if (( $(echo "$duration < $min_duration" | bc -l) )); then
        echo "WARNING: File too short: $file (duration: ${duration}s, minimum: ${min_duration}s)"
        too_short=$((too_short + 1))
    fi
    
    if (( $(echo "$duration > $max_duration" | bc -l) )); then
        echo "WARNING: File too long: $file (duration: ${duration}s, maximum: ${max_duration}s)"
        too_long=$((too_long + 1))
    fi
    
    # Imprime el progreso cada 100 archivos
    if [ $((total_files % 100)) -eq 0 ]; then
        echo "Processed $total_files files..."
    fi
done

# Imprime el resumen
echo "===== SUMMARY ====="
echo "Total files checked: $total_files"
echo "Non-mono files: $non_mono"
echo "Files with wrong sampling rate: $wrong_rate"
echo "Files too short (<${min_duration}s): $too_short"
echo "Files too long (>${max_duration}s): $too_long"

if [ $((non_mono + wrong_rate + too_short + too_long)) -eq 0 ]; then
    echo "All files passed basic checks!"
else
    echo "Some issues were found. Please review the warnings above."
fi
```

#### Verificar la Integridad del Archivo Manifest

```python
#!/usr/bin/env python3
# verify_manifest.py - Comprueba que todos los archivos del manifest existen y tienen transcripciones coincidentes
# Uso: python verify_manifest.py path/to/manifest.txt

import os
import sys
from pathlib import Path

def verify_manifest(manifest_path):
    """Verifica que todos los archivos de audio y transcripciones del manifest existen y son válidos."""
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        return False
    
    print(f"Verifying manifest: {manifest_path}")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    
    # Estadísticas
    total_entries = 0
    missing_audio = 0
    empty_transcripts = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_entries += 1
            
            # Analiza la línea (asumiendo formato separado por barra vertical: audio_path|transcript|speaker_id)
            parts = line.split('|')
            if len(parts) < 2:
                print(f"Line {line_num}: Invalid format. Expected at least 'audio_path|transcript'")
                continue
            
            audio_path = parts[0]
            transcript = parts[1]
            
            # Comprueba si la ruta del audio es relativa y resuélvela
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            
            # Comprueba si el archivo de audio existe
            if not os.path.exists(audio_path):
                print(f"Line {line_num}: Audio file not found: {audio_path}")
                missing_audio += 1
            
            # Comprueba si la transcripción está vacía
            if not transcript or transcript.isspace():
                print(f"Line {line_num}: Empty transcript for {audio_path}")
                empty_transcripts += 1
    
    # Imprime el resumen
    print("\n===== SUMMARY =====")
    print(f"Total entries: {total_entries}")
    print(f"Missing audio files: {missing_audio}")
    print(f"Empty transcripts: {empty_transcripts}")
    
    if missing_audio == 0 and empty_transcripts == 0:
        print("All manifest entries are valid!")
        return True
    else:
        print("Issues found in manifest. Please fix them before proceeding.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_manifest.py path/to/manifest.txt")
        sys.exit(1)
    
    success = verify_manifest(sys.argv[1])
    sys.exit(0 if success else 1)
```

#### Visualizar Espectrogramas de Audio para Evaluar la Calidad

Este script te ayuda a inspeccionar visualmente la calidad de tus archivos de audio generando espectrogramas:

```python
#!/usr/bin/env python3
# generate_spectrograms.py - Crea espectrogramas para evaluar la calidad del audio
# Uso: python generate_spectrograms.py /path/to/audio/directory /path/to/output/directory [num_samples]

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

def generate_spectrograms(audio_dir, output_dir, num_samples=10):
    """Genera espectrogramas para una muestra aleatoria de archivos de audio."""
    # Crea el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtiene todos los archivos WAV
    wav_files = list(Path(audio_dir).glob('**/*.wav'))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return False
    
    # Toma una muestra de los archivos si hay más de los solicitados
    if len(wav_files) > num_samples:
        wav_files = random.sample(wav_files, num_samples)
    
    print(f"Generating spectrograms for {len(wav_files)} files...")
    
    for i, wav_path in enumerate(wav_files):
        try:
            # Carga el archivo de audio
            y, sr = librosa.load(wav_path, sr=None)
            
            # Crea la figura con dos subgráficos
            plt.figure(figsize=(12, 8))
            
            # Dibuja la forma de onda
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {wav_path.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Dibuja el espectrograma
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram')
            
            # Guarda la figura
            output_path = os.path.join(output_dir, f'spectrogram_{i+1}_{wav_path.stem}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            print(f"Generated: {output_path}")
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
    
    print(f"Spectrograms saved to {output_dir}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_spectrograms.py /path/to/audio/directory /path/to/output/directory [num_samples]")
        sys.exit(1)
    
    audio_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    success = generate_spectrograms(audio_dir, output_dir, num_samples)
    sys.exit(0 if success else 1)
```

Estos scripts proporcionan herramientas prácticas para verificar la calidad de tu dataset antes del entrenamiento, ayudándote a identificar y solucionar problemas en una fase temprana del proceso.

---

Una vez que tu dataset pase esta comprobación de calidad, estarás listo para proceder con la configuración del entorno de entrenamiento.

**Siguiente Paso:** [Configuración del Entrenamiento](./2_TRAINING_SETUP.md){: .btn .btn-primary} |
[Volver Arriba](#top){: .btn .btn-primary}
