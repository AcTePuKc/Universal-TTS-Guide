# Guía 4: Inferencia - Generando Voz con tu Modelo

**Navegación:** [README Principal]({{ site.baseurl }}/languages/es/){: .btn .btn-primary} | [Paso Anterior: Entrenamiento del Modelo](./3_MODEL_TRAINING.md){: .btn .btn-primary} | [Siguiente Paso: Empaquetado y Compartición](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

¡Has entrenado o hecho fine-tuning de un modelo TTS con éxito y seleccionado un checkpoint prometedor! Ahora, usemos ese modelo para convertir texto nuevo en audio de voz, un proceso llamado **inferencia** o **síntesis**.

---

## 7. Inferencia: Sintetizando Voz

Esta sección explica cómo ejecutar el proceso de inferencia usando tu modelo entrenado.

### 7.1. Localiza el Script de Inferencia y el Mejor Checkpoint

-   **Script de Inferencia:** Encuentra el script de Python dentro de tu framework de TTS diseñado para generar audio. Los nombres comunes incluyen `inference.py`, `synthesize.py`, `infer.py`, `tts.py`.
-   **Mejor Checkpoint:** Identifica la ruta al checkpoint del modelo (`.pth`, `.pt`, `.ckpt`) que quieres usar. Normalmente es el que se guarda como `best_model.pth` (basado en el validation loss) u otro checkpoint que seleccionaste tras escuchar las muestras de validación durante el entrenamiento. Estará ubicado dentro de tu directorio de salida del entrenamiento (por ejemplo, `../checkpoints/my_yoruba_voice_run1/best_model.pth`).
-   **Archivo de Configuración:** Casi siempre necesitarás el archivo de configuración (`.yaml`, `.json`) que se usó *durante el entrenamiento* del checkpoint que estás usando. El script de inferencia lo necesita para conocer la arquitectura del modelo, los parámetros de audio (como el sampling rate) y otros ajustes. A menudo, se guarda una copia de la configuración junto a los checkpoints.

### 7.2. Inferencia Básica de una Sola Frase

-   **Objetivo:** Generar audio para una única pieza de texto proporcionada directamente a través de la línea de comandos.
-   **Estructura del Comando:** Los argumentos exactos variarán, pero un comando típico se ve así:

    ```bash
    # ¡Activa primero tu entorno virtual!
    # Comando de ejemplo:
    python inference.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --text "Hello, this is a test of my custom trained voice." \
      --output_wav_path ./output_sample.wav
      # Argumentos opcionales/dependientes del framework a continuación:
      # --speaker_id "main_speaker"  # Necesario para modelos multi-locutor
      # --device "cuda"              # Para especificar el uso de la GPU (a menudo por defecto)
    ```
-   **Argumentos Clave:**
    *   `--config` o `-c`: Ruta al archivo de configuración del entrenamiento.
    *   `--checkpoint_path` o `--model_path` o `-m`: Ruta al archivo de checkpoint del modelo.
    *   `--text` o `-t`: La frase de entrada que quieres sintetizar. Recuerda encerrarla entre comillas.
    *   `--output_wav_path` o `--out_path` o `-o`: La ruta y el nombre de archivo deseados para el archivo WAV generado.
    *   `--speaker_id` o `--spk`: **Requerido** si entrenaste un modelo multi-locutor. Proporciona el speaker ID exacto usado en tus archivos manifest para la voz deseada. Para modelos de un solo locutor, esto podría ser opcional o ignorarse.
    *   `--device`: A menudo opcional, por defecto es `cuda` si está disponible, de lo contrario `cpu`. La inferencia es mucho más rápida en GPU.

-   **Ejecución:** Ejecuta el comando. Cargará el modelo, procesará el texto, generará la forma de onda del audio y la guardará en el archivo de salida especificado. Escucha el archivo WAV de salida para comprobar la calidad.

### 7.3. Inferencia por Lotes (Sintetizando desde un Archivo)

-   **Objetivo:** Generar audio para múltiples frases listadas en un archivo de texto, guardando cada una como un archivo WAV separado.
-   **Prepara el Archivo de Entrada:** Crea un archivo de texto plano (por ejemplo, `sentences.txt`) donde cada línea contenga una frase que quieras sintetizar:
    ```text
    Esta es la primera frase.
    Aquí hay otra frase para sintetizar.
    El modelo debería manejar diferentes signos de puntuación, ¿como las preguntas?
    ¡Y también las exclamaciones!
    ```
-   **Estructura del Comando:** Muchos frameworks proporcionan un script separado o argumentos específicos para el procesamiento por lotes.

    ```bash
    # Comando de ejemplo (el nombre del script y los argumentos pueden variar):
    python inference_batch.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --input_file sentences.txt \
      --output_dir ./generated_batch_audio/
      # Argumentos opcionales/dependientes del framework a continuación:
      # --speaker_id "main_speaker"  # Necesario para modelos multi-locutor
      # --device "cuda"
    ```
-   **Argumentos Clave:**
    *   `--input_file` o `--text_file`: Ruta al archivo de texto que contiene las frases (una por línea).
    *   `--output_dir` o `--out_dir`: Ruta al directorio donde se deben guardar los archivos WAV generados. Asegúrate de que este directorio exista o de que el script lo cree. Los nombres de archivo de salida a menudo se basan en el número de línea o en el propio texto de entrada (por ejemplo, `output_0.wav`, `output_1.wav`).
    *   Otros argumentos (`--config`, `--checkpoint_path`, `--speaker_id`, `--device`) son normalmente los mismos que para la inferencia de una sola frase.

-   **Ejecución:** Ejecuta el comando. El script iterará por cada línea del archivo de entrada, sintetizará el audio y guardará los resultados en el directorio de salida especificado.

### 7.4. Inferencia de Modelos Multi-Locutor

-   Como se mencionó anteriormente, si tu modelo fue entrenado con datos de múltiples locutores, **debes** especificar qué voz de locutor quieres usar durante la inferencia.
-   Usa el argumento `--speaker_id` (o equivalente), proporcionando el ID exacto que corresponde al locutor deseado en tus archivos manifest de entrenamiento (por ejemplo, `speaker0`, `mary_smith`, `yoruba_male_spk1`).
-   Si omites el speaker ID para un modelo multi-locutor, el script podría fallar, usar por defecto un locutor específico (a menudo el locutor 0), o producir resultados promediados/distorsionados.

### 7.5. Controles de Inferencia Avanzados (Dependientes del Framework)

-   Algunos modelos y frameworks de TTS avanzados ofrecen controles adicionales durante la inferencia, a menudo pasados como argumentos de línea de comandos o parámetros en una API de Python:
    *   **Velocidad/Ritmo del Habla:** Argumentos como `--speed` o `--length_scale` podrían permitirte hacer que la voz hable más rápido o más lento (por ejemplo, `1.0` es normal, `<1.0` es más rápido, `>1.0` es más lento).
    *   **Control de Tono (Pitch):** Menos común, pero algunos modelos podrían permitir ajustes de tono.
    *   **Control de Estilo/Emoción:** Si el modelo fue entrenado con tokens de estilo o capacidades de audio de referencia (como StyleTTS2 o modelos con style embeddings), podrías proporcionar argumentos como `--style_text` o `--style_wav` para influir en la prosodia o la emoción de la salida.
    *   **Ajustes del Vocoder (si aplica):** Para modelos más antiguos al estilo Tacotron2 u otros que usan modelos de vocoder separados (como HiFi-GAN, MelGAN), podría haber ajustes relacionados con el vocoder (por ejemplo, la intensidad del denoising).
    *   **Modelos de Difusión:** Para los modelos TTS basados en difusión, podrían estar disponibles parámetros que controlen el número de pasos de difusión (intercambiando calidad por velocidad).
-   **Consulta la Documentación:** Siempre consulta la documentación de tu framework de TTS específico o la ayuda del script de inferencia (`python inference.py --help`) para ver qué controles están disponibles.

### 7.6. Problemas Potenciales de la Inferencia

-   **CUDA Out-of-Memory (OOM):** Aunque el entrenamiento haya funcionado, las frases muy largas durante la inferencia podrían consumir más memoria. Prueba con frases más cortas o comprueba si el framework ofrece opciones para la síntesis segmentada. Ejecutar en CPU (`--device cpu`) usa la RAM del sistema, pero es significativamente más lento.
-   **Desajuste de Modelo/Configuración:** Usar un checkpoint con el archivo de configuración incorrecto es un error común que conduce a fallos de carga o a una salida basura. Asegúrate de que correspondan a la misma ejecución de entrenamiento.
-   **Speaker ID Incorrecto:** Proporcionar un speaker ID inexistente para modelos multi-locutor causará errores.
-   **Problemas de Calidad (Ruido, Inestabilidad):** Si la calidad de la salida es pobre, revisa la Guía 1 (Preparación de Datos) y la Guía 3 (Entrenamiento del Modelo). Podría indicar problemas con la calidad de los datos, un entrenamiento insuficiente o la elección de un checkpoint subóptimo.

---

## 8. Evaluación y Despliegue del Modelo

### 8.1. Evaluando la Calidad del Modelo TTS

Si bien las pruebas de escucha subjetivas son el estándar de oro para la evaluación de TTS, también existen métricas objetivas que pueden ayudar a cuantificar el rendimiento de tu modelo:

#### Métricas de Evaluación Objetivas

| Métrica | Qué Mide | Herramientas/Implementación | Interpretación |
|:-------|:-----------------|:---------------------|:---------------|
| **MOS (Mean Opinion Score)** | Calidad percibida general | Evaluadores humanos valoran las muestras en una escala de 1 a 5 | Más alto es mejor; estándar de la industria pero requiere evaluadores humanos |
| **PESQ (Perceptual Evaluation of Speech Quality)** | Calidad del audio comparada con una referencia | Disponible en Python vía `pypesq` | Rango: -0.5 a 4.5; más alto es mejor |
| **STOI (Short-Time Objective Intelligibility)** | Inteligibilidad del habla | Disponible en Python vía `pystoi` | Rango: 0 a 1; más alto es mejor |
| **Tasa de Error de Caracteres/Palabras (CER/WER)** | Inteligibilidad mediante ASR | Ejecuta ASR sobre la voz sintetizada y compara con el texto de entrada | Más bajo es mejor; mide si las palabras se pronuncian correctamente |
| **Mel Cepstral Distortion (MCD)** | Distancia espectral respecto a la referencia | Implementación personalizada usando librosa | Más bajo es mejor; típicamente 2-8 para sistemas TTS |
| **F0 RMSE** | Precisión del tono (pitch) | Implementación personalizada usando librosa | Más bajo es mejor; mide la precisión del contorno de tono |
| **Error en la Decisión de Sonoridad (Voicing)** | Precisión sonora/sorda (voiced/unvoiced) | Implementación personalizada | Más bajo es mejor; mide si el habla/silencio se coloca correctamente |

#### Enfoque Práctico de Evaluación

1. **Prepara el Conjunto de Prueba**: Crea un conjunto de frases de prueba diversas no vistas durante el entrenamiento
   ```
   # Ejemplo de test_sentences.txt
   Esta es una simple frase declarativa.
   ¿Es esta una frase interrogativa?
   ¡Vaya! ¡Esta es una frase exclamativa!
   Esta frase contiene números como 123 y símbolos como %.
   Esta es una frase mucho más larga que continúa durante bastante tiempo, poniendo a prueba la capacidad del modelo para mantener la coherencia y una prosodia adecuada a lo largo de enunciados más largos con múltiples cláusulas y oraciones.
   ```

2. **Genera Muestras**: Usa tu modelo para sintetizar voz para todas las frases de prueba

3. **Realiza Pruebas de Escucha**: Haz que varios oyentes valoren las muestras según:
   - Naturalidad (escala de 1 a 5)
   - Calidad del audio/artefactos (escala de 1 a 5)
   - Precisión de la pronunciación (escala de 1 a 5)
   - Similitud del locutor (escala de 1 a 5, si se clona una voz específica)

4. **Implementa Métricas Objetivas**: Este fragmento de Python demuestra cómo calcular algunas métricas básicas:

   ```python
   import numpy as np
   import librosa
   from pesq import pesq
   from pystoi import stoi
   import torch
   from transformers import pipeline

   def evaluate_tts_sample(generated_audio_path, reference_audio_path=None, original_text=None):
       """Evalúa una muestra de TTS usando varias métricas."""
       results = {}
       
       # Carga el audio generado
       y_gen, sr_gen = librosa.load(generated_audio_path, sr=None)
       
       # Estadísticas básicas del audio
       results["duration"] = librosa.get_duration(y=y_gen, sr=sr_gen)
       results["rms_energy"] = np.sqrt(np.mean(y_gen**2))
       
       # Si hay audio de referencia disponible, calcula métricas de comparación
       if reference_audio_path:
           y_ref, sr_ref = librosa.load(reference_audio_path, sr=sr_gen)  # Iguala los sampling rates
           
           # Asegura la misma longitud para la comparación
           min_len = min(len(y_gen), len(y_ref))
           y_gen_trim = y_gen[:min_len]
           y_ref_trim = y_ref[:min_len]
           
           # PESQ (requiere audio de 16kHz u 8kHz)
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
       
       # Si hay texto original disponible, realiza ASR y calcula WER/CER
       if original_text:
           try:
               # Carga el modelo ASR (requiere transformers y torch)
               asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
               
               # Transcribe el audio generado
               transcription = asr(generated_audio_path)["text"].strip().lower()
               original_text = original_text.strip().lower()
               
               results["transcription"] = transcription
               results["original_text"] = original_text
               
               # Cálculo simple de la tasa de error de caracteres
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

### 8.2. Desplegando Modelos TTS

Una vez que has entrenado y evaluado tu modelo, es posible que quieras desplegarlo para un uso práctico. Aquí tienes algunas opciones de despliegue:

#### Consideraciones para el Despliegue en Producción

Al pasar de la experimentación al despliegue en producción, considera estos factores importantes:

1. **Optimización del Modelo**
   - **Cuantización (Quantization)**: reduce la precisión del modelo de FP32 a FP16 o INT8 para disminuir el tamaño y aumentar la velocidad de inferencia
   - **Poda (Pruning)**: elimina pesos innecesarios para crear modelos más pequeños y rápidos
   - **Destilación de Conocimiento (Knowledge Distillation)**: entrena un modelo "estudiante" más pequeño para imitar a tu modelo "profesor" más grande
   - **Conversión a ONNX**: convierte tu modelo de PyTorch/TensorFlow a formato ONNX para un mejor rendimiento multiplataforma

2. **Optimización de la Latencia**
   - **Procesamiento por Lotes (Batch Processing)**: para aplicaciones que no son en tiempo real, procesa múltiples solicitudes en lotes
   - **Síntesis en Streaming**: para aplicaciones en tiempo real, implementa un procesamiento segmento a segmento
   - **Caché (Caching)**: cachea las frases o secuencias de fonemas solicitadas con frecuencia
   - **Aceleración por Hardware**: utiliza GPU/TPU para el procesamiento paralelo o hardware especializado como NVIDIA TensorRT

3. **Escalabilidad**
   - **Contenerización (Containerization)**: empaqueta tu modelo y sus dependencias en contenedores Docker
   - **Kubernetes**: orquesta múltiples contenedores para alta disponibilidad y balanceo de carga
   - **Autoescalado (Auto-scaling)**: ajusta automáticamente los recursos según la demanda
   - **Sistemas de Colas (Queue Systems)**: implementa colas de solicitudes (RabbitMQ, Kafka) para gestionar picos de tráfico

4. **Monitorización y Mantenimiento**
   - **Métricas de Rendimiento**: monitoriza la latencia, el rendimiento (throughput), las tasas de error y la utilización de recursos
   - **Monitorización de la Calidad**: muestrea y evalúa periódicamente la calidad de la salida
   - **Pruebas A/B**: compara diferentes versiones del modelo en producción
   - **Entrenamiento Continuo**: configura pipelines para reentrenar los modelos con nuevos datos

#### Arquitectura de Ejemplo para Despliegue en Producción

```
[Aplicaciones Cliente] → [Balanceador de Carga] → [Pasarela de API (API Gateway)]
                                             ↓
[Validación de Solicitudes] → [Limitación de Tasa (Rate Limiting)] → [Autenticación]
                                             ↓
[Cola de Solicitudes] → [Pods de Workers TTS (Kubernetes)] → [Caché de Audio]
                         ↓                              ↑
                  [Contenedor del Modelo TTS]                 |
                         ↓                              |
                  [Postprocesamiento de Audio] → [Almacenamiento de Audio]
```

#### Opciones de Despliegue Local

1. **Interfaz de Línea de Comandos**: el enfoque más simple es crear un script que envuelva el código de inferencia:

   ```python
   # tts_cli.py
   import argparse
   import os
   import torch
   
   # Importa aquí tus módulos específicos del modelo
   # from your_tts_framework import load_model, synthesize_text
   
   def main():
       parser = argparse.ArgumentParser(description="Text-to-Speech CLI")
       parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
       parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
       parser.add_argument("--config", type=str, required=True, help="Path to model config")
       parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
       parser.add_argument("--speaker", type=str, default=None, help="Speaker ID for multi-speaker models")
       args = parser.parse_args()
       
       # Carga el modelo (la implementación depende de tu framework)
       model = load_model(args.model, args.config)
       
       # Sintetiza la voz
       audio = synthesize_text(model, args.text, speaker_id=args.speaker)
       
       # Guarda el audio
       save_audio(audio, args.output)
       print(f"Audio saved to {args.output}")
   
   if __name__ == "__main__":
       main()
   ```

2. **Interfaz Web Simple**: crea una interfaz web básica usando Flask o Gradio:

   ```python
   # app.py (ejemplo con Flask)
   from flask import Flask, request, send_file, render_template
   import os
   import torch
   import uuid
   
   # Importa aquí tus módulos específicos del modelo
   # from your_tts_framework import load_model, synthesize_text
   
   app = Flask(__name__)
   
   # Carga el modelo al inicio (para una inferencia más rápida)
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
       
       # Genera un nombre de archivo único
       output_file = f"static/audio/{uuid.uuid4()}.wav"
       os.makedirs(os.path.dirname(output_file), exist_ok=True)
       
       # Sintetiza la voz
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       
       # Guarda el audio
       save_audio(audio, output_file)
       
       return {'audio_path': output_file}
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000, debug=True)
   ```

3. **Interfaz con Gradio** (Aún más simple):

   ```python
   import gradio as gr
   import torch
   
   # Importa aquí tus módulos específicos del modelo
   # from your_tts_framework import load_model, synthesize_text
   
   # Carga el modelo
   MODEL_PATH = "path/to/best_model.pth"
   CONFIG_PATH = "path/to/config.yaml"
   model = load_model(MODEL_PATH, CONFIG_PATH)
   
   def tts_function(text, speaker_id=None):
       # Sintetiza la voz
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       sampling_rate = 22050  # Ajusta a la frecuencia de tu modelo
       return (sampling_rate, audio)
   
   # Crea la interfaz de Gradio
   iface = gr.Interface(
       fn=tts_function,
       inputs=[
           gr.Textbox(lines=3, placeholder="Enter text to synthesize..."),
           gr.Dropdown(choices=["speaker1", "speaker2"], label="Speaker", visible=True)  # Para modelos multi-locutor
       ],
       outputs=gr.Audio(type="numpy"),
       title="Text-to-Speech Demo",
       description="Enter text and generate speech using a custom TTS model."
   )
   
   iface.launch(server_name="0.0.0.0", server_port=7860)
   ```

#### Opciones de Despliegue en la Nube

Para uso en producción, considera estas opciones:

1. **Hugging Face Spaces**: sube tu modelo a Hugging Face y crea una app de Gradio o Streamlit
2. **API REST**: envuelve tu modelo en una aplicación FastAPI o Flask y despliégala en servicios en la nube
3. **Funciones Serverless**: para modelos ligeros, despliégalos como funciones serverless (AWS Lambda, Google Cloud Functions)
4. **Contenedores Docker**: empaqueta tu modelo y sus dependencias en un contenedor Docker para un despliegue consistente

#### Optimización del Rendimiento

Para mejorar la velocidad y eficiencia de la inferencia:

1. **Cuantización del Modelo (Quantization)**: convierte los pesos del modelo a menor precisión (FP16 o INT8)
   ```python
   # Ejemplo de conversión a FP16 con PyTorch
   model = model.half()  # Convierte a FP16
   ```

2. **Poda del Modelo (Pruning)**: elimina pesos innecesarios para crear modelos más pequeños
3. **Conversión a ONNX**: convierte los modelos de PyTorch a formato ONNX para una inferencia más rápida
   ```python
   # Ejemplo de exportación a ONNX
   import torch.onnx
   
   # Exporta el modelo
   torch.onnx.export(model,               # modelo que se está ejecutando
                     dummy_input,         # entrada del modelo (o una tupla para múltiples entradas)
                     "model.onnx",        # dónde guardar el modelo
                     export_params=True,  # almacena los pesos de los parámetros entrenados dentro del archivo del modelo
                     opset_version=11,    # la versión de ONNX a la que exportar el modelo
                     do_constant_folding=True)  # optimización
   ```

4. **Procesamiento por Lotes (Batch Processing)**: procesa múltiples entradas de texto a la vez para un mayor rendimiento (throughput)
5. **Caché (Caching)**: cachea las salidas solicitadas con frecuencia para evitar regenerarlas

Ahora que puedes generar voz usando tu modelo entrenado, el siguiente paso lógico es organizar correctamente los archivos de tu modelo para su uso futuro, compartición o despliegue.

**Siguiente Paso:** [Empaquetado y Compartición](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 
[Volver Arriba](#top){: .btn .btn-primary}
