# Guida 4: Inferenza - Generare il Parlato con il Tuo Modello

**Navigazione:** [README Principale]({{ site.baseurl }}/languages/it/){: .btn .btn-primary} | [Passo Precedente: Addestramento del Modello](./3_MODEL_TRAINING.md){: .btn .btn-primary} | [Passo Successivo: Packaging e Condivisione](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

Hai addestrato con successo o effettuato il fine-tuning di un modello TTS e selezionato un checkpoint promettente! Ora, usiamo quel modello per convertire nuovo testo in audio parlato, un processo chiamato **inferenza** o **sintesi**.

---

## 7. Inferenza: Sintetizzare il Parlato

Questa sezione spiega come eseguire il processo di inferenza usando il tuo modello addestrato.

### 7.1. Individuare lo Script di Inferenza e il Checkpoint Migliore

-   **Script di Inferenza:** Trova lo script Python all'interno del tuo framework TTS progettato per generare audio. I nomi comuni includono `inference.py`, `synthesize.py`, `infer.py`, `tts.py`.
-   **Checkpoint Migliore:** Identifica il percorso del checkpoint del modello (`.pth`, `.pt`, `.ckpt`) che vuoi usare. Questo è tipicamente quello salvato come `best_model.pth` (in base alla validation loss) o un altro checkpoint che hai selezionato ascoltando i sample di validazione durante l'addestramento. Sarà collocato all'interno della tua directory di output dell'addestramento (ad esempio `../checkpoints/my_yoruba_voice_run1/best_model.pth`).
-   **File di Configurazione:** Avrai quasi sempre bisogno del file di configurazione (`.yaml`, `.json`) che è stato usato *durante l'addestramento* del checkpoint che stai utilizzando. Lo script di inferenza ne ha bisogno per conoscere l'architettura del modello, i parametri audio (come il sampling rate) e altre impostazioni. Spesso, una copia della config viene salvata insieme ai checkpoint.

### 7.2. Inferenza Base di una Singola Frase

-   **Obiettivo:** Generare audio per un singolo testo fornito direttamente tramite la riga di comando.
-   **Struttura del Comando:** Gli argomenti esatti varieranno, ma un comando tipico si presenta così:

    ```bash
    # Attiva prima il tuo ambiente virtuale!
    # Comando di esempio:
    python inference.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --text "Hello, this is a test of my custom trained voice." \
      --output_wav_path ./output_sample.wav
      # Argomenti opzionali/dipendenti dal framework qui sotto:
      # --speaker_id "main_speaker"  # Necessario per modelli multi-speaker
      # --device "cuda"              # Per specificare l'uso della GPU (spesso predefinito)
    ```
-   **Argomenti Chiave:**
    *   `--config` o `-c`: Percorso al file di configurazione dell'addestramento.
    *   `--checkpoint_path` o `--model_path` o `-m`: Percorso al file di checkpoint del modello.
    *   `--text` o `-t`: La frase di input che vuoi sintetizzare. Ricorda di racchiuderla tra virgolette.
    *   `--output_wav_path` o `--out_path` o `-o`: Il percorso e il nome file desiderati per il file WAV generato.
    *   `--speaker_id` o `--spk`: **Richiesto** se hai addestrato un modello multi-speaker. Fornisci l'esatto speaker ID usato nei tuoi file manifest per la voce desiderata. Per i modelli a singolo speaker, questo potrebbe essere opzionale o ignorato.
    *   `--device`: Spesso opzionale, predefinito a `cuda` se disponibile, altrimenti `cpu`. L'inferenza è molto più veloce su GPU.

-   **Esecuzione:** Esegui il comando. Caricherà il modello, elaborerà il testo, genererà la forma d'onda audio e la salverà nel file di output specificato. Ascolta il file WAV di output per verificare la qualità.

### 7.3. Inferenza in Batch (Sintetizzare da un File)

-   **Obiettivo:** Generare audio per più frasi elencate in un file di testo, salvando ciascuna come file WAV separato.
-   **Prepara il File di Input:** Crea un file di testo semplice (ad esempio `sentences.txt`) dove ogni riga contiene una frase che vuoi sintetizzare:
    ```text
    Questa è la prima frase.
    Ecco un'altra frase da sintetizzare.
    Il modello dovrebbe gestire segni di punteggiatura diversi, come le domande?
    E anche le esclamazioni!
    ```
-   **Struttura del Comando:** Molti framework forniscono uno script separato o argomenti specifici per l'elaborazione batch.

    ```bash
    # Comando di esempio (il nome dello script e gli argomenti possono variare):
    python inference_batch.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --input_file sentences.txt \
      --output_dir ./generated_batch_audio/
      # Argomenti opzionali/dipendenti dal framework qui sotto:
      # --speaker_id "main_speaker"  # Necessario per modelli multi-speaker
      # --device "cuda"
    ```
-   **Argomenti Chiave:**
    *   `--input_file` o `--text_file`: Percorso al file di testo contenente le frasi (una per riga).
    *   `--output_dir` o `--out_dir`: Percorso alla directory dove devono essere salvati i file WAV generati. Assicurati che questa directory esista o che lo script la crei. I nomi dei file di output sono spesso basati sul numero di riga o sul testo di input stesso (ad esempio `output_0.wav`, `output_1.wav`).
    *   Gli altri argomenti (`--config`, `--checkpoint_path`, `--speaker_id`, `--device`) sono tipicamente gli stessi dell'inferenza di una singola frase.

-   **Esecuzione:** Esegui il comando. Lo script itererà su ogni riga del file di input, sintetizzerà l'audio e salverà i risultati nella directory di output specificata.

### 7.4. Inferenza di Modelli Multi-Speaker

-   Come menzionato sopra, se il tuo modello è stato addestrato su dati di più speaker, **devi** specificare quale voce di speaker vuoi usare durante l'inferenza.
-   Usa l'argomento `--speaker_id` (o equivalente), fornendo l'esatto ID che corrisponde allo speaker desiderato nei tuoi file manifest di addestramento (ad esempio `speaker0`, `mary_smith`, `yoruba_male_spk1`).
-   Se ometti lo speaker ID per un modello multi-speaker, lo script potrebbe fallire, usare per impostazione predefinita uno speaker specifico (spesso lo speaker 0) o produrre risultati mediati/confusi.

### 7.5. Controlli di Inferenza Avanzati (Dipendenti dal Framework)

-   Alcuni modelli e framework TTS avanzati offrono controlli aggiuntivi durante l'inferenza, spesso passati come argomenti da riga di comando o parametri in un'API Python:
    *   **Velocità del Parlato/Speed:** Argomenti come `--speed` o `--length_scale` potrebbero consentirti di far parlare la voce più velocemente o più lentamente (ad esempio, `1.0` è normale, `<1.0` è più veloce, `>1.0` è più lento).
    *   **Controllo del Pitch:** Meno comune, ma alcuni modelli potrebbero consentire regolazioni del pitch.
    *   **Controllo di Stile/Emozione:** Se il modello è stato addestrato con style token o capacità di audio di riferimento (come StyleTTS2 o modelli con style embedding), potresti fornire argomenti come `--style_text` o `--style_wav` per influenzare la prosodia o l'emozione dell'output.
    *   **Impostazioni del Vocoder (se applicabile):** Per i modelli più vecchi in stile Tacotron2 o altri che usano modelli vocoder separati (come HiFi-GAN, MelGAN), potrebbero esserci impostazioni relative al vocoder (ad esempio la forza del denoising).
    *   **Diffusion Models:** Per i modelli TTS basati su diffusion, potrebbero essere disponibili parametri che controllano il numero di diffusion step (scambiando qualità con velocità).
-   **Consulta la Documentazione:** Fai sempre riferimento alla documentazione del tuo specifico framework TTS o all'help dello script di inferenza (`python inference.py --help`) per vedere quali controlli sono disponibili.

### 7.6. Potenziali Problemi di Inferenza

-   **CUDA Out-of-Memory (OOM):** Anche se l'addestramento ha funzionato, frasi molto lunghe durante l'inferenza potrebbero consumare più memoria. Prova frasi più corte o verifica se il framework offre opzioni per la sintesi segmentata. L'esecuzione su CPU (`--device cpu`) usa la RAM di sistema ma è significativamente più lenta.
-   **Discrepanza Modello/Config:** Usare un checkpoint con il file di configurazione errato è un errore comune, che porta a fallimenti di caricamento o output incomprensibile. Assicurati che corrispondano alla stessa esecuzione di addestramento.
-   **Speaker ID Errato:** Fornire uno speaker ID inesistente per i modelli multi-speaker causerà errori.
-   **Problemi di Qualità (Rumore, Instabilità):** Se la qualità dell'output è scarsa, rivedi la Guida 1 (Preparazione dei Dati) e la Guida 3 (Addestramento del Modello). Potrebbe indicare problemi con la qualità dei dati, addestramento insufficiente o scelta di un checkpoint subottimale.

---

## 8. Valutazione e Deployment del Modello

### 8.1. Valutare la Qualità del Modello TTS

Sebbene i test di ascolto soggettivi siano il gold standard per la valutazione TTS, esistono anche metriche oggettive che possono aiutare a quantificare le prestazioni del tuo modello:

#### Metriche di Valutazione Oggettive

| Metrica | Cosa Misura | Strumenti/Implementazione | Interpretazione |
|:-------|:-----------------|:---------------------|:---------------|
| **MOS (Mean Opinion Score)** | Qualità percepita complessiva | Valutatori umani giudicano i sample su una scala da 1 a 5 | Più alto è meglio; standard del settore ma richiede valutatori umani |
| **PESQ (Perceptual Evaluation of Speech Quality)** | Qualità audio rispetto a un riferimento | Disponibile in Python tramite `pypesq` | Intervallo: da -0.5 a 4.5; più alto è meglio |
| **STOI (Short-Time Objective Intelligibility)** | Intelligibilità del parlato | Disponibile in Python tramite `pystoi` | Intervallo: da 0 a 1; più alto è meglio |
| **Character/Word Error Rate (CER/WER)** | Intelligibilità tramite ASR | Esegui l'ASR sul parlato sintetizzato e confronta con il testo di input | Più basso è meglio; misura se le parole sono pronunciate correttamente |
| **Mel Cepstral Distortion (MCD)** | Distanza spettrale dal riferimento | Implementazione personalizzata con librosa | Più basso è meglio; tipicamente 2-8 per i sistemi TTS |
| **F0 RMSE** | Accuratezza del pitch | Implementazione personalizzata con librosa | Più basso è meglio; misura l'accuratezza del contorno del pitch |
| **Voicing Decision Error** | Accuratezza voiced/unvoiced | Implementazione personalizzata | Più basso è meglio; misura se parlato/silenzio è collocato correttamente |

#### Approccio Pratico alla Valutazione

1. **Prepara un Test Set**: Crea un insieme di frasi di test diverse non viste durante l'addestramento
   ```
   # Esempio di test_sentences.txt
   Questa è una semplice frase dichiarativa.
   Questa è una frase interrogativa?
   Wow! Questa è una frase esclamativa!
   Questa frase contiene numeri come 123 e simboli come %.
   Questa è una frase molto più lunga che prosegue per parecchio tempo, mettendo alla prova la capacità del modello di mantenere la coerenza e una prosodia corretta su enunciati più lunghi con molteplici proposizioni e locuzioni.
   ```

2. **Genera i Sample**: Usa il tuo modello per sintetizzare il parlato per tutte le frasi di test

3. **Conduci Test di Ascolto**: Fai valutare i sample a più ascoltatori su:
   - Naturalezza (scala 1-5)
   - Qualità audio/artefatti (scala 1-5)
   - Accuratezza della pronuncia (scala 1-5)
   - Somiglianza dello speaker (scala 1-5, se si clona una voce specifica)

4. **Implementa Metriche Oggettive**: Questo snippet Python dimostra come calcolare alcune metriche di base:

   ```python
   import numpy as np
   import librosa
   from pesq import pesq
   from pystoi import stoi
   import torch
   from transformers import pipeline

   def evaluate_tts_sample(generated_audio_path, reference_audio_path=None, original_text=None):
       """Evaluate a TTS sample using various metrics."""
       results = {}
       
       # Carica l'audio generato
       y_gen, sr_gen = librosa.load(generated_audio_path, sr=None)
       
       # Statistiche audio di base
       results["duration"] = librosa.get_duration(y=y_gen, sr=sr_gen)
       results["rms_energy"] = np.sqrt(np.mean(y_gen**2))
       
       # Se l'audio di riferimento è disponibile, calcola le metriche di confronto
       if reference_audio_path:
           y_ref, sr_ref = librosa.load(reference_audio_path, sr=sr_gen)  # Allinea i sampling rate
           
           # Assicura la stessa lunghezza per il confronto
           min_len = min(len(y_gen), len(y_ref))
           y_gen_trim = y_gen[:min_len]
           y_ref_trim = y_ref[:min_len]
           
           # PESQ (richiede audio a 16kHz o 8kHz)
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
       
       # Se il testo originale è disponibile, esegui l'ASR e calcola WER/CER
       if original_text:
           try:
               # Carica il modello ASR (richiede transformers e torch)
               asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
               
               # Trascrivi l'audio generato
               transcription = asr(generated_audio_path)["text"].strip().lower()
               original_text = original_text.strip().lower()
               
               results["transcription"] = transcription
               results["original_text"] = original_text
               
               # Calcolo semplice del character error rate
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

### 8.2. Deployment dei Modelli TTS

Una volta addestrato e valutato il tuo modello, potresti volerlo distribuire per un uso pratico. Ecco alcune opzioni di deployment:

#### Considerazioni sul Deployment in Produzione

Quando si passa dalla sperimentazione al deployment in produzione, considera questi importanti fattori:

1. **Ottimizzazione del Modello**
   - **Quantization**: Riduci la precisione del modello da FP32 a FP16 o INT8 per diminuire la dimensione e aumentare la velocità di inferenza
   - **Pruning**: Rimuovi i pesi non necessari per creare modelli più piccoli e veloci
   - **Knowledge Distillation**: Addestra un modello "studente" più piccolo per imitare il tuo modello "insegnante" più grande
   - **ONNX Conversion**: Converti il tuo modello PyTorch/TensorFlow in formato ONNX per migliori prestazioni cross-platform

2. **Ottimizzazione della Latenza**
   - **Batch Processing**: Per applicazioni non in tempo reale, elabora più richieste in batch
   - **Streaming Synthesis**: Per applicazioni in tempo reale, implementa l'elaborazione chunk-by-chunk
   - **Caching**: Memorizza nella cache le frasi o le sequenze fonemiche richieste frequentemente
   - **Hardware Acceleration**: Utilizza GPU/TPU per l'elaborazione parallela o hardware specializzato come NVIDIA TensorRT

3. **Scalabilità**
   - **Containerization**: Impacchetta il tuo modello e le dipendenze in container Docker
   - **Kubernetes**: Orchestra più container per alta disponibilità e load balancing
   - **Auto-scaling**: Regola automaticamente le risorse in base alla domanda
   - **Queue Systems**: Implementa code di richieste (RabbitMQ, Kafka) per gestire i picchi di traffico

4. **Monitoraggio e Manutenzione**
   - **Performance Metrics**: Traccia latenza, throughput, tassi di errore e utilizzo delle risorse
   - **Quality Monitoring**: Campiona e valuta periodicamente la qualità dell'output
   - **A/B Testing**: Confronta diverse versioni del modello in produzione
   - **Continuous Training**: Configura pipeline per riaddestrare i modelli con nuovi dati

#### Esempio di Architettura di Deployment in Produzione

```
[Applicazioni Client] → [Load Balancer] → [API Gateway]
                                             ↓
[Validazione Richiesta] → [Rate Limiting] → [Autenticazione]
                                             ↓
[Coda Richieste] → [TTS Worker Pods (Kubernetes)] → [Cache Audio]
                         ↓                              ↑
                  [Container Modello TTS]               |
                         ↓                              |
                  [Post-Elaborazione Audio] → [Storage Audio]
```

#### Opzioni di Deployment Locale

1. **Interfaccia a Riga di Comando**: L'approccio più semplice è creare uno script che incapsula il codice di inferenza:

   ```python
   # tts_cli.py
   import argparse
   import os
   import torch
   
   # Importa qui i tuoi moduli specifici del modello
   # from your_tts_framework import load_model, synthesize_text
   
   def main():
       parser = argparse.ArgumentParser(description="Text-to-Speech CLI")
       parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
       parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
       parser.add_argument("--config", type=str, required=True, help="Path to model config")
       parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
       parser.add_argument("--speaker", type=str, default=None, help="Speaker ID for multi-speaker models")
       args = parser.parse_args()
       
       # Carica il modello (l'implementazione dipende dal tuo framework)
       model = load_model(args.model, args.config)
       
       # Sintetizza il parlato
       audio = synthesize_text(model, args.text, speaker_id=args.speaker)
       
       # Salva l'audio
       save_audio(audio, args.output)
       print(f"Audio saved to {args.output}")
   
   if __name__ == "__main__":
       main()
   ```

2. **Semplice UI Web**: Crea un'interfaccia web di base usando Flask o Gradio:

   ```python
   # app.py (esempio con Flask)
   from flask import Flask, request, send_file, render_template
   import os
   import torch
   import uuid
   
   # Importa qui i tuoi moduli specifici del modello
   # from your_tts_framework import load_model, synthesize_text
   
   app = Flask(__name__)
   
   # Carica il modello all'avvio (per un'inferenza più veloce)
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
       
       # Genera un nome file unico
       output_file = f"static/audio/{uuid.uuid4()}.wav"
       os.makedirs(os.path.dirname(output_file), exist_ok=True)
       
       # Sintetizza il parlato
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       
       # Salva l'audio
       save_audio(audio, output_file)
       
       return {'audio_path': output_file}
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000, debug=True)
   ```

3. **Interfaccia Gradio** (Ancora più semplice):

   ```python
   import gradio as gr
   import torch
   
   # Importa qui i tuoi moduli specifici del modello
   # from your_tts_framework import load_model, synthesize_text
   
   # Carica il modello
   MODEL_PATH = "path/to/best_model.pth"
   CONFIG_PATH = "path/to/config.yaml"
   model = load_model(MODEL_PATH, CONFIG_PATH)
   
   def tts_function(text, speaker_id=None):
       # Sintetizza il parlato
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       sampling_rate = 22050  # Adatta al rate del tuo modello
       return (sampling_rate, audio)
   
   # Crea l'interfaccia Gradio
   iface = gr.Interface(
       fn=tts_function,
       inputs=[
           gr.Textbox(lines=3, placeholder="Enter text to synthesize..."),
           gr.Dropdown(choices=["speaker1", "speaker2"], label="Speaker", visible=True)  # Per modelli multi-speaker
       ],
       outputs=gr.Audio(type="numpy"),
       title="Text-to-Speech Demo",
       description="Enter text and generate speech using a custom TTS model."
   )
   
   iface.launch(server_name="0.0.0.0", server_port=7860)
   ```

#### Opzioni di Deployment Cloud

Per l'uso in produzione, considera queste opzioni:

1. **Hugging Face Spaces**: Carica il tuo modello su Hugging Face e crea un'app Gradio o Streamlit
2. **REST API**: Incapsula il tuo modello in un'applicazione FastAPI o Flask e distribuiscilo su servizi cloud
3. **Serverless Functions**: Per modelli leggeri, distribuiscili come funzioni serverless (AWS Lambda, Google Cloud Functions)
4. **Docker Containers**: Impacchetta il tuo modello e le dipendenze in un container Docker per un deployment coerente

#### Ottimizzazione delle Prestazioni

Per migliorare la velocità e l'efficienza dell'inferenza:

1. **Model Quantization**: Converti i pesi del modello a una precisione inferiore (FP16 o INT8)
   ```python
   # Esempio di conversione FP16 con PyTorch
   model = model.half()  # Convert to FP16
   ```

2. **Model Pruning**: Rimuovi i pesi non necessari per creare modelli più piccoli
3. **ONNX Conversion**: Converti i modelli PyTorch in formato ONNX per un'inferenza più veloce
   ```python
   # Esempio di export ONNX
   import torch.onnx
   
   # Esporta il modello
   torch.onnx.export(model,               # model being run
                     dummy_input,         # model input (or a tuple for multiple inputs)
                     "model.onnx",        # where to save the model
                     export_params=True,  # store the trained parameter weights inside the model file
                     opset_version=11,    # the ONNX version to export the model to
                     do_constant_folding=True)  # optimization
   ```

4. **Batch Processing**: Elabora più input testuali contemporaneamente per un throughput più elevato
5. **Caching**: Memorizza nella cache gli output richiesti frequentemente per evitare la rigenerazione

Ora che puoi generare il parlato usando il tuo modello addestrato, il passo logico successivo è organizzare correttamente i file del modello per usi futuri, condivisione o deployment.

**Passo Successivo:** [Packaging e Condivisione](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 
[Torna in Cima](#top){: .btn .btn-primary}
