# Guida 1: Preparazione dei Dati per l'Addestramento TTS

**Navigazione:** [README Principale]({{ site.baseurl }}/languages/it/){: .btn .btn-primary} | [Passo Successivo: Configurazione dell'Addestramento](./2_TRAINING_SETUP.md){: .btn .btn-primary} | 

Questa guida copre la prima fase critica di qualsiasi progetto TTS: la preparazione di dati audio e testuali di alta qualità e formattati correttamente. La qualità del tuo dataset influisce direttamente sulla qualità del tuo modello TTS finale.

---

## 1. Passaggi per la Preparazione del Dataset

Segui questi passaggi in modo sistematico per trasformare l'audio grezzo in un dataset pronto per l'addestramento.

### 1.1. Acquisizione Audio ed Elaborazione Iniziale

-   **Raccogli l'Audio:** Raccogli i tuoi file audio grezzi (i formati comuni includono WAV, MP3, FLAC, OGG, M4A). Assicurati di avere i diritti per utilizzare questo audio.
-   **Converti in WAV:** La maggior parte dei framework TTS si aspetta il formato WAV. Usa strumenti come `ffmpeg` o librerie audio (`pydub`, `soundfile`) per convertire il tuo audio. Punta a una codifica WAV standard come PCM 16-bit.
    ```bash
    # Esempio con ffmpeg per convertire MP3 in WAV
    ffmpeg -i input_audio.mp3 output_audio.wav
    ```
-   **Standardizza i Canali (Mono):** I modelli TTS tipicamente si addestrano su audio a singolo canale (mono). Converti le tracce stereo in mono.
    ```bash
    # Esempio con ffmpeg per convertire WAV stereo in WAV mono
    ffmpeg -i stereo_input.wav -ac 1 mono_output.wav
    ```
    *   `-ac 1`: Imposta il numero di canali audio a 1.
-   **Ricampiona l'Audio:** Assicurati che tutti i file audio abbiano **esattamente lo stesso sampling rate**. Scegli il tuo rate target in base agli obiettivi del progetto e alla compatibilità del framework (ad esempio 16000 Hz, 22050 Hz, 48000 Hz). 22050 Hz è comune per molti modelli.
    ```bash
    # Esempio con ffmpeg per ricampionare a 22050 Hz
    ffmpeg -i input.wav -ar 22050 resampled_output.wav
    ```
    *   `-ar 22050`: Imposta il sampling rate audio (campioni al secondo).

### 1.2 Pulizia Audio Avanzata (Rimozione di Rumore/Musica) - *Opzionale ma Consigliata*

-   **Obiettivo:** Rimuovere suoni di sottofondo indesiderati come rumore (ronzio, fruscio, ventole), musica, riverbero o altre voci interferenti dal tuo audio sorgente, isolando il più possibile la voce dello speaker target. Questo passaggio è cruciale se il tuo audio sorgente non è di qualità da studio.
-   **Perché?** I modelli TTS apprendono dall'audio che ricevono. Se l'audio contiene rumore di fondo o musica, la voce TTS risultante probabilmente erediterà queste caratteristiche, suonando rumorosa o "ovattata". Un audio più pulito porta a una voce TTS più pulita.

-   **Strumenti e Tecniche:**
    *   **Strumenti di Separazione delle Sorgenti con IA (Consigliati per Musica/Voce):** Questi strumenti usano modelli IA per separare l'audio in diversi stem (voce, musica, batteria, basso, altro).
        *   **[Ultimate Vocal Remover (UVR)](https://ultimatevocalremover.com/)**: Un'applicazione GUI popolare, gratuita e open-source che fornisce accesso a vari modelli di separazione IA allo stato dell'arte. È eccellente per rimuovere la musica di sottofondo o isolare i dialoghi.
            *   **Modelli (come quelli citati):** UVR ti consente di usare diversi modelli IA. `MDX-Inst-HQ3` è uno di questi modelli, spesso valido nel separare la voce dagli strumenti (da cui "Inst"). Altri modelli MDX, i modelli Demucs (come `htdemucs`) e potenzialmente modelli come Mel-Roformer (se integrato o disponibile standalone) sono progettati per compiti simili, ciascuno con punti di forza e debolezza leggermente diversi. La sperimentazione è fondamentale. Scegli modelli focalizzati sull'**isolamento vocale**.
        *   **Altri Strumenti:** Servizi online (ad esempio Lalal.ai) o altri software standalone potrebbero usare modelli sottostanti simili (spesso varianti di Demucs o Spleeter).
    *   **Strumenti Tradizionali di Riduzione del Rumore:** Spesso presenti nelle Digital Audio Workstation (DAW) o negli editor audio.
        *   **[Audacity](https://www.audacityteam.org/):** Contiene effetti di riduzione del rumore integrati (richiede di campionare un profilo di rumore). Può essere efficace per il rumore di fondo costante (come fruscio o ronzio).
        *   **Plugin Commerciali (ad esempio Izotope RX, Waves Clarity):** Offrono strumenti più sofisticati basati su IA per rumore, riverbero e isolamento vocale, ma a pagamento.
    *   **Editing Spettrale:** Rimozione manuale di suoni indesiderati in un editor spettrale (come Adobe Audition, Izotope RX, Acon Digital Acoustica). Potente ma molto dispendioso in termini di tempo.

-   **Considerazioni sul Flusso di Lavoro:**
    *   **Quando Applicarla:** Generalmente è consigliato applicare la pulizia ai tuoi **file audio più lunghi *prima* del chunking (Passo 1.3 qui sotto)**. Questo consente ai modelli IA di lavorare con più contesto e può essere più efficiente rispetto all'elaborazione di migliaia di piccoli chunk. Tuttavia, se la pulizia introduce troppi artefatti sui file lunghi, potresti provare a pulire i singoli chunk problematici in seguito.
    *   **Processo:**
        1.  Carica il tuo file WAV standardizzato (dal Passo 1.1) nello strumento scelto (ad esempio UVR).
        2.  Seleziona un modello di isolamento vocale appropriato (ad esempio un modello vocale MDX o Demucs).
        3.  Elabora l'audio per generare una traccia "solo voce".
        4.  **Ascolta Attentamente:** Valuta criticamente la traccia vocale separata. Controlla la presenza di:
            *   **Artefatti:** La separazione IA può talvolta introdurre suoni "acquosi", glitch o parti della voce erroneamente rimosse.
            *   **Rumore/Musica Residui:** Quanto efficacemente è stato rimosso il suono indesiderato?
        5.  **Itera:** Potresti dover provare modelli diversi, regolare le impostazioni all'interno dello strumento o persino applicare un secondo passaggio di riduzione del rumore (ad esempio usando la riduzione del rumore di Audacity sulla voce separata dall'IA) per ottenere i migliori risultati.
    *   **Salva l'Output:** Salva la traccia vocale pulita come nuovo file WAV (ad esempio `original_file_cleaned.wav`). Usa questi file puliti come input per il passo *successivo* (Chunking).

-   **Avvertenze:**
    *   **Sono Possibili Artefatti:** Una pulizia aggressiva può degradare la naturalezza della voce target. Punta a un equilibrio tra la rimozione del rumore e la conservazione della qualità vocale.
    *   **Costo Computazionale:** I modelli di separazione IA possono essere computazionalmente intensivi e possono richiedere tempo significativo, specialmente su file audio lunghi e senza una GPU potente.


### 1.3. Chunking Audio (Suddivisione in Segmenti)

-   **Obiettivo:** Suddividere i file audio lunghi (come i capitoli di un audiolibro o gli episodi di un podcast) in segmenti più corti e gestibili. La lunghezza ideale di un segmento è tipicamente compresa tra **2 e 15 secondi**.
-   **Perché Suddividere in Chunk?**
    *   Allinea la durata dell'audio con le lunghezze tipiche delle frasi.
    *   Rende fattibile la trascrizione (trascrivere file di ore è difficile).
    *   Aiuta a gestire la memoria durante l'addestramento.
    *   Consente di filtrare i segmenti non adatti (ad esempio puro silenzio, rumore, musica).
-   **Metodo:** Usa strumenti che rilevano il silenzio per suddividere l'audio. `pydub` è una popolare libreria Python per questo.

    ```python
    # Esempio con pydub per la suddivisione basata sul silenzio
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import os

    input_file = "resampled_mono_audio.wav" # Usa l'output del passo 1.1
    output_dir = "audio_chunks"             # Crea questa directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading audio file: {input_file}")
    sound = AudioSegment.from_wav(input_file)
    print("Audio loaded. Splitting based on silence...")

    chunks = split_on_silence(
        sound,
        min_silence_len=500,    # Durata minima del silenzio in millisecondi per attivare una suddivisione. Regola secondo necessità.
        silence_thresh=-40,     # Soglia di silenzio in dBFS (decibel relativi al fondo scala). Valori più bassi (es. -50) rilevano silenzi più tenui. Regola in base al rumore di fondo del tuo audio.
        keep_silence=200        # Opzionale: Quantità di silenzio (in ms) da lasciare all'inizio/fine di ogni chunk. Aiuta a evitare tagli bruschi.
    )

    print(f"Found {len(chunks)} potential chunks before duration filtering.")

    # --- Filtraggio ed Esportazione ---
    min_duration_sec = 2.0  # Lunghezza minima del chunk in secondi
    max_duration_sec = 15.0 # Lunghezza massima del chunk in secondi
    target_sr = 22050       # Assicura che i chunk mantengano il sampling rate corretto (pydub di solito lo gestisce)

    exported_count = 0
    for i, chunk in enumerate(chunks):
        duration_sec = len(chunk) / 1000.0
        if min_duration_sec <= duration_sec <= max_duration_sec:
            # Assicura che il chunk usi il sampling rate target se necessario (pydub cerca di preservarlo)
            # chunk = chunk.set_frame_rate(target_sr) # Di solito non necessario se la sorgente era campionata correttamente
            
            chunk_filename = f"segment_{exported_count:05d}.wav" # Usa il padding per facilitare l'ordinamento
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            print(f"Exporting chunk {i} ({duration_sec:.2f}s) to {chunk_path}")
            chunk.export(chunk_path, format="wav")
            exported_count += 1
        else:
             print(f"Skipping chunk {i} due to duration: {duration_sec:.2f}s")


    print(f"\nExported {exported_count} chunks meeting duration criteria ({min_duration_sec}-{max_duration_sec}s) to '{output_dir}'.")
    ```
-   **Revisione:** Ascolta un campione dei chunk generati. Le suddivisioni sono logiche? Il parlato viene tagliato? Regola `min_silence_len` e `silence_thresh` e riesegui se necessario. La suddivisione manuale o il perfezionamento delle suddivisioni in un editor audio (come Audacity) potrebbe essere necessario per audio complessi.

### 1.4. Normalizzazione del Volume

-   **Obiettivo:** Assicurare che tutti i chunk audio abbiano un livello di volume coerente. Questo evita che i segmenti silenziosi o forti influenzino in modo sproporzionato l'addestramento.
-   **Metodi:**
    *   **Peak Normalization:** Regola l'audio in modo che il punto più forte raggiunga un livello specifico (ad esempio -3.0 dBFS). Semplice, ma non garantisce una loudness *percepita* coerente.
    *   **Loudness Normalization (LUFS):** Regola l'audio per raggiungere un livello target di loudness percepita (ad esempio -23 LUFS è comune per la trasmissione). Generalmente preferita poiché riflette meglio l'udito umano. Richiede librerie come `pyloudnorm`.
-   **Applica in Modo Coerente:** Applica il metodo di normalizzazione scelto a *tutti* i chunk creati nel passaggio precedente. Salva i file normalizzati in una **nuova directory** (ad esempio `normalized_chunks`) per mantenere intatti gli originali.

    ```python
    # Esempio con pydub per la PEAK normalization
    from pydub import AudioSegment
    import os
    import glob

    input_chunk_dir = "audio_chunks"
    output_norm_dir = "normalized_chunks"
    os.makedirs(output_norm_dir, exist_ok=True)
    
    target_dBFS = -3.0 # Ampiezza di picco target

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
            # Applica il gain solo se il suono non è silenzioso (dBFS non è -inf)
            if sound.dBFS > -float('inf'):
              normalized_sound = match_target_amplitude(sound, target_dBFS)
              normalized_sound.export(output_path, format="wav")
            else:
              print(f"Skipping silent file: {filename}")
              # Opzionalmente copia i file silenziosi o gestiscili secondo necessità
              # shutil.copy(wav_file, output_path) 
            
            if (i + 1) % 50 == 0: # Stampa l'avanzamento
                 print(f"Processed {i+1}/{len(wav_files)} files...")
                 
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nNormalization complete. Normalized files saved in '{output_norm_dir}'.")
    ```
    *   **Nota:** Per la normalizzazione LUFS, useresti una libreria come `pyloudnorm`, iterando sui file in modo simile.

### 1.5. Trascrizione: Creazione delle Coppie Testuali

-   **Obiettivo:** Ottenere una trascrizione testuale accurata per *ogni singolo chunk audio normalizzato*. Il testo deve rappresentare *esattamente* ciò che viene pronunciato nell'audio.
-   **Metodi:**
    *   **Automatic Speech Recognition (ASR):** Ideale per dataset di grandi dimensioni. Usa modelli ASR di alta qualità.
    *   **[OpenAI Whisper](https://github.com/openai/whisper):** Eccellente opzione multilingue e open-source. Si esegue localmente (GPU consigliata) o tramite API. *Nota: Sebbene sia potente per l'accuratezza delle parole, la punteggiatura e la maiuscolazione di Whisper potrebbero richiedere un'attenta revisione e correzione durante la fase di pulizia.* Vari modelli Whisper su cui la community ha effettuato il fine-tuning (spesso disponibili su Hugging Face) potrebbero offrire miglioramenti.
    *   **[Google Gemini Models](https://ai.google.dev/) (ad esempio tramite API o AI Studio):** Modelli come Gemini Pro o Flash possono eseguire la trascrizione audio. Spesso richiedono che l'audio sia in formati specifici e possono dare i migliori risultati su segmenti più corti (allineandosi bene con il passo di pre-chunking). Verifica le offerte API attuali e i potenziali piani gratuiti.
    *   **Servizi Cloud:** Google Cloud Speech-to-Text, AWS Transcribe, Azure Speech Service offrono API robuste, spesso con prezzi pay-as-you-go e potenziali piani gratuiti iniziali.
    *   **Altri Modelli:** Esplora gli [Hugging Face Models](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) per altri modelli ASR open-source o su cui è stato effettuato il fine-tuning specifici per la tua lingua.
    *   **Trascrizione Manuale:** La più accurata ma molto dispendiosa in termini di tempo. Adatta per dataset piccoli e di alto valore o per *correggere gli output ASR*.
    *   **Trascrizioni Esistenti:** Se il tuo audio sorgente è accompagnato da trascrizioni allineate (ad esempio alcuni audiolibri, archivi di trasmissioni), potresti aver bisogno di script per analizzarle e allinearle con i tuoi chunk.
-   **Formato di Output:** Crea un file `.txt` per ogni corrispondente file `.wav` nella tua directory `normalized_chunks`. I nomi dei file devono corrispondere esattamente (ad esempio, `normalized_chunks/segment_00001.wav` necessita di `transcripts/segment_00001.txt`).
-   **Pulizia e Normalizzazione del Testo:** **Questo è cruciale!**
    *   **Rimuovi il Non-Parlato:** Elimina i timestamp (come `[00:01:05]`), le etichette degli speaker ("SPEAKER A:", "John Doe:"), i tag di eventi sonori (`[laughter]`, `[music]`), i commenti di trascrizione.
    *   **Gestisci le Parole Riempitive:** Decidi se mantenere o rimuovere i comuni riempitivi ("uh", "um", "ah"). Mantenerli potrebbe rendere il TTS più naturale ma può anche introdurre esitazioni indesiderate. Rimuoverli porta a un parlato più pulito e diretto. La coerenza è fondamentale.
    *   **Punteggiatura:** Assicura una punteggiatura coerente e appropriata. Virgole, punti e punti interrogativi aiutano il modello ad apprendere la prosodia. Evita la punteggiatura eccessiva o non standard.
    *   **Numeri, Acronimi, Simboli:** Espandili in parole (ad esempio, "101" -> "centouno", "USA" -> "U S A" o "Stati Uniti d'America", "%" -> "percento"). Il modo in cui li espandi dipende da come vuoi che il TTS li pronunci. Crea un dizionario/insieme di regole di normalizzazione se necessario.
    *   **Maiuscole/Minuscole:** Di solito converti il testo in un formato di maiuscole/minuscole coerente (ad esempio minuscolo) a meno che il tuo framework/tokenizer TTS non gestisca la maiuscolazione in modo appropriato. Controlla la documentazione del framework.
    *   **Caratteri Speciali:** Rimuovi o sostituisci i caratteri che potrebbero confondere il tokenizer (ad esempio emoji, caratteri di controllo).

    ```
    # Esempio di struttura:
    my_tts_dataset/
    ├── normalized_chunks/
    │   ├── segment_00001.wav
    │   ├── segment_00002.wav
    │   └── ...
    └── transcripts/
        ├── segment_00001.txt  # Contiene "Hello world."
        ├── segment_00002.txt  # Contiene "This is a test sentence."
        └── ...
    ```

### 1.6. Strutturazione dei Dati e Creazione del File Manifest

-   **Obiettivo:** Creare file indice (manifest) che indicano allo script di addestramento TTS dove trovare i file audio e le loro corrispondenti trascrizioni.
-   **Formato del Manifest:** Il formato più comune è un file di testo semplice dove ogni riga rappresenta una coppia audio-testo, separata da un delimitatore (di solito una pipe `|`).
    ```
    path/to/audio_chunk.wav|Il testo di trascrizione corrispondente|speaker_id
    ```
    *   `path/to/audio_chunk.wav`: Percorso relativo al file audio normalizzato a partire dalla directory in cui verrà eseguito lo script di addestramento.
    *   `Il testo di trascrizione corrispondente`: Il testo pulito e normalizzato dal file `.txt`.
    *   `speaker_id`: Un identificatore per lo speaker (ad esempio `speaker0`, `mary_smith`). Per i dataset a singolo speaker, usa lo stesso ID per tutte le righe. Per i dataset multi-speaker, usa ID unici per ogni speaker distinto.
-   **Suddivisione dei Dati (Train/Validation):** Dividi i tuoi dati in un set di addestramento (usato per aggiornare i pesi del modello) e un set di validazione (usato per monitorare le prestazioni su dati non visti e prevenire l'overfitting). Una suddivisione comune è 90-98% per l'addestramento e 2-10% per la validazione. **In modo cruciale, assicurati che, se possibile, i segmenti provenienti dalla *stessa registrazione lunga originale* non finiscano sia nel set di train che in quello di validation, per evitare il data leakage.** Se suddividi casualmente, mescola prima.
-   **Script per Generare i Manifest:**

    ```python
    import os
    import random

    # --- Configurazione ---
    dataset_name = "my_tts_dataset"
    normalized_audio_dir = os.path.join(dataset_name, "normalized_chunks")
    transcripts_dir = os.path.join(dataset_name, "transcripts")
    output_dir = dataset_name # Dove verranno salvati i file manifest

    train_manifest_path = os.path.join(output_dir, "train_list.txt")
    val_manifest_path = os.path.join(output_dir, "val_list.txt")

    speaker_id = "main_speaker" # Usa un ID coerente per i dataset a singolo speaker
                                # Per multi-speaker, determina l'ID in base al nome del file o alla sorgente
    val_split_ratio = 0.05    # 5% per il set di validation
    random_seed = 42          # Per suddivisioni riproducibili
    # ---------------------

    manifest_entries = []
    print("Reading audio and transcript files...")

    # Itera sui file audio normalizzati
    wav_files = sorted([f for f in os.listdir(normalized_audio_dir) if f.endswith(".wav")])

    for wav_filename in wav_files:
        base_filename = os.path.splitext(wav_filename)[0]
        txt_filename = base_filename + ".txt"
        
        audio_path = os.path.join(normalized_audio_dir, wav_filename)
        # Usa os.path.relpath se il tuo script di addestramento viene eseguito da una root diversa
        # relative_audio_path = os.path.relpath(audio_path, start=training_script_dir) 
        relative_audio_path = audio_path # Si assume che lo script venga eseguito dalla root che contiene 'my_tts_dataset'

        transcript_path = os.path.join(transcripts_dir, txt_filename)

        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                # Pulizia di base: rimuovi i caratteri pipe, elimina gli spazi extra
                transcript = transcript.replace('|', ' ').strip()
                transcript = ' '.join(transcript.split()) # Normalizza gli spazi bianchi

                if transcript: # Assicura che la trascrizione non sia vuota dopo la pulizia
                    manifest_entries.append(f"{relative_audio_path}|{transcript}|{speaker_id}")
                else:
                    print(f"Warning: Empty transcript for {wav_filename}. Skipping.")
            except Exception as e:
                print(f"Error reading or processing transcript {txt_filename}: {e}. Skipping.")
        else:
            print(f"Warning: Missing transcript file {txt_filename} for {wav_filename}. Skipping.")

    print(f"Found {len(manifest_entries)} valid audio-transcript pairs.")

    # Mescola e suddividi
    random.seed(random_seed)
    random.shuffle(manifest_entries)

    split_idx = int(len(manifest_entries) * (1 - val_split_ratio))
    train_entries = manifest_entries[:split_idx]
    val_entries = manifest_entries[split_idx:]

    # Scrivi i file manifest
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

## 2. Checklist sulla Qualità dei Dati

Prima di passare alla configurazione dell'addestramento, rivedi rigorosamente il tuo dataset preparato usando questa checklist. Correggere i problemi ora fa risparmiare molto tempo in seguito.

| Aspetto                  | Verifica                                                               | Perché è Importante?                                             | Azione in caso di Fallimento                                                                      |
| :---------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Completezza Audio**  | Tutti i file `.wav` elencati nei manifest esistono effettivamente?               | L'addestramento andrà in crash se mancano file.                | Rigenera i manifest; controlla i percorsi dei file; assicurati che nessun file sia stato eliminato per errore. |
| **Corrispondenza Trascrizione**    | Ogni `.wav` ha un `.txt`/trascrizione corrispondente e accurato?    | Le coppie non corrispondenti insegnano al modello associazioni errate. | Verifica i nomi dei file; rivedi l'output ASR; correggi manualmente le trascrizioni.                     |
| **Lunghezza Audio**        | La maggior parte dei segmenti rientra nell'intervallo desiderato (es. 2-15s)? Pochi valori anomali? | Segmenti molto corti/lunghi possono destabilizzare l'addestramento.       | Riesegui il chunking con parametri regolati; filtra manualmente i valori anomali dai manifest.      |
| **Qualità Audio**       | Ascolta campioni casuali: Basso rumore di fondo? Niente musica/riverbero/eco? | Garbage In, Garbage Out. Il modello apprende il rumore.         | Migliora l'audio sorgente; applica la riduzione del rumore (con attenzione!); filtra i segmenti scadenti.     |
| **Coerenza Speaker** | Per il singolo speaker: È sempre la voce target? Nessun altro speaker? | Previene la diluizione o l'instabilità della voce.                    | Rivedi/filtra manualmente i segmenti; controlla i confini del chunking.                         |
| **Formato e Specifiche** | Tutti WAV? Sampling rate **identico**? Canali mono? PCM 16-bit?      | Le incoerenze causano errori o prestazioni scarse. | Riesegui i passaggi di conversione/ricampionamento (Sezione 1.1). Verifica in batch le specifiche usando strumenti da riga di comando come `ffprobe` o `soxi` (parte del pacchetto [SoX](http://sox.sourceforge.net/)). Esempio: `soxi -r *.wav` per controllare i rate. |
| **Livelli di Volume**       | Ascolta campioni casuali: I volumi sono relativamente coerenti?          | Variazioni di volume drastiche possono ostacolare l'apprendimento.               | Riesegui la normalizzazione (Sezione 1.3); controlla i parametri di normalizzazione.                 |
| **Pulizia della Trascrizione** | Nessun timestamp, etichette speaker? Riempitivi gestiti in modo coerente? Punteggiatura standard? Numeri/simboli espansi? | Assicura che il testo si mappi in modo pulito ai suoni del parlato/prosodia.      | Riesegui gli script di pulizia del testo; esegui revisione e correzione manuali.                   |
| **Formato del Manifest**     | Struttura `path|text|speaker_id` corretta? Percorsi validi? Nessuna riga extra? | Gli errori del parser impediranno il caricamento dei dati.                 | Controlla il delimitatore (`|`); valida i percorsi relativi alla posizione dello script di addestramento; controlla la codifica (UTF-8 preferibile). |
| **Suddivisione Train/Val**     | I file di validazione sono davvero non visti durante l'addestramento? Nessuna sovrapposizione?        | Dati sovrapposti danno punteggi di validazione fuorvianti.     | Assicura il mescolamento casuale prima della suddivisione; controlla la logica di suddivisione.                        |

**Suggerimento:** Usa strumenti come `soxi` (da SoX) o `ffprobe` per controllare in batch le proprietà audio (sampling rate, canali, durata). Scrivi piccoli script per verificare l'esistenza dei file e la formattazione di base del manifest.

### 2.1. Script Pratici di Verifica

Ecco alcuni script pratici per aiutarti a verificare la qualità del tuo dataset:

#### Controllare le Proprietà Audio (Sampling Rate, Canali, Durata)

```bash
#!/bin/bash
# verify_audio.sh - Controlla le proprietà audio di tutti i file WAV
# Uso: ./verify_audio.sh /path/to/audio/directory

AUDIO_DIR="$1"
echo "Checking audio files in $AUDIO_DIR..."

# Controlla se SoX è installato
if ! command -v soxi &> /dev/null; then
    echo "SoX not found. Please install it first (e.g., 'apt-get install sox' or 'brew install sox')."
    exit 1
fi

# Inizializza contatori e array
total_files=0
non_mono=0
wrong_rate=0
too_short=0
too_long=0
target_rate=22050  # Cambia questo con il tuo sampling rate target
min_duration=1.0   # Durata minima in secondi
max_duration=15.0  # Durata massima in secondi

# Elabora tutti i file WAV
find "$AUDIO_DIR" -name "*.wav" | while read -r file; do
    total_files=$((total_files + 1))
    
    # Ottieni le proprietà audio
    channels=$(soxi -c "$file")
    rate=$(soxi -r "$file")
    duration=$(soxi -d "$file" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    
    # Controlla le proprietà
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
    
    # Stampa l'avanzamento ogni 100 file
    if [ $((total_files % 100)) -eq 0 ]; then
        echo "Processed $total_files files..."
    fi
done

# Stampa il riepilogo
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

#### Verificare l'Integrità del File Manifest

```python
#!/usr/bin/env python3
# verify_manifest.py - Controlla che tutti i file nel manifest esistano e abbiano trascrizioni corrispondenti
# Uso: python verify_manifest.py path/to/manifest.txt

import os
import sys
from pathlib import Path

def verify_manifest(manifest_path):
    """Verify that all audio files and transcripts in the manifest exist and are valid."""
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        return False
    
    print(f"Verifying manifest: {manifest_path}")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    
    # Statistiche
    total_entries = 0
    missing_audio = 0
    empty_transcripts = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_entries += 1
            
            # Analizza la riga (assumendo il formato separato da pipe: audio_path|transcript|speaker_id)
            parts = line.split('|')
            if len(parts) < 2:
                print(f"Line {line_num}: Invalid format. Expected at least 'audio_path|transcript'")
                continue
            
            audio_path = parts[0]
            transcript = parts[1]
            
            # Controlla se il percorso audio è relativo e risolvilo
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            
            # Controlla se il file audio esiste
            if not os.path.exists(audio_path):
                print(f"Line {line_num}: Audio file not found: {audio_path}")
                missing_audio += 1
            
            # Controlla se la trascrizione è vuota
            if not transcript or transcript.isspace():
                print(f"Line {line_num}: Empty transcript for {audio_path}")
                empty_transcripts += 1
    
    # Stampa il riepilogo
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

#### Visualizzare gli Spettrogrammi Audio per la Valutazione della Qualità

Questo script ti aiuta a ispezionare visivamente la qualità dei tuoi file audio generando spettrogrammi:

```python
#!/usr/bin/env python3
# generate_spectrograms.py - Crea spettrogrammi per la valutazione della qualità audio
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
    """Generate spectrograms for a random sample of audio files."""
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Ottieni tutti i file WAV
    wav_files = list(Path(audio_dir).glob('**/*.wav'))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return False
    
    # Campiona i file se ce ne sono più di quelli richiesti
    if len(wav_files) > num_samples:
        wav_files = random.sample(wav_files, num_samples)
    
    print(f"Generating spectrograms for {len(wav_files)} files...")
    
    for i, wav_path in enumerate(wav_files):
        try:
            # Carica il file audio
            y, sr = librosa.load(wav_path, sr=None)
            
            # Crea la figura con due sottografici
            plt.figure(figsize=(12, 8))
            
            # Traccia la forma d'onda
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {wav_path.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Traccia lo spettrogramma
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram')
            
            # Salva la figura
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

Questi script forniscono strumenti pratici per verificare la qualità del tuo dataset prima dell'addestramento, aiutandoti a identificare e correggere i problemi nelle prime fasi del processo.

---

Una volta che il tuo dataset supera questo controllo di qualità, sei pronto a procedere con la configurazione dell'ambiente di addestramento.

**Passo Successivo:** [Configurazione dell'Addestramento](./2_TRAINING_SETUP.md){: .btn .btn-primary} |
[Torna in Cima](#top){: .btn .btn-primary}
