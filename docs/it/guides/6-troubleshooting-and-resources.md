# Guida alla risoluzione dei problemi e alle risorse TTS


Questa guida fornisce soluzioni per i problemi comuni incontrati durante il processo di preparazione dei dati, addestramento e inferenza TTS, insieme a un elenco di strumenti e risorse utili.

Se un termine di questa guida non ti è familiare, consulta il [glossario](../glossary.md#glossary-of-technical-terms). Risolvere i problemi è più veloce quando sai distinguere con sicurezza [checkpoint](../glossary.md#glossary-checkpoint), [file manifest](../glossary.md#glossary-manifest-file), [CUDA](../glossary.md#glossary-cuda) e [VRAM](../glossary.md#glossary-vram) senza dover tirare a indovinare.

---

## Risoluzione dei Problemi Comuni

Fai riferimento a questa tabella quando incontri problemi. I problemi spesso risalgono alla qualità dei dati o alle impostazioni di configurazione.

Prima di cambiare cinque impostazioni tutte insieme, controlla le basi in questo ordine:

1. conferma percorsi, nomi dei file e struttura delle cartelle
2. conferma che checkpoint e configurazione appartengano davvero alla stessa esecuzione di addestramento
3. conferma che le impostazioni audio corrispondano all'addestramento, soprattutto il sampling rate
4. conferma che l'ambiente sia quello che pensi: ambiente Python corretto, versioni delle dipendenze e visibilità di CUDA

Quest'ordine intercetta una grande parte degli errori da principiante e di livello intermedio prima ancora di iniziare un debug più profondo.

| Categoria del Problema | Problema Specifico | Possibili Cause e Soluzioni | Guida/e Pertinente/i |
| :--- | :--- | :--- | :--- |
| **Preparazione dei Dati** | Errori di script durante la segmentazione o la normalizzazione | Percorsi dei file errati; formato audio inizialmente non supportato; dipendenze mancanti (`ffmpeg`, `pydub`); audio estremamente rumoroso o silenzioso che confonde il rilevamento del silenzio. **Controlla i percorsi dello script, installa le dipendenze e regola i parametri del silenzio.** | [1_DATA_PREPARATION.md](./1-data-preparation.md) |
| **Preparazione dei Dati** | La generazione del manifest salta molti file | Nomi dei file non corrispondenti tra audio e trascrizioni; file di trascrizione vuoti; percorsi errati nello script; file di testo senza codifica UTF-8. **Verifica i nomi, controlla i percorsi e assicurati che i file di testo abbiano contenuto e codifica UTF-8.** | [1_DATA_PREPARATION.md](./1-data-preparation.md) |
| **Configurazione dell'Addestramento** | `pip install` fallisce | Librerie di sistema mancanti come `libsndfile-dev`; versione di Python incompatibile; problemi di rete; conflitti tra pacchetti. **Leggi attentamente i messaggi di errore, installa le librerie di sistema, usa un ambiente virtuale e controlla la documentazione del framework.** | [2_TRAINING_SETUP.md](./2-training-setup.md) |
| **Configurazione dell'Addestramento** | PyTorch `cuda is not available` | Versione errata di PyTorch installata (solo CPU); versione incompatibile del driver NVIDIA o del toolkit CUDA; GPU non rilevata dal sistema operativo. **Reinstalla PyTorch con la versione CUDA corretta dal sito ufficiale e aggiorna i driver.** | [2_TRAINING_SETUP.md](./2-training-setup.md) |
| **Esecuzione dell'Addestramento** | Errore CUDA Out-of-Memory (OOM) all'avvio o durante l'addestramento | `batch_size` troppo grande per la VRAM della GPU; architettura del modello troppo complessa; fuga di memoria nel framework o nel codice personalizzato. **Riduci `batch_size`, abilita AMP/FP16 se disponibile e controlla gli aggiornamenti del framework.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Esecuzione dell'Addestramento** | La Training Loss è `NaN` o diverge | Learning rate troppo alto; gradienti instabili; batch di dati difettoso; problemi di precisione numerica. **Abbassa il learning rate, controlla la qualità dei dati, usa il gradient clipping e prova FP32 se usi AMP/FP16.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Esecuzione dell'Addestramento** | La Training Loss ristagna | Learning rate troppo basso; qualità o varietà dei dati scarsa; modello bloccato in un minimo locale; configurazione errata. **Aumenta leggermente il learning rate, migliora o amplia i dati, controlla la configurazione e prova un optimizer diverso.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Esecuzione dell'Addestramento** | La Validation Loss aumenta mentre la Training Loss diminuisce | Il modello memorizza i dati di addestramento; il set di validazione è insufficiente o poco rappresentativo; l'addestramento dura troppo a lungo. **Interrompi l'addestramento prima, aggiungi dati più vari, usa regolarizzazione e migliora il set di validazione.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Qualità dell'Inferenza** | L'output suona robotico o monotono | Addestramento insufficiente; prosodia scarsa nei dati; limiti dell'architettura del modello; problemi di normalizzazione del testo. **Addestra più a lungo, migliora varietà e qualità dei dati, prova un'altra architettura e assicurati che il testo sia ben punteggiato e normalizzato.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [3_MODEL_TRAINING.md](./3-model-training.md), [4_INFERENCE.md](./4-inference.md) |
| **Qualità dell'Inferenza** | L'output è rumoroso, confuso o incomprensibile | Scarsa qualità dei dati; il modello non è convergente; discrepanza tra configurazione di addestramento e di inferenza; `sampling rate` errato in inferenza. **Pulisci rigorosamente i dati, addestra più a lungo, assicura una corrispondenza esatta tra configurazione e checkpoint e verifica i parametri audio.** | Tutte le Guide |
| **Qualità dell'Inferenza** | L'output suona come lo speaker sbagliato nel fine-tuning | Modello pre-addestrato caricato male; learning rate troppo alto all'inizio; dati o step di fine-tuning insufficienti; speaker ID non coerente. **Verifica `pretrained_model_path` e `ignore_layers`, usa un learning rate più basso e controlla lo speaker ID.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md), [4_INFERENCE.md](./4-inference.md) |
| **Qualità dell'Inferenza** | L'inferenza si interrompe presto o parla troppo veloce o lento | Limite del modello; impostazione di inferenza che limita la lunghezza massima dell'output; parametro di velocità o `length scale` errato. **Controlla la documentazione del framework per lunghezza massima e impostazioni del decoder e regola i controlli della velocità.** | [4_INFERENCE.md](./4-inference.md) |
| **Utilizzo del Modello** | Impossibile caricare il file di checkpoint | File corrotto; checkpoint usato con versione incompatibile del framework o del file di configurazione; percorso errato. **Riscarica il file, verifica la sua integrità, usa la configurazione corretta e controlla il percorso.** | [5_PACKAGING_AND_SHARING.md](./5-packaging-and-sharing.md), [4_INFERENCE.md](./4-inference.md) |

---

## Se Hai Ancora Bisogno di Aiuto

Quando scrivi in un issue tracker, su Discord o in un forum, includi dettagli sufficienti perché un'altra persona possa riprodurre il problema:

- il nome del framework, il branch o la release, e le versioni di Python e PyTorch
- il modello della GPU, la quantità di VRAM e se stai lavorando su CUDA o CPU
- il comando esatto che hai eseguito e il messaggio di errore esatto
- se il problema avviene durante la preparazione dei dati, l'avvio dell'addestramento, il caricamento del checkpoint o l'inferenza
- un piccolo esempio della configurazione coinvolta, di una riga del manifest o del testo di input, se rilevante

I buoni bug report ricevono risposte utili molto più in fretta di un vago "non funziona".

Se possibile, riduci il problema a un comando breve, un input piccolo e un messaggio di errore preciso. Le persone riescono quasi sempre ad aiutare più velocemente quando non devono prima ricostruire tutto il tuo progetto.

## Risorse e Strumenti Utili

Questo elenco include software, librerie e community utili per i progetti TTS.

Considera questa sezione come una mappa iniziale, non come un elenco fisso di raccomandazioni. I repository TTS, i fork mantenuti, gli strumenti cloud e i modelli di prezzo cambiano regolarmente, quindi verifica attività e documentazione attuali prima di impegnarti in un workflow specifico.

### Elaborazione e Analisi Audio:

*   **[Audacity](https://www.audacityteam.org/):** Editor audio gratuito, open source e multipiattaforma. Ottimo per l'ispezione manuale, la pulizia e l'elaborazione di base.
*   **[FFmpeg](https://ffmpeg.org/):** Strumento da riga di comando essenziale per conversione audio e video, ricampionamento e automazione in batch.
*   **[SoX (Sound eXchange Compiled)](http://sox.sourceforge.net/) o [Sox - Codice Sorgente](https://codeberg.org/sox_ng/sox_ng/):** Utility da riga di comando utile per effetti, conversione di formato e informazioni audio con `soxi`.
*   **[pydub](https://github.com/jiaaro/pydub):** Libreria Python per una manipolazione audio semplice.
*   **[librosa](https://librosa.org/doc/latest/index.html):** Libreria Python per analisi audio avanzata, estrazione di feature e visualizzazione.
*   **[soundfile](https://python-soundfile.readthedocs.io/en/latest/):** Libreria Python per leggere e scrivere file audio.
*   **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm):** Libreria Python per la normalizzazione della loudness (LUFS).

### Trascrizione (ASR):

*   **[OpenAI Whisper](https://github.com/openai/whisper):** Modello ASR open source di alta qualità, compatibile con molte lingue. Buona base di partenza, ma la punteggiatura spesso richiede revisione.
*   **[Strumenti e API Google per la trascrizione audio](https://ai.google.dev/):** Google può offrire servizi o modelli utili per la trascrizione. I nomi dei prodotti, i limiti e i piani gratuiti cambiano nel tempo, quindi verifica la documentazione attuale prima di scegliere un flusso di lavoro specifico.
*   **Servizi Cloud ASR:**
    *   [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
    *   [AWS Transcribe](https://aws.amazon.com/transcribe/)
    *   [Azure Speech Service](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/)
    *   *Spesso affidabili, pay-as-you-go, con possibili quote gratuite iniziali.*
*   **[Hugging Face Transformers - Modelli ASR](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition):** Hub per molti modelli ASR pre-addestrati, incluse versioni affinate di Whisper e altri.
*   **[ElevenLabs Speech To Text (Scribe)](https://elevenlabs.io/speech-to-text):** *Servizio commerciale.* Noto per l'alta accuratezza, ma a pagamento e potenzialmente costoso.

### Framework e Codebase TTS (Esempi - Cerca fork o successori attivi):

*   **[StyleTTS2 (Research Repo)](https://github.com/yl4579/StyleTTS2):** Lavoro influente sul controllo dello stile. Cerca fork mantenuti attivamente con pipeline complete.
*   **[VITS (Research Repo)](https://github.com/jaywalnut310/vits):** Architettura popolare end-to-end. Esistono molti fork e implementazioni.
*   **[Coqui TTS (Archiviato)](https://github.com/coqui-ai/TTS):** Riferimento storico. Il progetto è stato molto influente, ma per nuovi flussi di lavoro è meglio che i principianti preferiscano progetti attivi o fork realmente mantenuti.
*   **[ESPnet](https://github.com/espnet/espnet):** Toolkit per il parlato che include ricette TTS per vari modelli.
*   **Cerca su GitHub:** Usa parole chiave come "TTS", "VITS training", "StyleTTS2 training" o "PyTorch TTS" per trovare progetti attuali.

### Ambiente Python e Deep Learning:

*   **[Python](https://www.python.org/):** Il linguaggio di programmazione principale.
*   **[PyTorch](https://pytorch.org/):** La principale libreria di deep learning usata dalla maggior parte dei framework TTS moderni.
*   **[TensorBoard](https://www.tensorflow.org/tensorboard):** Essenziale per visualizzare i progressi dell'addestramento.
*   **[pip](https://pip.pypa.io/en/stable/) / [uv](https://github.com/astral-sh/uv):** Installer di pacchetti Python. `uv` è un'alternativa più recente e spesso più veloce.
*   **[conda](https://docs.conda.io/en/latest/) / [venv](https://docs.python.org/3/library/venv.html):** Strumenti per creare ambienti Python isolati.
*   **[Git](https://git-scm.com/):** Sistema di controllo versione essenziale per clonare repository e gestire il codice.
*   **[Hugging Face Hub](https://huggingface.co/):** Piattaforma per condividere modelli, dataset e codice.

### Community:

*   **GitHub Discussions/Issues del Framework TTS:** Controlla il repository specifico che stai usando.
*   **Server Discord:** Molte community AI e ML hanno canali dedicati al TTS.
*   **Reddit:** Subreddit come `r/SpeechSynthesis` e `r/MachineLearning`.

---

Questo conclude la serie principale di guide. Ricorda che costruire buoni modelli TTS spesso richiede iterazione: rivedere la preparazione dei dati o regolare i parametri di addestramento in base ai risultati è una pratica comune.

---
