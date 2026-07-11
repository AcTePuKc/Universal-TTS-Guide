# Guida alla configurazione dell'addestramento TTS


Con il dataset già pronto, il passaggio successivo consiste nel configurare l'ambiente software e una prima configurazione utilizzabile per l'addestramento o il fine-tuning del modello TTS.

Se un termine relativo all'hardware o all'addestramento non è chiaro, consulta il [glossario](../glossary.md#glossary-of-technical-terms). Questa pagina spiega solo i termini che influenzano direttamente le decisioni di configurazione.

---

## Setup dell'ambiente di addestramento

Questa sezione copre l'installazione del software necessario e l'organizzazione dei file del progetto.

### Scegliere e clonare un framework TTS

Se sei all'inizio, scegli un framework mantenuto attivamente, con istruzioni di installazione chiare e con supporto già presente per il tipo di addestramento di cui hai bisogno. Non partire da "l'architettura più avanzata". Parti da qualcosa che puoi installare, eseguire e debuggare.

-   **Seleziona un framework:** Scegli una codebase adatta ai tuoi obiettivi. Considera:
    *   **Architettura:** VITS, StyleTTS2, Tacotron2 + vocoder e simili.
    *   **Supporto al fine-tuning:** Se esiste un percorso chiaro partendo da modelli pre-addestrati.
    *   **Supporto linguistico:** Se tokenizzazione e normalizzazione funzionano bene per la tua lingua.
    *   **Community e manutenzione:** Se il repository è ancora attivo e ben documentato.
    *   **Modelli pre-addestrati:** Se hai un buon punto di partenza.

### Confronto tra architetture TTS

| Architettura | Vantaggi | Svantaggi | Ideale per | Requisiti hardware |
|:-------------|:---------|:----------|:-----------|:-------------------|
| **VITS** | End-to-end, audio di alta qualità, inferenza rapida, buona per il fine-tuning | Più complessa, può essere instabile, richiede tuning accurato | Voice cloning single-speaker e progetti orientati alla qualità | Addestramento: 8GB+ VRAM, inferenza: 4GB+ VRAM |
| **StyleTTS2** | Ottimo controllo di voce e stile, qualità molto alta | Più recente, più complessa, meno risorse pratiche | Voce espressiva e controllo dello stile | Addestramento: 12GB+ VRAM, inferenza: 6GB+ VRAM |
| **Tacotron2 + HiFi-GAN** | Più stabile, più facile da capire, più tutorial | Pipeline a due stadi, qualità inferiore ai modelli più recenti | Progetti didattici o più prevedibili | Addestramento: 6GB+ VRAM, inferenza: 2GB+ VRAM |
| **FastSpeech2** | Inferenza veloce, più stabile di Tacotron2, buona documentazione | Richiede allineamenti fonemici e preprocessing più complesso | Applicazioni rapide e output più controllato | Addestramento: 8GB+ VRAM, inferenza: 2GB+ VRAM |
| **YourTTS / XTTS** | Supporto multilingue, zero-shot, flessibilità tra lingue | Setup complesso, richiede più attenzione ai dati | Progetti multilingue e scenari cross-lingual | Addestramento: 10GB+ VRAM, inferenza: 4GB+ VRAM |
| **TTS basato su diffusion** | Alto potenziale di qualità, prosodia più naturale | Inferenza lenta e addestramento costoso | Generazione offline e ricerca | Addestramento: 16GB+ VRAM, inferenza: 8GB+ VRAM |

**Nota sull'hardware:**
- Si tratta di minimi approssimativi.
- Batch size più grandi e configurazioni più pesanti richiederanno più VRAM.
- L'inferenza su CPU è possibile, ma sarà molto più lenta.

**Scorciatoia pratica:** se stai scegliendo il primo framework per un progetto reale, invece di confrontare le architetture, scegli quello con la documentazione di installazione più chiara, il tracker dei problemi più attivo e un esempio di fine-tuning più vicino al tuo caso d'uso.

### Requisiti hardware in base alla scala del progetto

#### Requisiti GPU per modello e dimensione del dataset

| Tipo di modello | Dataset piccolo (<10h) | Dataset medio (10-50h) | Dataset grande (>50h) | GPU consigliate |
|:----------------|:-----------------------|:------------------------|:----------------------|:----------------|
| **Tacotron2 + HiFi-GAN** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **FastSpeech2** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **VITS** | 12GB VRAM | 16GB VRAM | 24GB+ VRAM | RTX 3080, RTX 3090, A5000 |
| **StyleTTS2** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |
| **XTTS-v2** | 24GB VRAM | 32GB VRAM | 40GB+ VRAM | RTX 4090, A100, A6000 |
| **TTS basato su diffusion** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |

#### CPU, RAM e spazio su disco

| Scala | CPU | RAM | Storage |
|:------|:----|:----|:--------|
| **Personale** | 4+ core, 2.5GHz+ | 16GB | 50GB SSD |
| **Ricerca** | 8+ core, 3.0GHz+ | 32GB | 100GB+ SSD |
| **Produzione** | 16+ core, 3.5GHz+ | 64GB+ | 500GB+ NVMe SSD |

#### Opzioni cloud GPU indicative*

**\*Nota sulla validità nel tempo:** i provider e gli esempi di GPU riportati di seguito riflettono il panorama cloud al momento della stesura di questa guida. Offerte, disponibilità e prezzi cambiano spesso in base a regione, sconti e spot pricing. Usa questa tabella solo come riferimento iniziale e verifica opzioni e prezzi attuali sul sito del provider prima di stimare il budget.

| Provider | Opzione GPU | VRAM | Costo relativo | Ideale per |
|:---------|:------------|:-----|:---------------|:-----------|
| **Google Colab** | T4/P100 (i livelli gratuiti possono variare)<br>V100/A100 (i livelli a pagamento possono variare) | 16GB<br>16-40GB | Basso a medio | Test e dataset piccoli |
| **Kaggle** | P100/T4 | 16GB | Basso | Dataset piccoli e medi |
| **AWS** | g4dn.xlarge (T4)<br>p3.2xlarge (V100)<br>p4d.24xlarge (A100) | 16GB<br>16GB<br>40GB | Medio a molto alto | Qualsiasi scala |
| **GCP** | Istanze T4<br>Istanze A100 | 16GB<br>40GB | Medio a molto alto | Qualsiasi scala |
| **Azure** | Istanze classe V100 o A100 | 16GB+ | Medio a molto alto | Qualsiasi scala |
| **Lambda Labs** | 1x RTX 3090<br>1x A100 | 24GB<br>40GB | Medio | Ricerca e dataset medi |
| **Vast.ai** | Varie GPU consumer | 8-24GB | Basso a medio | Budget ridotto |

#### Tempi di addestramento molto approssimativi

**Nota sui tempi:** questi intervalli cambiano molto in base all'implementazione, al batch size, alla pulizia del dataset, al tokenizer, al checkpoint di partenza e al fatto che tu stia facendo fine-tuning o addestramento da zero. Considerali come ordine di grandezza, non come promessa.

| Modello | Dimensione dataset | GPU | Tempo approssimativo | Step fino alla convergenza |
|:--------|:-------------------|:----|:---------------------|:---------------------------|
| **Tacotron2 + HiFi-GAN** | 10 ore | RTX 3080 | 2-3 giorni | 50-100K step |
| **FastSpeech2** | 10 ore | RTX 3080 | 2-3 giorni | 150-200K step |
| **VITS** | 10 ore | RTX 3090 | 3-5 giorni | 300-500K step |
| **StyleTTS2** | 10 ore | RTX 3090 | 4-7 giorni | 500-800K step |
| **XTTS-v2** | 10 ore | RTX 4090 | 5-10 giorni | 1M+ step |

#### Suggerimenti per ridurre i requisiti hardware

1. **Gradient accumulation:** simula batch size più grandi accumulando i gradienti durante più passaggi forward/backward.
2. **Mixed precision training:** usa FP16 invece di FP32 per ridurre l'uso di VRAM fino al 50%.
3. **Gradient checkpointing:** scambia memoria con calcolo ricalcolando le attivazioni durante il passaggio backward.
4. **Model parallelism:** distribuisce modelli grandi su più GPU.
5. **Addestramento progressivo:** inizia con modelli o configurazioni più piccoli e aumenta gradualmente la complessità.

Questi requisiti dovrebbero aiutarti a pianificare l'hardware in base agli obiettivi e al budget del progetto.

-   **Clona il repository:** Una volta scelto, clona il framework con Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY>
    ```
    *   Esempio: `git clone https://github.com/some-user/some-tts-framework.git`

### Configurare l'ambiente Python e installare le dipendenze

-   **Ambiente virtuale:** È consigliabile usare un ambiente virtuale dedicato.
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
-   **Installare PyTorch con CUDA:** Usa il [configuratore ufficiale di PyTorch](https://pytorch.org/get-started/locally/) per far combaciare versione CUDA, driver e package manager.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    ```
-   **Installare le dipendenze del framework:**
    ```bash
    pip install -r requirements.txt

    # Se usi uv:
    uv pip install -r requirements.txt
    ```
    *   **Se compaiono errori:** Controlla librerie di sistema mancanti, incompatibilità di versione o problemi tra CUDA e PyTorch.

#### Test minimo dell'ambiente

Prima di modificare una configurazione grande o avviare un addestramento lungo, verifica che questi comandi funzionino:

```bash
python --version
ffmpeg -version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__)"
```

Se uno di questi fallisce, fermati e sistema l'ambiente prima. Costa molto meno che debuggare un addestramento rotto dopo ore di attesa.

### Organizzare la cartella del progetto

-   Una struttura chiara evita confusione con manifest, checkpoint e configurazioni.

    ```mermaid
    flowchart TD
        root["Project Root"] --> repo["TTS Repo Directory"]
        repo --> scripts["Script principali"]
        scripts --> train["train.py"]
        scripts --> inference["inference.py"]
        repo --> config["configs/base_config.yaml"]
        repo --> requirements["requirements.txt"]
        repo --> repoMore["altri file del framework"]
    ```

    ```mermaid
    flowchart TD
        root["Project Root"] --> dataset["my_tts_dataset"]
        dataset --> audio["normalized_chunks"]
        audio --> wav1["segment_00001.wav"]
        audio --> wavMore["altri file .wav"]
        dataset --> transcripts["transcripts (opzionale)"]
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

-   **Percorsi:** Assicurati che i percorsi nella configurazione siano corretti rispetto al punto da cui eseguirai davvero `train.py`, di solito dentro `<TTS_REPO_DIRECTORY>`.

---

## Configurare l'esecuzione dell'addestramento

Prima di avviare l'addestramento, hai bisogno di un file di configurazione che dica al framework come addestrare il modello usando i tuoi dati.

### 1. Trovare e copiare una configurazione base

-   **Cerca esempi:** Controlla la cartella `configs/` del framework scelto.
-   **Scegli in base all'obiettivo:**
    *   **Fine-tuning:** Cerca nomi come `config_ft.yaml` o `finetune_*.yaml`; spesso presuppongono un modello pre-addestrato.
    *   **Addestramento da zero:** Cerca `config_base.yaml` o `train_*.yaml`.
    *   **Dimensione del dataset:** Alcune implementazioni offrono varianti per dataset piccoli (`_sm`) o grandi (`_lg`).
-   **Copia e rinomina:** Copia il modello in una cartella personale e assegnagli un nome descrittivo per questa esecuzione, ad esempio `my_yoruba_voice_ft_config.yaml`.
    ```bash
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```
    In Windows PowerShell usa `Copy-Item` se non stai lavorando in Git Bash.

**Consiglio per chi inizia:** parti dall'esempio funzionante più vicino che il framework già fornisce. Non costruire la tua prima configurazione da zero.

### 2. Modificare il file di configurazione personalizzato

-   Apri la copia della configurazione in un editor di testo.
-   **Modifica i parametri chiave:** I nomi esatti varieranno tra framework, ma le categorie sono quasi sempre simili.

    ```yaml
    # --- Dataset e caricamento dati ---
    # Percorsi relativi al punto da cui esegui train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt"
    val_filelist_path: "../my_tts_dataset/val_list.txt"
    # Alcuni framework possono richiedere anche data_path o audio_root per la directory audio.

    # --- Output e log ---
    output_directory: "../checkpoints/my_yoruba_voice_run1"
    log_interval: 100
    validation_interval: 1000
    save_checkpoint_interval: 5000

    # --- Iperparametri principali ---
    epochs: 1000
    batch_size: 16                     # Riduci in caso di errori CUDA OOM.
    learning_rate: 1e-4                # Può richiedere tuning; spesso è più basso nel fine-tuning.
    # lr_scheduler: "cosine_decay"
    # weight_decay: 0.01

    # --- Parametri audio ---
    sampling_rate: 22050               # DEVE corrispondere alla frequenza di campionamento del dataset preparato (dalla Guida 1).
    # filter_length: 1024
    # hop_length: 256
    # win_length: 1024
    # n_mel_channels: 80
    # mel_fmin: 0.0
    # mel_fmax: 8000.0

    # --- Architettura del modello ---
    # model_type: "VITS"
    # hidden_channels: 192
    # num_speakers: 1

    # --- Dettagli del fine-tuning (se applicabile) ---
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth"
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```

-   **Leggi la documentazione del framework:** È la fonte corretta per il significato preciso di ogni parametro.
-   **Nota sui termini:** in questa configurazione, un [checkpoint](../glossary.md#glossary-checkpoint) è un'istantanea salvata del modello, mentre il [sampling rate](../glossary.md#glossary-sampling-rate) deve corrispondere esattamente al dataset preparato.
-   **Errore comune all'inizio:** al primo tentativo modifica solo ciò che è obbligatorio, come percorsi del dataset, directory di output, sampling rate, batch size e checkpoint di fine-tuning se necessario. Non cambiare dieci impostazioni insieme prima di aver verificato che la pipeline parta almeno una volta.

### 3. Considerazioni su hardware e dataset

-   **VRAM GPU:** la [VRAM](../glossary.md#glossary-vram) è la memoria della scheda grafica. `batch_size` è la leva principale per controllare l'uso della memoria. Inizia con un valore consigliato e riducilo in caso di errore «CUDA out of memory».
-   **Dimensione del dataset rispetto alle epoche:**
    *   **Dataset piccoli (<20h):** possono richiedere meno epoche (ad esempio 300-1500), ma hanno bisogno di un monitoraggio attento della [validation loss](../glossary.md#glossary-validation-loss) e dei sample per evitare l'[overfitting](../glossary.md#glossary-overfitting). Considera learning rate più bassi.
    *   **Dataset grandi (>50h):** possono trarre beneficio da più epoche (1000+) per apprendere completamente i pattern dei dati.
-   **CPU:** anche se la GPU svolge il lavoro principale, serve una CPU multi-core adeguata per caricare e preelaborare i dati.
-   **Storage:** assicurati di avere spazio per dataset, ambiente Python, codice del framework e soprattutto checkpoint, che possono occupare centinaia di MB o diversi GB.

### 4. Strumenti di monitoraggio (TensorBoard)

-   La maggior parte dei framework TTS moderni si integra con [TensorBoard](https://www.tensorflow.org/tensorboard).
-   Nella configurazione spesso trovi opzioni come `use_tensorboard: True` o `log_directory`.
-   Durante l'addestramento puoi di solito eseguire `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (ad esempio `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) in un terminale separato per monitorare curve di loss, learning rate e sample di validazione sintetizzati.
-   Se TensorBoard risulta vuoto, controlla prima che il framework stia davvero scrivendo i file evento nella directory prevista. Una dashboard vuota spesso dipende solo da un percorso log errato.

---

Con l'ambiente pronto e la configurazione adattata ai tuoi dati, puoi passare al vero addestramento del modello.

## Prima di continuare

- [ ] Il tuo ambiente Python è attivo e le dipendenze del framework si installano senza errori.
- [ ] `torch.cuda.is_available()` restituisce `True` se intendi addestrare su GPU.
- [ ] `ffmpeg` e le librerie di sistema necessarie sono installati e visibili nel PATH.
- [ ] I percorsi della configurazione puntano a manifest, checkpoint e cartelle di output reali.
- [ ] Il `sampling_rate` della configurazione corrisponde esattamente al dataset preparato.
