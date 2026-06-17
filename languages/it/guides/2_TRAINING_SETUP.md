# Guida 2: Configurazione e Setup dell'Ambiente di Addestramento

**Navigazione:** [README Principale]({{ site.baseurl }}/languages/it/){: .btn .btn-primary} | [Passo Precedente: Preparazione dei Dati](./1_DATA_PREPARATION.md){: .btn .btn-primary} | [Passo Successivo: Addestramento del Modello](./3_MODEL_TRAINING.md){: .btn .btn-primary}

Con il tuo dataset preparato, la fase successiva comporta la configurazione dell'ambiente software necessario e l'impostazione dei parametri per la tua specifica esecuzione di addestramento.

---

## 3. Setup dell'Ambiente di Addestramento

Questa sezione copre l'installazione del software necessario e l'organizzazione dei file di progetto.

### 3.1. Scegliere e Clonare un Framework TTS

-   **Seleziona un Framework:** Scegli una codebase TTS adatta ai tuoi obiettivi. Considera fattori come:
    *   **Architettura:** VITS, StyleTTS2, Tacotron2+Vocoder, ecc. Le architetture più recenti spesso producono una qualità migliore.
    *   **Supporto al Fine-tuning:** Il framework supporta esplicitamente il fine-tuning da modelli pre-addestrati? Questo è spesso più facile che addestrare da zero.
    *   **Supporto Linguistico:** Verifica se il modello/tokenizer gestisce bene la tua lingua target.
    *   **Community e Manutenzione:** Il repository è mantenuto attivamente? Ci sono discussioni della community o canali di supporto?
    *   **Modelli Pre-addestrati:** Il framework fornisce modelli pre-addestrati adatti come punto di partenza per il fine-tuning?

#### Confronto delle Architetture TTS

Quando selezioni un'architettura TTS, considera queste opzioni popolari e le loro caratteristiche:

| Architettura | Pro | Contro | Ideale Per | Requisiti Hardware |
|:-------------|:-----|:-----|:---------|:---------------------|
| **VITS** | • End-to-end (nessun vocoder separato)<br>• Audio di alta qualità<br>• Inferenza veloce<br>• Buono per il fine-tuning | • Complesso da comprendere<br>• Può essere instabile durante l'addestramento<br>• Richiede un'attenta messa a punto degli iperparametri | • Voice cloning a singolo speaker<br>• Progetti che necessitano di output di alta qualità<br>• Quando hai più di 5 ore di dati | • Addestramento: 8GB+ VRAM<br>• Inferenza: 4GB+ VRAM |
| **StyleTTS2** | • Eccellente controllo di voce e stile<br>• Qualità allo stato dell'arte<br>• Buono per emozione/prosodia | • Più recente, implementazioni potenzialmente meno stabili<br>• Architettura più complessa<br>• Meno risorse della community | • Progetti che richiedono controllo dello stile<br>• Sintesi vocale espressiva<br>• Multi-speaker con style transfer | • Addestramento: 12GB+ VRAM<br>• Inferenza: 6GB+ VRAM |
| **Tacotron2 + HiFi-GAN** | • Ben consolidato, stabile<br>• Più facile da comprendere<br>• Più tutorial disponibili<br>• Componenti separati per un debug più facile | • Pipeline a due stadi (più lenta)<br>• Generalmente qualità inferiore rispetto ai modelli più recenti<br>• Più soggetto a errori di attention su testi lunghi | • Progetti didattici<br>• Quando la stabilità ha priorità sulla qualità<br>• Ambienti con meno risorse | • Addestramento: 6GB+ VRAM<br>• Inferenza: 2GB+ VRAM |
| **FastSpeech2** | • Non-autoregressivo (inferenza più veloce)<br>• Più stabile di Tacotron2<br>• Buona documentazione | • Richiede allineamenti fonemici<br>• Preprocessing più complesso<br>• Qualità non alta come VITS/StyleTTS2 | • Applicazioni in tempo reale<br>• Quando la velocità di inferenza è critica<br>• Output più controllato | • Addestramento: 8GB+ VRAM<br>• Inferenza: 2GB+ VRAM |
| **YourTTS (variante di VITS)** | • Supporto multilingue<br>• Voice cloning zero-shot<br>• Buono per il transfer linguistico | • Setup di addestramento complesso<br>• Richiede un'attenta preparazione dei dati<br>• Potrebbe necessitare di dataset più grandi | • Progetti multilingue<br>• Voice cloning cross-linguale<br>• Quando è necessaria flessibilità linguistica | • Addestramento: 10GB+ VRAM<br>• Inferenza: 4GB+ VRAM |
| **TTS basato su Diffusion** | • Massimo potenziale di qualità<br>• Prosodia più naturale<br>• Migliore gestione delle parole rare | • Inferenza molto lenta<br>• Addestramento estremamente intensivo in termini di calcolo<br>• Più recente, meno consolidato | • Generazione offline<br>• Quando la qualità conta più della velocità<br>• Progetti di ricerca | • Addestramento: 16GB+ VRAM<br>• Inferenza: 8GB+ VRAM |

**Nota sui Requisiti Hardware:**
- Questi sono minimi approssimativi; batch size più grandi o configurazioni del modello richiederanno più VRAM
- I tempi di addestramento variano significativamente: VITS/StyleTTS2 tipicamente necessitano di più epoche di Tacotron2
- L'inferenza su CPU è possibile per tutti i modelli ma sarà significativamente più lenta

### 1.3. Requisiti Hardware Dettagliati

Scegliere l'hardware giusto è fondamentale per un addestramento riuscito di modelli TTS. Ecco una ripartizione dettagliata dei requisiti per diversi scenari:

#### Requisiti GPU per Tipo di Modello e Dimensione del Dataset

| Tipo di Modello | Dataset Piccolo (<10h) | Dataset Medio (10-50h) | Dataset Grande (>50h) | Modelli GPU Consigliati |
|:-----------|:---------------------|:------------------------|:---------------------|:-----------------------|
| **Tacotron2 + HiFi-GAN** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **FastSpeech2** | 8GB VRAM | 12GB VRAM | 16GB+ VRAM | RTX 3060, RTX 2080, T4 |
| **VITS** | 12GB VRAM | 16GB VRAM | 24GB+ VRAM | RTX 3080, RTX 3090, A5000 |
| **StyleTTS2** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |
| **XTTS-v2** | 24GB VRAM | 32GB VRAM | 40GB+ VRAM | RTX 4090, A100, A6000 |
| **TTS basato su Diffusion** | 16GB VRAM | 24GB VRAM | 32GB+ VRAM | RTX 3090, RTX 4090, A100 |

#### CPU e Memoria di Sistema

| Scala di Addestramento | Requisiti CPU | RAM di Sistema | Storage |
|:---------------|:-----------------|:-----------|:--------|
| **Hobby/Personale** | 4+ core, 2.5GHz+ | 16GB | 50GB SSD |
| **Ricerca** | 8+ core, 3.0GHz+ | 32GB | 100GB+ SSD |
| **Produzione** | 16+ core, 3.5GHz+ | 64GB+ | 500GB+ NVMe SSD |

#### Opzioni di GPU Cloud e Costi Approssimativi

| Provider Cloud | Opzione GPU | VRAM | Costo Approx./Ora | Ideale Per |
|:---------------|:-----------|:-----|:------------------|:---------|
| **Google Colab** | T4/P100 (Free)<br>V100/A100 (Pro) | 16GB<br>16-40GB | Free<br>$10-$15 | Sperimentazione, dataset piccoli |
| **Kaggle** | P100/T4 | 16GB | Free (ore limitate) | Dataset piccoli-medi |
| **AWS** | g4dn.xlarge (T4)<br>p3.2xlarge (V100)<br>p4d.24xlarge (A100) | 16GB<br>16GB<br>40GB | $0.50-$0.75<br>$3.00-$3.50<br>$20.00-$32.00 | Qualsiasi scala, produzione |
| **GCP** | n1-standard-8 + T4<br>a2-highgpu-1g (A100) | 16GB<br>40GB | $0.35-$0.50<br>$3.80-$4.50 | Qualsiasi scala, produzione |
| **Azure** | NC6s_v3 (V100)<br>NC24ads_A100_v4 | 16GB<br>80GB | $3.00-$3.50<br>$16.00-$24.00 | Qualsiasi scala, produzione |
| **Lambda Labs** | 1x RTX 3090<br>1x A100 | 24GB<br>40GB | $1.10<br>$1.99 | Ricerca, dataset medi |
| **Vast.ai** | Varie GPU consumer | 8-24GB | $0.20-$1.00 | Addestramento con budget limitato |

#### Stime dei Tempi di Addestramento

| Modello | Dimensione Dataset | GPU | Tempo di Addestramento Approssimativo | Epoche fino alla Convergenza |
|:------|:-------------|:----|:--------------------------|:----------------------|
| **Tacotron2 + HiFi-GAN** | 10 ore | RTX 3080 | 2-3 giorni | 50-100K step |
| **FastSpeech2** | 10 ore | RTX 3080 | 2-3 giorni | 150-200K step |
| **VITS** | 10 ore | RTX 3090 | 3-5 giorni | 300-500K step |
| **StyleTTS2** | 10 ore | RTX 3090 | 4-7 giorni | 500-800K step |
| **XTTS-v2** | 10 ore | RTX 4090 | 5-10 giorni | 1M+ step |

#### Suggerimenti di Ottimizzazione per Ridurre i Requisiti Hardware

1. **Gradient Accumulation**: Simula batch size più grandi accumulando i gradienti su più passaggi forward/backward
2. **Mixed Precision Training**: Usa FP16 invece di FP32 per ridurre l'uso di VRAM fino al 50%
3. **Gradient Checkpointing**: Scambia il calcolo con la memoria ricalcolando le attivazioni durante il backward pass
4. **Model Parallelism**: Suddividi i modelli grandi su più GPU
5. **Progressive Training**: Inizia con modelli/configurazioni più piccoli e aumenta gradualmente la complessità

Questi requisiti dovrebbero aiutarti a pianificare le tue necessità hardware in base agli obiettivi specifici del tuo progetto e ai vincoli di budget.
-   **Clona il Repository:** Una volta scelto, clona il repository del codice del framework usando Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY> # Naviga nella directory clonata
    ```
    *   Esempio: `git clone https://github.com/some-user/some-tts-framework.git`

### 3.2. Configurare l'Ambiente Python e Installare le Dipendenze

-   **Ambiente Virtuale (Consigliato):** Crea e attiva un ambiente virtuale Python dedicato per isolare le dipendenze ed evitare conflitti con altri progetti o con i pacchetti Python di sistema.
    *   **Usando `venv` (integrato):**
        ```bash
        python -m venv venv_tts  # Crea un ambiente chiamato 'venv_tts'
        # Attivalo:
        # Windows: .\venv_tts\Scripts\activate
        # Linux/macOS: source venv_tts/bin/activate
        ```
    *   **Usando `conda`:**
        ```bash
        conda create --name tts_env python=3.9 # O la versione Python desiderata
        conda activate tts_env
        ```
-   **Installa PyTorch con CUDA:** Questo è fondamentale per l'accelerazione GPU. Visita la [Guida Ufficiale di Installazione di PyTorch](https://pytorch.org/get-started/locally/) e seleziona le opzioni corrispondenti al tuo OS, package manager (`pip` o `conda`), piattaforma di calcolo (versione CUDA) e versione di PyTorch desiderata. **Assicurati che i driver NVIDIA installati siano compatibili con la versione CUDA scelta.**
    ```bash
    # Comando di esempio con pip per CUDA 11.8 (controlla il sito di PyTorch per i comandi attuali!)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Verifica l'installazione:
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    # Dovrebbe restituire la versione di PyTorch, True e la tua versione CUDA se ha successo.
    ```
-   **Installa i Requisiti del Framework:** La maggior parte dei framework elenca le proprie dipendenze in un file `requirements.txt`. Installale usando `pip` (o `uv`, che spesso è più veloce).
    ```bash
    # Naviga prima nella directory del framework se non sei già lì
    # Usando pip:
    pip install -r requirements.txt

    # Usando uv (se installato: pip install uv):
    uv pip install -r requirements.txt
    ```
    *   **Risoluzione dei Problemi:** Presta attenzione a eventuali errori di installazione. Potrebbero indicare librerie di sistema mancanti (come `libsndfile`), versioni di pacchetti incompatibili o problemi con la tua configurazione CUDA/PyTorch. Controlla la documentazione del framework per i prerequisiti specifici.

### 3.3. Organizzare la Cartella del Progetto

-   Una struttura di cartelle ben organizzata semplifica la gestione del tuo progetto. Colloca il tuo dataset preparato (o crea un link simbolico ad esso) all'interno o accanto al codice del framework. Una struttura comune si presenta così:

    ```bash
    <YOUR_PROJECT_ROOT>/
    ├── <TTS_REPO_DIRECTORY>/         # Il codice del framework clonato
    │   ├── train.py                 # Script di addestramento principale (il nome può variare)
    │   ├── inference.py             # Script di inferenza (il nome può variare)
    │   ├── configs/                 # Directory per i file di configurazione
    │   │   └── base_config.yaml     # Esempio di config del framework
    │   ├── requirements.txt
    │   └── ... (altri file del framework)
    │
    ├── my_tts_dataset/              # Il tuo dataset preparato dalla Guida 1
    │   ├── normalized_chunks/       # File audio finali
    │   │   ├── segment_00001.wav
    │   │   └── ...
    │   ├── transcripts/             # Opzionale: file di testo se non direttamente nel manifest
    │   ├── train_list.txt           # Manifest di addestramento
    │   └── val_list.txt             # Manifest di validazione
    │
    ├── checkpoints/                 # Crea questa directory: Dove verranno salvati i modelli
    │   └── my_custom_model/         # Sottodirectory per una specifica esecuzione di addestramento
    │
    └── my_configs/                  # Opzionale: Colloca qui le tue config personalizzate
        └── my_training_run_config.yaml
    ```
-   **Percorsi:** Assicurati che i percorsi specificati successivamente nel tuo file di configurazione (per dataset, output) siano corretti rispetto alla posizione da cui *eseguirai* lo script `train.py` (di solito da dentro la `<TTS_REPO_DIRECTORY>`).

---

## 4. Configurare l'Esecuzione dell'Addestramento

Prima di avviare l'addestramento, devi creare un file di configurazione che indichi al framework *come* addestrare il modello, usando i *tuoi* dati specifici.

### 4.1. Trovare e Copiare una Configurazione Base

-   **Individua gli Esempi:** Esplora la directory `configs/` all'interno del framework TTS. Cerca file di configurazione (`.yaml`, `.json` o simili) che fungano da template.
-   **Scegli in Modo Appropriato:** Seleziona un file di config che corrisponda al tuo obiettivo:
    *   **Fine-tuning:** Cerca nomi come `config_ft.yaml`, `finetune_*.yaml`. Questi spesso presuppongono che fornirai un modello pre-addestrato.
    *   **Addestramento da Zero:** Cerca nomi come `config_base.yaml`, `train_*.yaml`.
    *   **Dimensione del Dataset:** Alcuni framework potrebbero offrire config ottimizzate per dataset piccoli (`_sm`) o grandi (`_lg`).
-   **Copia e Rinomina:** Copia il file template scelto in una nuova posizione (ad esempio la tua directory `my_configs/` o all'interno della directory `configs/` del framework) e dagli un nome descrittivo per la tua esecuzione specifica (ad esempio `my_yoruba_voice_ft_config.yaml`).
    ```bash
    # Esempio: Copiare una config di fine-tuning
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```

### 4.2. Modificare il Tuo File di Configurazione Personalizzato

-   Apri il tuo file di configurazione appena copiato (`my_yoruba_voice_ft_config.yaml`) in un editor di testo.
-   **Modifica i Parametri Chiave:** Rivedi e modifica con attenzione i parametri. I nomi dei parametri **varieranno significativamente** tra i framework, ma le categorie comuni includono:

    ```yaml
    # --- Dataset e Caricamento dei Dati ---
    # Percorsi relativi alla posizione da cui esegui train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt" # Percorso al tuo manifest di addestramento
    val_filelist_path: "../my_tts_dataset/val_list.txt"   # Percorso al tuo manifest di validazione
    # Alcuni framework potrebbero necessitare di 'data_path' o 'audio_root' che puntino alla directory audio in alternativa/aggiunta.

    # --- Output e Logging ---
    output_directory: "../checkpoints/my_yoruba_voice_run1" # MOLTO IMPORTANTE: Dove vengono salvati modelli, log e sample. Crea questa dir base se necessario.
    log_interval: 100                  # Ogni quanto (in step/batch) stampare i log
    validation_interval: 1000          # Ogni quanto (in step/batch) eseguire la validazione
    save_checkpoint_interval: 5000     # Ogni quanto (in step/batch) salvare i checkpoint del modello

    # --- Iperparametri Principali di Addestramento ---
    epochs: 1000                       # Numero totale di passaggi sui dati di addestramento. Regola in base alla dimensione del dataset e alla convergenza.
    batch_size: 16                     # Numero di sample elaborati in parallelo per GPU. RIDUCI se ottieni errori CUDA OOM. AUMENTA per un addestramento più veloce se la VRAM lo consente.
    learning_rate: 1e-4                # Learning rate iniziale. Potrebbe richiedere una messa a punto (es. più basso per il fine-tuning: 5e-5 o 1e-5).
    # lr_scheduler: "cosine_decay"     # Schedule del learning rate (es. step decay, exponential decay) - dipende dal framework
    # weight_decay: 0.01               # Parametro di regolarizzazione

    # --- Parametri Audio ---
    sampling_rate: 22050               # CRITICO: DEVE corrispondere al sampling rate del tuo dataset preparato (dalla Guida 1).
    # Altri parametri audio (spesso dipendono dall'architettura del modello):
    # filter_length: 1024              # Dimensione FFT per la STFT
    # hop_length: 256                  # Hop size per la STFT
    # win_length: 1024                 # Window size per la STFT
    # n_mel_channels: 80               # Numero di bande Mel
    # mel_fmin: 0.0                    # Frequenza Mel minima
    # mel_fmax: 8000.0                 # Frequenza Mel massima (spesso sampling_rate / 2)

    # --- Architettura del Modello ---
    # model_type: "VITS"               # Tipo di architettura del modello
    # hidden_channels: 192             # Dimensione dei layer interni
    # num_speakers: 1                  # Imposta a >1 per i dataset multi-speaker (deve corrispondere alla preparazione dati)

    # --- Specifiche del Fine-tuning (Se Applicabile) ---
    # Imposta 'True' o fornisci il percorso durante il fine-tuning
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth" # Percorso al checkpoint pre-addestrato da cui partire.
    # Opzionale: Specifica i layer da ignorare/reinizializzare se necessario
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```
-   **Leggi la Documentazione del Framework:** Consulta la documentazione specifica del framework TTS scelto per capire cosa fa ciascun parametro nel suo file di configurazione.

### 4.3. Considerazioni su Hardware e Dataset

-   **VRAM GPU:** Il `batch_size` è la manopola principale per controllare l'uso della memoria GPU. Inizia con un valore consigliato (ad esempio 16 o 32) e riducilo se incontri errori "CUDA out of memory" durante l'avvio dell'addestramento. Batch size più grandi generalmente portano a una convergenza più veloce ma richiedono più VRAM.
-   **Dimensione del Dataset vs. Epoche:**
    *   **Dataset Piccoli (< 20h):** Potrebbero richiedere meno epoche (ad esempio 300-1500) ma necessitano di un attento monitoraggio tramite validation loss/sample per evitare l'overfitting (dove il modello memorizza i dati di addestramento ma ha prestazioni scarse su nuovo testo). Considera learning rate più bassi.
    *   **Dataset Grandi (> 50h):** Possono beneficiare di più epoche (1000+) per apprendere completamente i pattern nei dati.
-   **CPU:** Mentre la GPU fa il lavoro pesante, è necessaria una CPU multi-core decente per il caricamento dei dati e il pre-processing, che altrimenti possono diventare un collo di bottiglia.
-   **Storage:** Assicurati di avere abbastanza spazio su disco per il dataset, l'ambiente Python, il codice del framework e specialmente i checkpoint salvati, che possono diventare grandi (da centinaia di MB a GB per checkpoint).

### 4.4. Strumenti di Monitoraggio (TensorBoard)

-   La maggior parte dei framework TTS moderni si integra con [TensorBoard](https://www.tensorflow.org/tensorboard) per visualizzare i progressi dell'addestramento.
-   Il file di configurazione spesso ha impostazioni relative al logging (ad esempio `use_tensorboard: True`, `log_directory`).
-   Durante l'addestramento, puoi tipicamente avviare TensorBoard eseguendo `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (ad esempio `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) in un terminale separato. Questo ti consente di monitorare le curve di loss, i learning rate e potenzialmente di ascoltare i sample di validazione sintetizzati nel tuo browser web.

---

Con il tuo ambiente configurato e il file di configurazione adattato ai tuoi dati e obiettivi, sei ora pronto ad avviare il vero e proprio processo di addestramento del modello.

**Passo Successivo:** [Addestramento del Modello](./3_MODEL_TRAINING.md){: .btn .btn-primary} | 
[Torna in Cima](#top){: .btn .btn-primary}
