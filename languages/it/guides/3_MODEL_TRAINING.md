# Guida 3: Addestramento e Fine-tuning del Modello

**Navigazione:** [README Principale]({{ site.baseurl }}/languages/it/){: .btn .btn-primary} | [Passo Precedente: Configurazione dell'Addestramento](./2_TRAINING_SETUP.md){: .btn .btn-primary} | [Passo Successivo: Inferenza](./4_INFERENCE.md){: .btn .btn-primary}

Hai preparato i tuoi dati e configurato il tuo ambiente di addestramento. Ora è il momento di addestrare effettivamente (o effettuare il fine-tuning) del tuo modello Text-to-Speech. Questa fase comporta l'esecuzione dello script di addestramento, il monitoraggio dei suoi progressi e la comprensione di come gestire il processo.

---

## 5. Esecuzione dell'Addestramento

Questa sezione descrive in dettaglio come avviare, monitorare e gestire il processo di addestramento.

### 5.1. Avviare lo Script di Addestramento

-   **Naviga nella Directory Corretta:** Apri il tuo terminale o prompt dei comandi e naviga nella directory principale del repository del framework TTS clonato (la directory contenente lo script `train.py` o il suo equivalente).
-   **Attiva l'Ambiente Virtuale:** Assicurati che il tuo ambiente virtuale Python dedicato (ad esempio `venv_tts`, `tts_env`) sia attivato.
    ```bash
    # Esempio di attivazione (adatta percorso/nome secondo necessità)
    # Windows: ..\venv_tts\Scripts\activate
    # Linux/macOS: source ../venv_tts/bin/activate
    # Conda: conda activate tts_env
    ```
-   **Esegui il Comando di Addestramento:** Esegui lo script di addestramento del framework, indirizzandolo al tuo file di configurazione personalizzato creato nella Guida 2. La struttura esatta del comando varia tra i framework. I pattern comuni includono:
    ```bash
    # Pattern Comune 1: Usando l'argomento --config
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Pattern Comune 2: Usando -c per la config e -m per il nome della directory del modello/output
    # (Verifica se la tua output_directory nella config viene sovrascritta da -m)
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Pattern Comune 3: Specificando direttamente la directory dei checkpoint
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```
    *   **Addestramento Multi-GPU:** Se hai più GPU e il framework supporta l'addestramento distribuito (controlla la sua documentazione), potresti usare comandi che coinvolgono `torchrun` o `python -m torch.distributed.launch`. Esempio:
        ```bash
        # Esempio con torchrun (regola nproc_per_node in base al tuo numero di GPU)
        torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
        ```

### 5.2. Monitorare i Progressi dell'Addestramento

-   **Output della Console:** Il terminale da cui hai avviato l'addestramento mostrerà informazioni sui progressi. Cerca:
    *   **Inizializzazione:** Messaggi che indicano che il modello viene costruito, i data loader vengono preparati e potenzialmente un modello pre-addestrato viene caricato (per il fine-tuning).
    *   **Epoche/Step:** Progressi attuali dell'addestramento (ad esempio `Epoch: [1/1000]`, `Step: [500/100000]`).
    *   **Valori di Loss:** Metriche cruciali che indicano quanto bene il modello sta apprendendo. Aspettati di vedere `train_loss` (loss sul batch corrente) e, periodicamente, `validation_loss` (loss sul set di validazione non visto). Entrambi dovrebbero generalmente diminuire nel tempo. Componenti specifici della loss (come `mel_loss`, `duration_loss`, `kl_loss`) potrebbero anche essere riportati a seconda dell'architettura del modello.
    *   **Learning Rate:** Il learning rate corrente potrebbe essere stampato, specialmente se uno scheduler lo sta riducendo nel tempo.
    *   **Timestamp/Velocità:** Tempo impiegato per step o epoca.
-   **TensorBoard (Altamente Consigliato):** Se abilitato nella tua config, usa TensorBoard per il monitoraggio visivo.
    *   **Avvio:** Apri un *nuovo* terminale (mantieni in esecuzione quello dell'addestramento), attiva lo stesso ambiente virtuale ed esegui:
        ```bash
        # Punta logdir alla output_directory specificata nella tua config
        tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
        ```
    *   **Accesso:** Apri l'URL fornito da TensorBoard (di solito `http://localhost:6006/`) nel tuo browser web.
    *   **Visualizza:** Puoi vedere i grafici delle loss di addestramento e validazione nel tempo, gli schedule del learning rate e potenzialmente altre metriche.
    *   **Ascolta i Sample Audio:** Molti framework registrano periodicamente sample audio sintetizzati dal set di validazione su TensorBoard (controlla la scheda `AUDIO`). Ascoltarli è il modo *migliore* per valutare qualitativamente il miglioramento del modello e identificare problemi come rumore, errori di pronuncia o output robotico.
-   **Directory di Output:** Controlla la `output_directory` specificata nella tua config (`../checkpoints/my_yoruba_voice_run1`). Dovrebbe contenere:
    *   Checkpoint del modello salvati (file `.pth`, `.pt`, `.ckpt`).
    *   File di log (`train.log`, ecc.).
    *   Copie del file di configurazione utilizzato.
    *   File di evento TensorBoard (di solito in una sottodirectory `logs` o `events`).
    *   Possibilmente sample audio sintetizzati.

### 5.3. Comprendere i Checkpoint

-   **Cosa sono:** I checkpoint sono istantanee dello stato del modello (tutti i suoi pesi appresi e potenzialmente lo stato dell'optimizer) salvate a intervalli specifici durante l'addestramento.
-   **Perché sono importanti:**
    *   **Riprendere l'Addestramento:** Consentono di continuare l'addestramento se viene interrotto (a causa di crash, interruzioni di corrente o arresto manuale).
    *   **Valutare i Progressi:** Puoi usare i checkpoint di diverse fasi per sintetizzare audio e vedere come il modello è evoluto.
    *   **Selezionare il Modello Migliore:** La validation loss aiuta a identificare i buoni checkpoint, ma il modello *migliore* viene spesso scelto ascoltando l'audio sintetizzato da diversi checkpoint promettenti vicini alla validation loss più bassa. A volte un checkpoint leggermente precedente suona meglio di quello con la loss in assoluto più bassa.
-   **Frequenza di Salvataggio:** Configura il `save_checkpoint_interval` nella tua config. Salvare troppo spesso consuma spazio su disco; salvare troppo raramente rischia di perdere progressi significativi se si verifica un crash. Salvare ogni qualche migliaio di step o una volta per epoca è comune. Molti framework salvano anche automaticamente il checkpoint "migliore" in base alla validation loss.

### 5.4. Riprendere un Addestramento Interrotto

-   Se il tuo addestramento si arresta inaspettatamente o lo fermi manualmente, di solito puoi riprendere dall'ultimo checkpoint salvato.
-   Trova il percorso dell'ultimo file di checkpoint nella tua directory di output (ad esempio `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` o `latest_checkpoint.pth`).
-   Usa l'argomento di ripresa del framework quando avvii nuovamente lo script di addestramento. Il nome dell'argomento varia:
    ```bash
    # Esempio con --resume_checkpoint
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

    # Esempio con --restore_path o --resume_path
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
    ```
-   Lo script dovrebbe caricare i pesi del modello e lo stato dell'optimizer dal checkpoint e continuare l'addestramento da quel punto.

### 5.5. Quando Interrompere l'Addestramento

-   **Limite di Epoche:** L'addestramento si arresta automaticamente quando viene raggiunto il numero massimo di `epochs` specificato nella config.
-   **Early Stopping:** Monitora la `validation_loss`. Se smette di diminuire e inizia ad aumentare costantemente per un periodo prolungato (ad esempio, su diversi intervalli di validazione), il modello potrebbe iniziare a fare overfitting. Potresti considerare di interrompere l'addestramento manualmente intorno al punto in cui la validation loss era più bassa.
-   **Valutazione Qualitativa:** Ascolta regolarmente i sample audio di validazione generati in TensorBoard o sintetizza manualmente sample usando checkpoint recenti. Interrompi l'addestramento quando sei soddisfatto della qualità e della stabilità dell'audio, anche se la loss sta ancora diminuendo leggermente. Un ulteriore addestramento potrebbe non produrre miglioramenti percepibili o potrebbe persino degradare la qualità.

---

## 6. Fine-tuning vs. Addestramento da Zero

### 6.1. Scegliere il Tuo Approccio

Quando avvii un progetto TTS, una delle decisioni più importanti è se effettuare il fine-tuning di un modello esistente o addestrarne uno nuovo da zero. Questa tabella ti aiuta a decidere quale approccio è il migliore per la tua situazione specifica:

| Fattore | Fine-tuning | Addestramento da Zero |
|:-------|:------------|:----------------------|
| **Dimensione del Dataset** | Funziona bene con dataset più piccoli (5-20 ore)<br>Può produrre buoni risultati anche con solo 1-2 ore per alcune voci | Tipicamente richiede dataset più grandi (30+ ore)<br>Meno di 20 ore spesso porta a una qualità scadente |
| **Somiglianza Vocale** | Migliore quando la tua voce target è simile alle voci nei dati di addestramento del modello pre-addestrato | Necessario quando la tua voce target è molto unica o significativamente diversa dai modelli pre-addestrati disponibili |
| **Lingua** | Funziona bene se si effettua il fine-tuning nella stessa lingua<br>Può funzionare per il cross-linguale con un'attenta preparazione | Richiesto per lingue senza modelli pre-addestrati disponibili<br>Migliore per catturare la fonetica specifica della lingua |
| **Tempo di Addestramento** | Molto più veloce (giorni invece di settimane)<br>Richiede meno epoche per convergere | Tempo di addestramento significativamente più lungo<br>Può richiedere da 2 a 5 volte più epoche |
| **Requisiti Hardware** | Requisiti GPU simili ma per meno tempo<br>Può spesso usare batch size più piccoli | Necessita di accesso GPU sostenuto per periodi più lunghi<br>Può beneficiare maggiormente di setup multi-GPU |
| **Potenziale di Qualità** | Può raggiungere rapidamente un'eccellente qualità<br>Può ereditare i limiti del modello base | Massima flessibilità e qualità potenziale<br>Nessun vincolo da addestramenti precedenti |
| **Stabilità** | Processo di addestramento generalmente più stabile<br>Meno soggetto a collasso o non-convergenza | Più sensibile agli iperparametri<br>Rischio più elevato di instabilità dell'addestramento |

#### Quando Scegliere il Fine-tuning

Il fine-tuning è generalmente consigliato quando:
- Hai dati limitati (meno di 20 ore)
- Hai bisogno di risultati più rapidi
- La tua voce/lingua target è ragionevolmente simile ai modelli pre-addestrati disponibili
- Hai risorse computazionali limitate
- Sei nuovo all'addestramento TTS (il fine-tuning è più indulgente)

#### Quando Scegliere l'Addestramento da Zero

L'addestramento da zero è migliore quando:
- Hai dati abbondanti (30+ ore)
- La tua voce target è altamente unica o ha caratteristiche non rappresentate nei modelli pre-addestrati
- Stai lavorando con una lingua scarsamente supportata dai modelli esistenti
- Hai bisogno del massimo controllo su tutti gli aspetti del modello
- Hai accesso a risorse computazionali significative
- Stai costruendo un foundation model su cui altri effettueranno il fine-tuning

### 6.2. Specifiche del Fine-tuning

Il fine-tuning sfrutta un potente modello pre-addestrato e lo adatta al tuo dataset specifico (speaker, lingua, stile). È solitamente più veloce e richiede meno dati rispetto all'addestramento da zero.

#### L'Obiettivo

-   Trasferire le capacità generali di sintesi vocale (come la comprensione della mappatura testo-suono, la prosodia di base) dal grande dataset su cui è stato addestrato il modello base, specializzando al contempo l'identità vocale e potenzialmente l'accento/stile per corrispondere al tuo dataset più piccolo e specifico.

### 6.2. Differenze Chiave di Configurazione (Riepilogo dal Setup)

-   **`pretrained_model_path`:** DEVI fornire il percorso al file di checkpoint del modello pre-addestrato nella tua configurazione.
-   **`fine_tuning: True`:** Assicurati che qualsiasi flag che indichi la modalità di fine-tuning sia abilitato se il framework lo richiede.
-   **Learning Rate:** Inizia con un learning rate *più basso* di quello tipicamente usato per l'addestramento da zero (ad esempio `1e-5`, `2e-5`, `5e-5`). Un learning rate elevato può distruggere le preziose informazioni apprese dal modello pre-addestrato.
-   **Batch Size:** Può spesso essere simile all'addestramento da zero, regola in base alla VRAM.
-   **Epoche:** Il numero di epoche richiesto per il fine-tuning è solitamente significativamente inferiore rispetto all'addestramento da zero, ma dipende comunque dalla dimensione del dataset e dalla qualità desiderata. Monitora attentamente la validation loss e i sample audio.

### 6.3. Possibili Strategie (Dipendenti dal Framework)

-   **Fine-tuning dell'Intera Rete:** L'approccio predefinito è spesso aggiornare i pesi attraverso l'intera rete, ma con un learning rate basso.
-   **Congelare i Layer:** Alcuni framework consentono di congelare parti della rete (ad esempio il text encoder o il duration predictor) inizialmente e di addestrare solo componenti specifici (come gli speaker embedding o il decoder). Questo può talvolta aiutare a preservare i punti di forza del modello base adattando al contempo aspetti specifici. Controlla la documentazione del tuo framework per opzioni `--freeze_layers` o simili.
-   **Ignorare i Layer:** Quando carichi il modello pre-addestrato, potresti voler usare `ignore_layers` (o `reinitialize_layers`) come il layer di output finale o il layer di speaker embedding, specialmente se il tuo dataset ha un numero di speaker diverso dal modello pre-addestrato.

### 6.4. Monitorare il Fine-tuning

-   **Rapido Miglioramento Iniziale:** Dovresti vedere la validation loss calare relativamente in fretta all'inizio, man mano che il modello si adatta alla voce target.
-   **Concentrati sulla Qualità Audio:** Presta molta attenzione ai sample di validazione sintetizzati. L'identità vocale si sta spostando verso il tuo speaker target? Il parlato è chiaro e stabile? Il fine-tuning riguarda spesso più la qualità percettiva che il raggiungimento del valore di loss minimo assoluto.

---

## 7. Guida Completa alla Risoluzione dei Problemi

Addestrare modelli TTS può essere impegnativo, con molti potenziali problemi. Questa sezione fornisce soluzioni per i problemi comuni che potresti incontrare.

### 7.1. Messaggi di Errore Comuni e Soluzioni

| Messaggio di Errore | Possibili Cause | Soluzioni |
|:--------------|:----------------|:----------|
| `CUDA out of memory` | • Batch size troppo grande<br>• Modello troppo grande per la GPU<br>• Memory leak | • Riduci il batch size<br>• Abilita il gradient checkpointing<br>• Usa il mixed precision training<br>• Riduci la lunghezza della sequenza |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | • Tipo di dato errato nel dataset<br>• Tipi di tensore incompatibili | • Controlla il preprocessing dei dati<br>• Assicurati che tutti i tensori abbiano il dtype corretto<br>• Aggiungi una conversione esplicita del tipo |
| `ValueError: too many values to unpack` | • Discrepanza tra gli output del modello e le aspettative della loss function<br>• Formato dei dati errato | • Controlla la struttura dell'output del modello<br>• Verifica l'implementazione della loss function<br>• Esegui il debug degli output del data loader |
| `FileNotFoundError: [Errno 2] No such file or directory` | • Percorsi errati nella config<br>• File di dati mancanti | • Verifica tutti i percorsi dei file<br>• Controlla l'integrità del file manifest<br>• Assicurati che i dati siano scaricati/estratti |
| `KeyError: 'speaker_id'` | • Informazioni sullo speaker mancanti<br>• Formato del dataset errato | • Controlla il formato del dataset<br>• Verifica il file di mapping dello speaker<br>• Aggiungi le informazioni sullo speaker al manifest |
| `Loss is NaN` | • Learning rate troppo alto<br>• Inizializzazione instabile<br>• Esplosione del gradiente | • Riduci il learning rate<br>• Aggiungi il gradient clipping<br>• Controlla la divisione per zero<br>• Normalizza i dati di input |
| `ModuleNotFoundError: No module named 'X'` | • Dipendenza mancante<br>• Problema dell'ambiente | • Installa il pacchetto mancante<br>• Controlla l'ambiente virtuale<br>• Verifica le versioni dei pacchetti |
| `RuntimeError: expected scalar type Float but found Double` | • Tipi di tensore incoerenti | • Aggiungi `.float()` ai tensori<br>• Controlla il preprocessing dei dati<br>• Standardizza il dtype in tutto il modello |

### 7.2. Problemi di Qualità dell'Addestramento

| Sintomo | Possibili Cause | Soluzioni |
|:--------|:----------------|:----------|
| **Audio Robotico/Ronzante** | • Problemi del vocoder<br>• Addestramento insufficiente<br>• Preprocessing audio scadente | • Addestra il vocoder più a lungo<br>• Controlla la normalizzazione audio<br>• Verifica la coerenza del sampling rate |
| **Salto/Ripetizione di Parole** | • Problemi di attention<br>• Addestramento instabile<br>• Dati insufficienti | • Usa la guided attention loss<br>• Aggiungi più varietà di dati<br>• Riduci il learning rate<br>• Controlla i silenzi lunghi nei dati |
| **Pronuncia Errata** | • Problemi di normalizzazione del testo<br>• Errori fonemici<br>• Discrepanza linguistica | • Migliora il preprocessing del testo<br>• Usa input basato sui fonemi<br>• Aggiungi un dizionario di pronuncia |
| **Perdita dell'Identità dello Speaker** | • Overfitting verso lo speaker dominante<br>• Speaker embedding deboli<br>• Dati insufficienti sullo speaker | • Bilancia i dati degli speaker<br>• Aumenta la dimensione dello speaker embedding<br>• Usa la speaker adversarial loss |
| **Convergenza Lenta** | • Problemi di learning rate<br>• Inizializzazione scadente<br>• Dataset complesso | • Prova diversi schedule del LR<br>• Usa il transfer learning<br>• Semplifica inizialmente il dataset |
| **Addestramento Instabile** | • Varianza del batch<br>• Valori anomali nel dataset<br>• Problemi dell'optimizer | • Usa la gradient accumulation<br>• Pulisci i sample anomali<br>• Prova diversi optimizer |

### 7.3. Problemi Specifici del Framework

#### Coqui TTS
```
# Error: "RuntimeError: Error in applying gradient to param_name"
# Solution: Check for NaN values in your dataset or reduce learning rate
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # Run before training to debug

# Error: "ValueError: Tacotron training requires `r` > 1"
# Solution: Set reduction factor correctly in config
# Example fix in config.json:
"r": 2  # Try values between 2-5
```

#### ESPnet
```
# Error: "TypeError: forward() missing 1 required positional argument: 'feats'"
# Solution: Check data formatting and ensure feats are provided
# Debug data loading:
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS/StyleTTS
```
# Error: "RuntimeError: expected scalar type Half but found Float"
# Solution: Ensure consistent precision throughout model
# Add to your training script:
model = model.half()  # If using mixed precision
# OR
model = model.float()  # If not using mixed precision
```

### 7.4. Problemi di Hardware e Ambiente

1. **Frammentazione della Memoria GPU**
   - **Sintomo**: Errori OOM dopo diverse ore di addestramento nonostante VRAM sufficiente
   - **Soluzione**: Riavvia periodicamente l'addestramento dal checkpoint, usa batch più piccoli

2. **Colli di Bottiglia della CPU**
   - **Sintomo**: L'utilizzo della GPU fluttua o rimane basso
   - **Soluzione**: Aumenta num_workers nel DataLoader, usa storage più veloce, pre-memorizza i dataset nella cache

3. **Colli di Bottiglia dell'I/O su Disco**
   - **Sintomo**: L'addestramento si blocca periodicamente durante il caricamento dei dati
   - **Soluzione**: Usa storage SSD, aumenta il prefetch factor, memorizza il dataset in RAM

4. **Conflitti di Ambiente**
   - **Sintomo**: Crash misteriosi o errori di import
   - **Soluzione**: Usa ambienti isolati (conda/venv), controlla la compatibilità CUDA/PyTorch

### 7.5. Strategie di Debugging

1. **Isola il Problema**
   ```bash
   # Testa il caricamento dei dati separatamente
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"
   
   # Testa il forward pass con dati fittizi
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Semplifica per Identificare i Problemi**
   - Addestra su un piccolo sottoinsieme (10-20 sample)
   - Disabilita temporaneamente la data augmentation
   - Prova prima con un singolo speaker

3. **Visualizza gli Output Intermedi**
   - Traccia gli allineamenti di attention
   - Visualizza i mel spectrogram nelle diverse fasi
   - Monitora le norme dei gradienti

4. **Abilita il Logging Verboso**
   ```bash
   # Aggiungi al tuo script di addestramento
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

5. **Usa il Profiling di TensorBoard**
   ```python
   # Aggiungi al tuo codice di addestramento
   from torch.profiler import profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Il tuo forward pass
   print(prof.key_averages().table())
   ```

---

Con l'addestramento avviato e monitorato, il passo successivo, dopo aver selezionato un buon checkpoint, è usare il modello per generare il parlato su nuovo testo.

**Passo Successivo:** [Inferenza](./4_INFERENCE.md){: .btn .btn-primary} | 
[Torna in Cima](#top){: .btn .btn-primary}
