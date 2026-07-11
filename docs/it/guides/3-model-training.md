# Guida all'addestramento e al fine-tuning di modelli TTS


Hai preparato i dati e l'ambiente. Ora è il momento di addestrare o rifinire il tuo modello TTS, seguirne i progressi e gestire i checkpoint in modo sicuro.

Se un termine relativo all'addestramento non è chiaro, consulta il [glossario](../glossary.md#glossary-of-technical-terms). Questa pagina spiega solo i termini che influenzano direttamente l'avvio, il monitoraggio o il debug dell'addestramento.

---

## Eseguire l'addestramento

Questa sezione spiega come avviare, monitorare e gestire l'addestramento.

### Avviare lo script di addestramento

-   **Vai nella directory corretta:** Apri il terminale e spostati nella cartella principale del framework TTS che contiene `train.py` o l'equivalente.
-   **Attiva l'ambiente virtuale:** Assicurati che sia attivo lo stesso ambiente Python usato nella preparazione.

    ```bash
    # Esempio di attivazione
    # Windows: ..\venv_tts\Scripts\activate
    # Linux/macOS: source ../venv_tts/bin/activate
    # Conda: conda activate tts_env
    ```

-   **Avvia l'addestramento con la tua config personalizzata:** la struttura esatta del comando dipende dal framework. Questi sono pattern comuni:

    ```bash
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Variante con nome del run
    # Verifica se -m sostituisce output_directory nella tua config.
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Variante con directory dei checkpoint indicata direttamente
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```

-   **Addestramento multi-GPU:** Se hai più GPU e il framework supporta l'addestramento distribuito, consulta la documentazione e puoi usare `torchrun`.

    ```bash
    # Esempio con torchrun; adatta nproc_per_node al numero di GPU
    torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
    ```

#### Primo controllo dell'avvio completo

Prima di lasciare attivo il lavoro di addestramento per ore o giorni, controlla nei primi minuti:

- lo script supera l'avvio e inizia a caricare batch reali
- la cartella di output inizia a ricevere log, checkpoint o file evento
- le loss appaiono come numeri normali, non come `NaN` o `inf`
- l'uso della memoria GPU si stabilizza invece di crescere fino al crash immediato

Se il lavoro fallisce qui, sistema prima questo. Cinque minuti iniziali non funzionanti diventano spesso una giornata persa.

### Monitorare i progressi

-   **Output in console:** Controlla:
    *   inizializzazione del modello e dei data loader
    *   epoch o step corrente
    *   `train_loss` e `validation_loss`
    *   learning rate
    *   tempo per step o epoch
-   **TensorBoard:** Se è attivo nella config, avvialo in un altro terminale.

    ```bash
    tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
    ```

    Apri l'URL mostrato da TensorBoard (di solito `http://localhost:6006/`). Qui puoi seguire le curve della loss, il learning rate e, se il framework li registra, i sample di validazione sintetizzati.

-   **Directory di output:** Dovrebbe contenere checkpoint (`.pth`, `.pt` o `.ckpt`), log, una copia della config, file evento di TensorBoard e a volte sample audio sintetizzati.

#### Come appare un buon progresso iniziale

In un primo run sano, di solito vuoi vedere:

- addestramento senza crash immediati, `NaN` o memoria fuori controllo
- `train_loss` e `validation_loss` che scendono nel tempo invece di esplodere
- sample di validazione sempre più chiari e stabili
- checkpoint successivi che suonano meglio dei primissimi, anche se ancora non perfetti

Non fissarti su un solo numero di loss. Nel TTS l'ascolto conta quanto le metriche.

### Capire i checkpoint

-   I checkpoint sono istantanee dello stato del modello e spesso anche dell'ottimizzatore.
-   Servono per:
    *   **riprendere un addestramento interrotto**
    *   **confrontare diverse fasi**
    *   **scegliere il modello migliore**
-   **Frequenza di salvataggio:** Imposta un `save_checkpoint_interval` ragionevole. Salvare troppo spesso consuma disco; salvare troppo raramente aumenta il rischio di perdere progressi utili. Molti framework salvano anche automaticamente il checkpoint «best» in base alla validation loss.

### Riprendere un addestramento interrotto

Se l'addestramento si interrompe in modo imprevisto, spesso puoi riprendere dall'ultimo checkpoint:

- Trova il checkpoint più recente nella directory di output, ad esempio `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` o `latest_checkpoint.pth`.
- Usa l'argomento di ripresa del framework; il nome può variare:

```bash
python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
```

Lo script dovrebbe caricare i pesi e lo stato dell'ottimizzatore e continuare dallo stesso punto.

### Quando fermare l'addestramento

-   **Limite di epoche:** l'addestramento si ferma quando raggiunge il numero massimo di `epochs` indicato nella config.
-   **Arresto anticipato:** se la validation loss smette di diminuire e aumenta in modo costante per diversi intervalli di validazione, il modello potrebbe iniziare a soffrire di overfitting.
-   **Valutazione tramite ascolto:** ascolta regolarmente i sample di validazione e ferma l'addestramento quando qualità e stabilità sono sufficienti per il tuo obiettivo, anche se la loss continua a diminuire leggermente.

A volte un checkpoint leggermente precedente suona meglio di quello con la loss minima assoluta.

---

## Fine-tuning rispetto ad addestramento da zero

### Scegliere l'approccio

Quando inizi un progetto TTS, una delle decisioni più importanti è scegliere se fare fine-tuning di un modello esistente o addestrarne uno nuovo da zero. Questa tabella aiuta a capire quale approccio è più adatto al tuo caso:

| Fattore | Fine-tuning | Addestramento da zero |
|:--------|:------------|:----------------------|
| **Dimensione del dataset** | Funziona bene con dataset più piccoli (5-20 ore)<br>Per alcune voci può dare risultati utili anche con 1-2 ore | In genere richiede dataset più grandi (30+ ore)<br>Sotto le 20 ore la qualità è spesso peggiore |
| **Somiglianza della voce** | Funziona meglio se la voce target è simile a quelle del modello base | È preferibile se la voce target è molto particolare o molto diversa |
| **Lingua** | Funziona bene nella stessa lingua<br>Può anche funzionare tra lingue con una preparazione accurata | Serve quando non esistono buoni modelli base per quella lingua<br>Cattura meglio la fonetica specifica |
| **Tempo di addestramento** | Molto più rapido (giorni invece di settimane)<br>Richiede meno epoche per convergere | Richiede molto più tempo<br>Può servire da 2 a 5 volte più epoche |
| **Requisiti hardware** | Richiede GPU simili ma per meno tempo<br>Spesso tollera batch size più piccoli | Richiede accesso continuativo alla GPU per più tempo<br>Trae maggior beneficio da setup multi-GPU |
| **Potenziale di qualità** | Può raggiungere rapidamente un'ottima qualità<br>Può ereditare i limiti del modello base | Offre massima flessibilità e potenziale qualitativo<br>Non dipende da vincoli di addestramenti precedenti |
| **Stabilità** | In genere è più stabile<br>È meno soggetto a collassi o mancata convergenza | È più sensibile agli iperparametri<br>Ha un rischio più alto di instabilità |

#### Quando scegliere il fine-tuning

- Hai pochi dati
- Ti servono risultati più in fretta
- La tua voce o lingua è simile a un modello pre-addestrato disponibile
- Hai risorse computazionali limitate
- Sei nuovo nell'addestramento TTS, perché il fine-tuning è di solito più tollerante

#### Quando scegliere l'addestramento da zero

- Hai molti dati (30+ ore)
- La voce target è molto diversa o ha caratteristiche poco rappresentate nei modelli base
- Non esiste un buon modello base per la tua lingua
- Hai bisogno del massimo controllo su tutti gli aspetti del modello
- Hai accesso a risorse computazionali importanti
- Stai costruendo un modello base che altri potranno poi rifinire

### Particolarità del fine-tuning

Il fine-tuning sfrutta un modello base potente e lo adatta al tuo dataset specifico, che si tratti di una voce, di una lingua o di uno stile. In genere è più rapido e richiede meno dati rispetto all'addestramento da zero.

#### L'obiettivo

- Trasferire le capacità generali di sintesi vocale del modello base, come la relazione testo-audio e una prosodia di base, adattando allo stesso tempo l'identità della voce e, in certi casi, accento o stile al tuo dataset più piccolo.

### Differenze chiave nella configurazione

- **`pretrained_model_path`:** Devi indicare esplicitamente il percorso del checkpoint del modello pre-addestrato nella configurazione.
- **`fine_tuning: True`:** Attiva qualsiasi flag di modalità fine-tuning se il framework lo richiede.
- **Learning rate:** Parti con un learning rate più basso di quello usato per addestrare da zero, ad esempio `1e-5`, `2e-5` o `5e-5`. Un learning rate troppo alto può distruggere informazioni utili del modello base.
- **Batch size:** Spesso può restare simile a quello dell'addestramento da zero, regolato in base alla VRAM disponibile.
- **Epoche:** Di solito servono meno epoche rispetto all'addestramento da zero, ma dipende comunque dalla dimensione del dataset e dalla qualità desiderata. Osserva con attenzione validation loss e campioni audio.

### Strategie di fine-tuning

- **Fine-tuning completo della rete:** L'approccio più comune consiste nell'aggiornare tutta la rete con un learning rate basso.
- **Congelare layer:** Alcuni framework permettono di congelare inizialmente parti della rete, come il text encoder o il duration predictor, e addestrare solo componenti specifici. Consulta la documentazione per opzioni come `--freeze_layers`.
- **Ignorare o reinizializzare layer:** Durante il caricamento del modello pre-addestrato, può essere utile usare `ignore_layers` o `reinitialize_layers` per layer come l'output finale o gli speaker embeddings, soprattutto se il dataset ha un numero diverso di speaker.

### Cosa monitorare durante il fine-tuning

- **Miglioramento rapido all'inizio:** La validation loss dovrebbe scendere abbastanza rapidamente nelle prime fasi.
- **Qualità percepita:** Ascolta i campioni generati. La voce dovrebbe avvicinarsi allo speaker target senza perdere chiarezza o stabilità.
- **Stabilità:** Controlla se compaiono artefatti, ripetizioni o degradazione continuando l'addestramento.

Il fine-tuning riguarda spesso più la qualità percepita che la ricerca della loss minima assoluta.

---

## Guida ai problemi durante l'addestramento

Addestrare modelli TTS può essere complesso e causare diversi problemi. Le sezioni seguenti offrono indicazioni per i casi più comuni.

### Errori comuni e cosa indicano spesso

| Errore | Possibili cause | Soluzioni |
|:-------|:---------------|:----------|
| `CUDA out of memory` | Batch size troppo grande<br>Modello troppo pesante per la GPU<br>Perdita o pressione di memoria | Riduci batch size<br>Attiva gradient checkpointing<br>Usa mixed precision<br>Riduci la lunghezza delle sequenze |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | Tipo di dato errato nel dataset<br>Tensori incompatibili | Controlla il preprocessing<br>Assicurati che tutti i tensori abbiano il dtype corretto<br>Aggiungi conversioni di tipo esplicite |
| `ValueError: too many values to unpack` | Disallineamento tra output del modello e loss<br>Formato dati errato | Controlla la struttura degli output del modello<br>Verifica l'implementazione della loss<br>Debugga l'output del data loader |
| `FileNotFoundError: [Errno 2] No such file or directory` | Percorsi errati nella configurazione<br>File dati mancanti | Verifica tutti i percorsi<br>Controlla l'integrità dei manifest<br>Assicurati che tutti i dati siano stati scaricati o estratti |
| `KeyError: 'speaker_id'` | Informazioni speaker mancanti<br>Formato dataset errato | Controlla il formato del dataset<br>Verifica il file di mapping degli speaker<br>Aggiungi le informazioni speaker al manifest |
| `Loss is NaN` | Learning rate troppo alto<br>Inizializzazione instabile<br>Esplosione dei gradienti | Abbassa il learning rate<br>Aggiungi gradient clipping<br>Controlla divisioni per zero<br>Normalizza i dati in ingresso |
| `ModuleNotFoundError: No module named 'X'` | Dipendenza mancante<br>Problema di ambiente | Installa il pacchetto mancante<br>Controlla l'ambiente virtuale<br>Verifica le versioni dei pacchetti |
| `RuntimeError: expected scalar type Float but found Double` | Tipi di tensore incoerenti | Aggiungi `.float()` ai tensori<br>Controlla il preprocessing<br>Uniforma il dtype in tutto il modello |

### Problemi di qualità

| Sintomo | Possibili cause | Soluzioni |
|:--------|:----------------|:----------|
| **Audio robotico o ronzante** | Problemi di vocoder<br>Addestramento insufficiente<br>Preprocessing audio debole | Addestra più a lungo il vocoder<br>Controlla la normalizzazione audio<br>Verifica la coerenza del sampling rate |
| **Parole saltate o ripetute** | Problemi di attenzione<br>Addestramento instabile<br>Dati insufficienti | Usa guided attention loss<br>Aggiungi più varietà nei dati<br>Abbassa il learning rate<br>Cerca silenzi lunghi nel dataset |
| **Pronuncia errata** | Problemi di normalizzazione del testo<br>Errori fonemici<br>Disallineamento di lingua | Migliora il preprocessing del testo<br>Usa input basato su fonemi<br>Aggiungi un dizionario di pronuncia |
| **Perdita di identità dello speaker** | Overfitting sullo speaker dominante<br>Speaker embeddings deboli<br>Pochi dati per speaker | Bilancia i dati degli speaker<br>Aumenta la dimensione dello speaker embedding<br>Rivedi la strategia multi-speaker |
| **Convergenza lenta** | Problemi di learning rate<br>Cattiva inizializzazione<br>Dataset complesso | Prova altre strategie di learning rate<br>Usa transfer learning<br>Semplifica il dataset all'inizio |
| **Addestramento instabile** | Alta varianza tra batch<br>Outlier nel dataset<br>Problemi di optimizer | Usa gradient accumulation<br>Pulisci i campioni anomali<br>Prova un optimizer diverso |

### Problemi di framework e ambiente

#### Coqui TTS

```bash
# Error: "RuntimeError: Error in applying gradient to param_name"
# Soluzione: cerca valori NaN nel dataset o riduci il learning rate
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # Esegui questo prima dell'addestramento per fare debug
```

```bash
# Error: "ValueError: Tacotron training requires `r` > 1"
# Soluzione: imposta correttamente il reduction factor nella config
# Esempio di correzione in config.json:
"r": 2  # Prova valori tra 2 e 5
```

#### ESPnet

```bash
# Error: "TypeError: forward() missing 1 required positional argument: 'feats'"
# Soluzione: controlla il formato dei dati e assicurati che feats venga fornito
# Debug del caricamento dati:
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS / StyleTTS

```python
# Error: "RuntimeError: expected scalar type Half but found Float"
# Soluzione: mantieni una precisione coerente in tutto il modello
# Aggiungi allo script di addestramento:
model = model.half()  # Se usi mixed precision
# OPPURE
model = model.float()  # Se non usi mixed precision
```

### Problemi hardware e di ambiente

1. **Frammentazione della memoria GPU**
   - **Sintomo:** errori OOM dopo molte ore anche se la VRAM dovrebbe bastare
   - **Soluzione:** riavvia periodicamente l'addestramento da checkpoint e prova batch più piccoli

2. **Collo di bottiglia della CPU**
   - **Sintomo:** uso della GPU basso o molto irregolare
   - **Soluzione:** aumenta `num_workers` nel DataLoader, usa storage più veloce e precarica i dataset se possibile

3. **Collo di bottiglia del disco / I/O**
   - **Sintomo:** pause periodiche durante il caricamento dei dati
   - **Soluzione:** usa SSD, aumenta il prefetch factor o cachea il dataset in RAM

4. **Conflitti di ambiente**
   - **Sintomo:** crash strani o errori di importazione difficili da spiegare
   - **Soluzione:** usa ambienti isolati, verifica la compatibilità CUDA/PyTorch ed evita di mescolare installazioni vecchie

5. **Attiva il logging dettagliato:** se serve, aggiungi questo allo script di addestramento:

   ```python
   # Da aggiungere allo script di addestramento
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

6. **Usa il profiling di TensorBoard:** profila il tempo CPU/GPU per individuare i colli di bottiglia:

   ```python
   # Da aggiungere al codice di addestramento
   from torch.profiler import ProfilerActivity, profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Il tuo forward pass
           pass
   print(prof.key_averages().table())
   ```

### Strategie di debug

1. **Isola il problema**

   ```bash
   # Testare il caricamento dei dati separatamente
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"

   # Testare un forward pass con dati fittizi
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Semplifica per trovare la causa**
   - Addestra su un sottoinsieme molto piccolo e pulito
   - Disattiva temporaneamente augmentation
   - Usa una configurazione più leggera se il framework lo permette

3. **Controlla gli artefatti intermedi**
   - Guarda attention alignments, mel spectrograms, log e campioni di validazione
   - Verifica se il problema compare dall'inizio o solo dopo alcuni checkpoint

4. **Aggiungi più visibilità**
   - Attiva logging più dettagliato se disponibile
   - Salva più campioni intermedi
   - Usa `torch.autograd.set_detect_anomaly(True)` solo durante il debug, non come impostazione permanente

---

Una volta che l'addestramento è partito ed è sotto controllo, il passo successivo è scegliere un buon checkpoint e usare il modello per generare parlato da nuovo testo.
