# Guida Universale all'Addestramento di Modelli TTS e alla Preparazione dei Dataset

## Lingue Disponibili

- [English](../../README.md) (Originale)
- [Español](../es/README.md)
- [Français](../fr/README.md)
- **Italiano** (Corrente)
- [Български](../bg/README.md)

*Vuoi contribuire con una traduzione? Consulta la [Guida alla Traduzione](../../README.md#translation-guide) qui sotto.*

## Introduzione

Benvenuto! Questa guida completa fornisce un processo universale per preparare i tuoi dataset vocali e addestrare un modello Text-to-Speech (TTS) personalizzato. Che tu abbia un piccolo dataset (ad esempio 10 ore) o uno più grande (oltre 100 ore), questi passaggi ti aiuteranno a organizzare i dati correttamente e a navigare il processo di addestramento per la maggior parte dei framework TTS moderni.

**Obiettivo:** Metterti in grado di effettuare il fine-tuning o di addestrare un modello TTS su una voce o lingua specifica utilizzando le tue coppie audio-testo.

**Cosa Copre Questa Guida:**
Questa guida è suddivisa in diverse parti, che coprono l'intero flusso di lavoro dalla pianificazione all'utilizzo del modello addestrato:

1.  **Pianificazione:** Considerazioni iniziali prima di avviare il tuo progetto.
2.  **Preparazione dei Dati:** Acquisizione, elaborazione e strutturazione dei dati audio e testuali.
3.  **Configurazione dell'Addestramento:** Preparazione dell'ambiente e configurazione dei parametri di addestramento.
4.  **Addestramento del Modello:** Avvio, monitoraggio e fine-tuning del modello TTS.
5.  **Inferenza:** Utilizzo del modello addestrato per sintetizzare il parlato.
6.  **Packaging e Condivisione:** Organizzazione e documentazione del modello per usi futuri o distribuzione.
7.  **Risoluzione dei Problemi e Risorse:** Problemi comuni e strumenti utili.

---

## 0. Prima di Iniziare: Pianificare il Tuo Dataset

Prima di raccogliere i dati, considera questi punti cruciali per assicurarti che il tuo progetto sia ben definito e realizzabile:

1.  **Speaker:** Sarà un singolo speaker o più speaker? I dataset a singolo speaker sono più semplici da cui partire per il fine-tuning o l'addestramento iniziale. I modelli multi-speaker richiedono un attento bilanciamento dei dati e una gestione degli speaker ID.
2.  **Fonte dei Dati:** Da dove otterrai l'audio? (Audiolibri, podcast, archivi radiofonici, dati vocali registrati professionalmente, le tue registrazioni). **In modo cruciale, assicurati di avere i diritti o le licenze necessarie per utilizzare i dati nell'addestramento dei modelli.**
3.  **Qualità Audio:** Punta alla qualità più alta possibile. Dai priorità a registrazioni pulite con il minimo di rumore di fondo, riverbero, musica o parlato sovrapposto. La coerenza nelle condizioni di registrazione è molto vantaggiosa.
4.  **Lingua e Dominio:** Quale lingua o quali lingue parlerà il modello? Qual è lo stile di parlato o il dominio (ad esempio narrazione, conversazione, lettura di notizie)? Il modello darà i risultati migliori su testi simili ai suoi dati di addestramento.
5.  **Quantità di Dati Target:** Quanti dati prevedi di raccogliere o utilizzare?
    *   **~1-5 ore:** Potrebbe essere sufficiente per il *cloning* vocale di base se si utilizza un modello pre-addestrato robusto, ma la qualità potrebbe essere limitata.
    *   **~5-20 ore:** Generalmente considerato il minimo per un *fine-tuning* decente di una voce specifica su un modello pre-addestrato.
    *   **50-100+ ore:** Migliore per addestrare modelli robusti o per addestrare modelli con minore dipendenza dai pesi pre-addestrati, specialmente per lingue meno comuni.
    *   **1000+ ore:** Necessarie per addestrare modelli di alta qualità e di uso generale, in gran parte da zero.
6.  **Sampling Rate:** Decidi in anticipo un sampling rate target (ad esempio 16000 Hz, 22050 Hz, 44100 Hz, 48000 Hz). Sampling rate più alti catturano più dettagli ma richiedono più spazio di archiviazione e potenza di calcolo. **Tutti i tuoi dati di addestramento DEVONO utilizzare in modo coerente il rate scelto.** 22050 Hz è un compromesso comune per molti modelli TTS.

---

## Panoramica del Processo e Navigazione

Questa guida è suddivisa in moduli mirati. Segui i link qui sotto per i passaggi dettagliati di ciascuna fase:

1.  **➡️ [Preparazione dei Dati](./guides/1_DATA_PREPARATION.md)**
    *   Copre l'acquisizione, la pulizia, la segmentazione, la normalizzazione dell'audio, la trascrizione del testo e la creazione dei file manifest necessari per l'addestramento. Include la fondamentale checklist sulla qualità dei dati.

2.  **➡️ [Configurazione dell'Addestramento](./guides/2_TRAINING_SETUP.md)**
    *   Ti guida attraverso la configurazione del tuo ambiente Python, l'installazione delle dipendenze (come PyTorch con CUDA), la scelta di un framework TTS e la configurazione dei parametri di addestramento nel tuo file di configurazione.

3.  **➡️ [Addestramento del Modello](./guides/3_MODEL_TRAINING.md)**
    *   Spiega come avviare lo script di addestramento, monitorarne i progressi (loss, validation), riprendere un addestramento interrotto e fornisce suggerimenti specifici per il fine-tuning di modelli esistenti.

4.  **➡️ [Inferenza](./guides/4_INFERENCE.md)**
    *   Descrive in dettaglio come utilizzare il checkpoint del modello addestrato per sintetizzare il parlato da nuovo testo, inclusa la singola frase, l'elaborazione batch e le considerazioni multi-speaker.

5.  **➡️ [Packaging e Condivisione](./guides/5_PACKAGING_AND_SHARING.md)**
    *   Fornisce le best practice per organizzare i file del modello addestrato (checkpoint, config, sample), documentarli con un README, gestire il versioning e prepararli per la condivisione o l'archiviazione.

6.  **➡️ [Risoluzione dei Problemi e Risorse](./guides/6_TROUBLESHOOTING_AND_RESOURCES.md)** 
    *   Offre soluzioni per i problemi comuni incontrati durante l'addestramento e l'inferenza, ed elenca strumenti, librerie e community esterne utili.

---

## Conclusione

Seguendo queste guide, acquisirai una comprensione completa del flusso di lavoro per la preparazione dei dati e l'addestramento dei tuoi modelli Text-to-Speech. Ricorda che una meticolosa preparazione dei dati è il fondamento di una voce di alta qualità, e il processo di addestramento spesso comporta un perfezionamento iterativo.

Ora, procedi alla sezione pertinente in base a dove ti trovi nel ciclo di vita del tuo progetto. Buona fortuna nella costruzione delle tue voci personalizzate! 🚀

## Contribuire 

I contributi per migliorare questa guida sono benvenuti! Che tu trovi errori di battitura, imprecisioni, abbia suggerimenti per spiegazioni più chiare, voglia aggiungere informazioni su strumenti o framework specifici, o abbia idee per nuove sezioni, il tuo contributo è prezioso.

Sentiti libero di:

*   **Aprire una Issue:** Per segnalare errori, suggerire miglioramenti o discutere potenziali modifiche.
*   **Inviare una Pull Request:** Per correzioni o aggiunte concrete. Cerca di assicurarti che le tue modifiche siano chiare e in linea con la struttura e il tono generali della guida.

Apprezziamo qualsiasi sforzo per rendere questa guida più accurata, completa e utile per la community!

## Glossario dei Termini Tecnici

Questo glossario spiega i termini tecnici chiave utilizzati nelle guide per aiutare i principianti a comprendere la terminologia:

- **ASR (Automatic Speech Recognition)**: Tecnologia che converte il linguaggio parlato in testo scritto; utilizzata per trascrivere i dati audio.
- **Batch Size**: Il numero di esempi di addestramento elaborati insieme in un singolo passaggio forward/backward; influisce sulla velocità di addestramento e sull'uso della memoria.
- **Checkpoint**: Un'istantanea salvata dei pesi di un modello durante o dopo l'addestramento, che consente di riprendere l'addestramento o di utilizzare il modello per l'inferenza.
- **CUDA**: La piattaforma di calcolo parallelo di NVIDIA che abilita l'accelerazione GPU per i task di deep learning.
- **dBFS (Decibels relative to Full Scale)**: Un'unità di misura per i livelli audio nei sistemi digitali, dove 0 dBFS rappresenta il livello massimo possibile.
- **Diffusion Models**: Una classe di modelli generativi che aggiungono e poi rimuovono gradualmente il rumore dai dati; alcuni sistemi TTS recenti utilizzano questo approccio.
- **FFT (Fast Fourier Transform)**: Un algoritmo che converte i segnali nel dominio del tempo in rappresentazioni nel dominio della frequenza; fondamentale per l'elaborazione audio.
- **Fine-tuning**: Il processo di prendere un modello pre-addestrato e addestrarlo ulteriormente su un dataset più piccolo e specifico per adattarlo a una nuova voce o lingua.
- **LUFS (Loudness Units relative to Full Scale)**: Una misura standardizzata della loudness percepita, più rappresentativa dell'udito umano rispetto alle misure di picco.
- **Manifest File**: Un file di testo che elenca i file audio e le relative trascrizioni, utilizzato per indicare allo script di addestramento dove trovare i dati.
- **Mel Spectrogram**: Una rappresentazione visiva dell'audio che approssima la percezione uditiva umana utilizzando la scala mel; comunemente usata come rappresentazione intermedia nei sistemi TTS.
- **Overfitting**: Quando un modello apprende troppo bene i dati di addestramento, incluso il loro rumore e i valori anomali, con il risultato di prestazioni scarse su nuovi dati.
- **Sampling Rate**: Il numero di campioni audio al secondo (misurato in Hz); rate più alti catturano più dettagli audio ma richiedono più spazio di archiviazione e potenza di elaborazione.
- **STFT (Short-Time Fourier Transform)**: Una tecnica che determina il contenuto in frequenza di sezioni locali di un segnale mentre questo cambia nel tempo.
- **TTS (Text-to-Speech)**: Tecnologia che converte il testo scritto in output vocale parlato.
- **Validation Loss**: Una metrica che misura l'errore di un modello su un dataset di validazione (dati non utilizzati per l'addestramento); aiuta a rilevare l'overfitting.
- **VRAM (Video RAM)**: Memoria su una scheda grafica; i modelli di deep learning e i loro calcoli intermedi vengono qui memorizzati durante l'addestramento.
- **Vocoder**: Un componente di alcuni sistemi TTS che converte le feature acustiche (come i mel spectrogram) in forme d'onda (audio effettivo).

## Guida alla Traduzione

Accogliamo con piacere le traduzioni di questa guida per renderla accessibile a un pubblico più ampio. Se desideri contribuire con una traduzione, segui questi passaggi:

1. **Effettua il fork del repository** sul tuo account GitHub
2. **Crea la struttura di directory necessaria** per la tua lingua:
   ```
   languages/[language_code]/
   ├── README.md
   └── guides/
       ├── 1_DATA_PREPARATION.md
       ├── 2_TRAINING_SETUP.md
       └── ... (all guide files)
   ```
   Dove `[language_code]` è il codice ISO 639-1 a due lettere per la tua lingua (ad esempio `es` per lo spagnolo)

3. **Traduci il contenuto** iniziando dal README.md e poi dai singoli file delle guide
   - Mantieni la stessa struttura dei file e la stessa formattazione Markdown
   - Mantieni invariati tutti gli esempi di codice (dovrebbero rimanere in inglese)
   - Traduci tutto il testo esplicativo, le intestazioni e i commenti

4. **Aggiorna i link di navigazione** in modo che puntino ai file corretti all'interno della directory della tua lingua

5. **Invia una Pull Request** con la tua traduzione

**Note Importanti per i Traduttori:**
- I termini tecnici possono essere difficili da tradurre. In caso di dubbio, puoi mantenere il termine inglese seguito da una breve spiegazione nella tua lingua.
- Cerca di mantenere lo stesso tono e lo stesso livello di dettaglio tecnico dell'originale.
- Se mentre traduci trovi errori o aree di miglioramento nel contenuto originale in inglese, apri una issue separata per affrontarli.

## [Licenza](../../LICENCE.md)
Il contenuto di questa guida è rilasciato sotto licenza [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). Sei libero di condividere e adattare il materiale a condizione di fornire un'attribuzione appropriata. Il contenuto è inoltre protetto dal copyright 2025 di AcTePuKc e qualsiasi nuovo contributo sarà soggetto alla stessa licenza.
