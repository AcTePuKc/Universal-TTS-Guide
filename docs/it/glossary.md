<a id="glossary-of-technical-terms"></a>
# Glossario dei Termini Tecnici

Questo glossario spiega i termini tecnici chiave utilizzati nelle guide per aiutare i principianti a comprendere la terminologia:

- **ASR (Automatic Speech Recognition)**: Tecnologia che converte il linguaggio parlato in testo scritto; utilizzata per trascrivere i dati audio.
- **Batch Size**: Il numero di esempi di addestramento elaborati insieme in un singolo passaggio forward/backward; influisce sulla velocità di addestramento e sull'uso della memoria.
<a id="glossary-checkpoint"></a>
- **Checkpoint**: Un'istantanea salvata dei pesi di un modello durante o dopo l'addestramento, che consente di riprendere l'addestramento o di utilizzare il modello per l'inferenza.
<a id="glossary-cuda"></a>
- **CUDA**: La piattaforma di calcolo parallelo di NVIDIA che abilita l'accelerazione GPU per i task di deep learning.
- **dBFS (Decibels relative to Full Scale)**: Un'unità di misura per i livelli audio nei sistemi digitali, dove 0 dBFS rappresenta il livello massimo possibile.
- **Diffusion Models**: Una classe di modelli generativi che aggiungono e poi rimuovono gradualmente il rumore dai dati; alcuni sistemi TTS recenti utilizzano questo approccio.
- **FFT (Fast Fourier Transform)**: Un algoritmo che converte i segnali nel dominio del tempo in rappresentazioni nel dominio della frequenza; fondamentale per l'elaborazione audio.
- **Fine-tuning**: Il processo di prendere un modello pre-addestrato e addestrarlo ulteriormente su un dataset più piccolo e specifico per adattarlo a una nuova voce o lingua.
- **LUFS (Loudness Units relative to Full Scale)**: Una misura standardizzata della loudness percepita, più rappresentativa dell'udito umano rispetto alle misure di picco.
<a id="glossary-manifest-file"></a>
- **Manifest File**: Un file di testo che elenca i file audio e le relative trascrizioni, utilizzato per indicare allo script di addestramento dove trovare i dati.
- **Mel Spectrogram**: Una rappresentazione visiva dell'audio che approssima la percezione uditiva umana utilizzando la scala mel; comunemente usata come rappresentazione intermedia nei sistemi TTS.
- <a id="glossary-overfitting"></a>**Overfitting**: Quando un modello apprende troppo bene i dati di addestramento, incluso il loro rumore e i valori anomali, con il risultato di prestazioni scarse su nuovi dati.
- <a id="glossary-sampling-rate"></a>**Sampling Rate**: Il numero di campioni audio al secondo (misurato in Hz); rate più alti catturano più dettagli audio ma richiedono più spazio di archiviazione e potenza di elaborazione.
- **STFT (Short-Time Fourier Transform)**: Una tecnica che determina il contenuto in frequenza di sezioni locali di un segnale mentre questo cambia nel tempo.
- **TTS (Text-to-Speech)**: Tecnologia che converte il testo scritto in output vocale parlato.
- <a id="glossary-validation-loss"></a>**Validation Loss**: Una metrica che misura l'errore di un modello su un dataset di validazione (dati non utilizzati per l'addestramento); aiuta a rilevare l'overfitting.
<a id="glossary-vram"></a>
- **VRAM (Video RAM)**: Memoria su una scheda grafica; i modelli di deep learning e i loro calcoli intermedi vengono qui memorizzati durante l'addestramento.
- **Vocoder**: Un componente di alcuni sistemi TTS che converte le feature acustiche (come i mel spectrogram) in forme d'onda (audio effettivo).
