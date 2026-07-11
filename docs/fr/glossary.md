<a id="glossary-of-technical-terms"></a>
# Glossaire des termes techniques

Ce glossaire explique les principaux termes techniques utilisés tout au long des guides afin d'aider les débutants à comprendre la terminologie :

- **ASR (Automatic Speech Recognition)** : Technologie qui convertit le langage parlé en texte écrit ; utilisée pour transcrire les données audio (reconnaissance automatique de la parole).
- **Batch Size** : Le nombre d'exemples d'entraînement traités ensemble lors d'une passe avant/arrière (forward/backward) ; affecte la vitesse d'entraînement et l'utilisation de la mémoire (taille de lot).
<a id="glossary-checkpoint"></a>
- **Checkpoint** : Un instantané sauvegardé des poids d'un modèle pendant ou après l'entraînement, vous permettant de reprendre l'entraînement ou d'utiliser le modèle pour l'inférence.
<a id="glossary-cuda"></a>
- **CUDA** : La plateforme de calcul parallèle de NVIDIA qui permet l'accélération GPU pour les tâches de deep learning.
- **dBFS (Decibels relative to Full Scale)** : Une unité de mesure des niveaux audio dans les systèmes numériques, où 0 dBFS représente le niveau maximal possible (décibels par rapport à la pleine échelle).
- **Diffusion Models** : Une classe de modèles génératifs qui ajoutent progressivement puis suppriment du bruit des données ; certains systèmes TTS récents utilisent cette approche (modèles de diffusion).
- **FFT (Fast Fourier Transform)** : Un algorithme qui convertit les signaux du domaine temporel en représentations du domaine fréquentiel ; fondamental pour le traitement audio (transformée de Fourier rapide).
- **Fine-tuning** : Le processus consistant à prendre un modèle pré-entraîné et à poursuivre son entraînement sur un jeu de données plus petit et spécifique afin de l'adapter à une nouvelle voix ou langue.
- **LUFS (Loudness Units relative to Full Scale)** : Une mesure standardisée de la sonie perçue, plus représentative de l'audition humaine que les mesures de crête.
<a id="glossary-manifest-file"></a>
- **Manifest File** : Un fichier texte qui répertorie les fichiers audio et leurs transcriptions correspondantes, utilisé pour indiquer au script d'entraînement où trouver les données (fichier manifest).
- **Mel Spectrogram** : Une représentation visuelle de l'audio qui approxime la perception auditive humaine en utilisant l'échelle mel ; couramment utilisée comme représentation intermédiaire dans les systèmes TTS (spectrogramme mel).
- <a id="glossary-overfitting"></a>**Overfitting** : Lorsqu'un modèle apprend trop bien les données d'entraînement, y compris leur bruit et leurs valeurs aberrantes, ce qui entraîne de mauvaises performances sur de nouvelles données (surapprentissage).
- <a id="glossary-sampling-rate"></a>**Sampling Rate** : Le nombre d'échantillons audio par seconde (mesuré en Hz) ; des fréquences plus élevées capturent davantage de détails audio mais nécessitent plus de stockage et de puissance de traitement (fréquence d'échantillonnage).
- **STFT (Short-Time Fourier Transform)** : Une technique qui détermine le contenu fréquentiel de sections locales d'un signal au fur et à mesure qu'il évolue dans le temps (transformée de Fourier à court terme).
- **TTS (Text-to-Speech)** : Technologie qui convertit le texte écrit en sortie vocale parlée (synthèse vocale).
- <a id="glossary-validation-loss"></a>**Validation Loss** : Une métrique qui mesure l'erreur d'un modèle sur un jeu de données de validation (données non utilisées pour l'entraînement) ; aide à détecter l'overfitting (perte de validation).
<a id="glossary-vram"></a>
- **VRAM (Video RAM)** : La mémoire d'une carte graphique ; les modèles de deep learning et leurs calculs intermédiaires y sont stockés pendant l'entraînement.
- **Vocoder** : Un composant de certains systèmes TTS qui convertit les caractéristiques acoustiques (comme les spectrogrammes mel) en formes d'onde (audio réel).
