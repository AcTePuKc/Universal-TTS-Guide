# Guide 6 : Dépannage et ressources

**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape précédente : Packaging et partage](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

Ce guide fournit des solutions aux problèmes courants rencontrés lors des processus de préparation des données, d'entraînement et d'inférence TTS, ainsi qu'une liste d'outils et de ressources utiles.

---

## 8. Dépannage des problèmes courants

Référez-vous à ce tableau lorsque vous rencontrez des problèmes. Les problèmes sont souvent liés à la qualité des données ou aux réglages de configuration.

| Catégorie de problème     | Problème spécifique                                 | Causes possibles et solutions                                                                                                                                                            | Guide(s) pertinent(s)                                                             |
| :----------------------- | :-------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Préparation des données** | Erreurs de script pendant le découpage/la normalisation | Chemins de fichiers incorrects ; format audio initialement non pris en charge ; dépendances manquantes (`ffmpeg`, `pydub`) ; audio extrêmement bruyant/silencieux perturbant la détection de silence. **Vérifiez les chemins du script, installez les dépendances, ajustez les paramètres de silence.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md)                                  |
|                          | La génération du manifest ignore de nombreux fichiers | Noms de fichiers incohérents entre audio et transcriptions ; fichiers de transcription vides ; chemins incorrects spécifiés dans le script ; encodage non-UTF8 dans les fichiers texte. **Vérifiez les noms, vérifiez les chemins, assurez-vous que les fichiers texte ont du contenu et un encodage UTF-8.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md)                                  |
| **Configuration de l'entraînement** | Échec de `pip install`                    | Bibliothèques système manquantes (par ex. `libsndfile-dev`) ; version de Python incompatible ; problèmes réseau ; conflits entre paquets. **Lisez attentivement les messages d'erreur, installez les libs système, utilisez un environnement virtuel, vérifiez la documentation du framework pour les prérequis.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md)                                    |
|                          | PyTorch `cuda is not available`                   | Version de PyTorch incorrecte installée (CPU uniquement) ; version du pilote NVIDIA/toolkit CUDA incompatible ; GPU non détecté par l'OS. **Réinstallez PyTorch avec la bonne version de CUDA depuis le site officiel, mettez à jour les pilotes.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md)                                    |
| **Exécution de l'entraînement** | Erreur CUDA Out-of-Memory (OOM) au démarrage/pendant l'entraînement | `batch_size` trop grand pour la VRAM du GPU ; architecture du modèle trop complexe ; fuite de mémoire dans le framework/code personnalisé. **Réduisez le `batch_size` dans la config ; activez l'Automatic Mixed Precision (AMP/FP16) si disponible ; vérifiez les mises à jour du framework.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | La training loss est `NaN` ou diverge (explose)   | Learning rate trop élevé ; gradients instables ; mauvais batch de données (par ex. audio/texte corrompu) ; problèmes de précision numérique. **Baissez le learning rate ; vérifiez la qualité des données ; utilisez le gradient clipping (souvent activé par défaut) ; essayez le FP32 si vous utilisez AMP/FP16.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | La training loss stagne (ne diminue pas)          | Learning rate trop bas ; mauvaise qualité/variété des données ; modèle bloqué dans un minimum local ; configuration du modèle incorrecte. **Augmentez légèrement le learning rate ; améliorez/augmentez les données ; vérifiez la config (surtout les paramètres audio) ; essayez un autre optimiseur.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
|                          | La validation loss augmente tandis que la training loss diminue (Overfitting) | Modèle mémorisant les données d'entraînement ; jeu de validation insuffisant/non représentatif ; entraînement trop long. **Arrêtez l'entraînement tôt (selon la meilleure val loss) ; ajoutez des données d'entraînement plus variées ; utilisez la régularisation (weight decay, dropout - vérifiez la config) ; améliorez le jeu de validation.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md) |
| **Qualité de l'inférence** | La sortie sonne robotique/monotone                | Entraînement insuffisant ; mauvaise prosodie dans les données d'entraînement ; limitations de l'architecture du modèle ; problèmes de normalisation du texte. **Entraînez plus longtemps ; améliorez la variété/qualité des données ; essayez une autre architecture de modèle ; assurez-vous que le texte est bien ponctué/normalisé.** | [1_DATA_PREPARATION.md](./1_DATA_PREPARATION.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md), [4_INFERENCE.md](./4_INFERENCE.md) |
|                          | La sortie est bruyante/brouillée/inintelligible   | Mauvaise qualité des données (bruit incorporé) ; le modèle n'a pas convergé ; incompatibilité entre la config d'entraînement et la config/le checkpoint d'inférence ; sampling rate incorrect utilisé en inférence. **Nettoyez rigoureusement les données d'entraînement ; entraînez plus longtemps ; assurez une correspondance EXACTE config/checkpoint ; vérifiez les paramètres audio.** | Tous les guides                                                                   |
|                          | La sortie sonne comme le mauvais locuteur (fine-tuning) | Modèle pré-entraîné non chargé correctement ; learning rate trop élevé initialement ; données/steps de fine-tuning insuffisants ; incompatibilité d'ID de locuteur. **Vérifiez `pretrained_model_path` et `ignore_layers` dans la config ; utilisez un LR plus bas pour le fine-tuning ; entraînez plus longtemps ; vérifiez l'ID de locuteur.** | [2_TRAINING_SETUP.md](./2_TRAINING_SETUP.md), [3_MODEL_TRAINING.md](./3_MODEL_TRAINING.md), [4_INFERENCE.md](./4_INFERENCE.md) |
|                          | L'inférence se coupe trop tôt ou parle trop vite/lentement | Limitation du modèle (prédiction de durée) ; réglage d'inférence limitant la longueur de sortie maximale ; paramètre length scale/vitesse incorrect. **Vérifiez la documentation du framework pour les réglages de max decoder steps / max length ; ajustez les paramètres de contrôle de la vitesse.** | [4_INFERENCE.md](./4_INFERENCE.md)                                                |
| **Utilisation du modèle** | Impossible de charger le fichier de checkpoint    | Téléchargement/fichier corrompu ; utilisation d'un checkpoint avec une version de framework ou un fichier de config incompatible ; chemin de fichier incorrect. **Retéléchargez/vérifiez l'intégrité du fichier ; utilisez la bonne config ; assurez-vous que la version du framework correspond à celle utilisée pour l'entraînement ; vérifiez le chemin.** | [5_PACKAGING_AND_SHARING.md](./5_PACKAGING_AND_SHARING.md), [4_INFERENCE.md](./4_INFERENCE.md) |

---

## 10. Ressources et outils utiles

Cette liste comprend des logiciels, bibliothèques et communautés utiles pour les projets TTS.

### Traitement et analyse audio :

*   **[Audacity](https://www.audacityteam.org/) :** Éditeur audio gratuit, open source et multiplateforme. Excellent pour l'inspection manuelle, le nettoyage, l'étiquetage et le traitement basique des fichiers audio.
*   **[FFmpeg](https://ffmpeg.org/) :** Le couteau suisse en ligne de commande pour la conversion audio/vidéo, le rééchantillonnage, la manipulation des canaux, les changements de format, et bien plus. Essentiel pour scripter des opérations par lots.
*   **[SoX (Sound eXchange Compiled)](http://sox.sourceforge.net/) ou [Sox - Code source](https://codeberg.org/sox_ng/sox_ng/) :** Utilitaire en ligne de commande pour le traitement audio. Utile pour les effets, la conversion de format et l'obtention d'informations audio (commande `soxi`).
*   **[pydub](https://github.com/jiaaro/pydub) :** Bibliothèque Python pour une manipulation audio facile (découpage, conversion de format, ajustement du volume, détection de silence). Utilise le backend ffmpeg/libav.
*   **[librosa](https://librosa.org/doc/latest/index.html) :** Bibliothèque Python pour l'analyse audio avancée, l'extraction de caractéristiques (comme les spectrogrammes mel) et la visualisation. Souvent utilisée en interne par les frameworks TTS.
*   **[soundfile](https://python-soundfile.readthedocs.io/en/latest/) :** Bibliothèque Python pour la lecture/écriture de fichiers audio, basée sur libsndfile. Prend en charge de nombreux formats.
*   **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) :** Bibliothèque Python pour la normalisation de la sonie (LUFS), généralement préférée à la simple normalisation de crête pour une cohérence perçue.

### Transcription (ASR) :

*   **[OpenAI Whisper](https://github.com/openai/whisper) :** Modèle ASR open source de haute qualité, prend en charge de nombreuses langues. Bonne base de référence, mais la ponctuation nécessite souvent une révision. Peut s'exécuter localement (GPU recommandé) ou via API. Diverses implémentations communautaires existent.
*   **[Modèles Google Gemini (via API/AI Studio)](https://ai.google.dev/) :** Modèles capables pour la transcription, souvent performants sur un audio clair, potentiellement meilleurs sur des segments pré-découpés. Vérifiez l'API/Studio pour les capacités et tarifs/niveaux gratuits actuels.
*   **Services ASR cloud :**
    *   [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
    *   [AWS Transcribe](https://aws.amazon.com/transcribe/)
    *   [Azure Speech Service](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/)
    *   *Souvent fiables, paiement à l'usage, peuvent avoir des quotas gratuits initiaux.*
*   **[Hugging Face Transformers - Modèles ASR](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) :** Hub pour de nombreux modèles ASR pré-entraînés, y compris des versions affinées de Whisper et d'autres. Explorez les modèles affinés pour des langues spécifiques ou l'amélioration de la ponctuation.
*   **[ElevenLabs Speech To Text (Scribe)](https://elevenlabs.io/speech-to-text) :** *Service commercial.* Connu pour une très haute précision à la fois en transcription et en ponctuation, mais c'est un service payant qui peut être relativement coûteux comparé aux autres. À envisager si le budget le permet et qu'une précision maximale prête à l'emploi est requise.

### Frameworks et bases de code TTS (Exemples - Recherchez des forks/successeurs actifs) :

*   **[StyleTTS2 (Dépôt de recherche)](https://github.com/yl4579/StyleTTS2) :** Travail influent sur le contrôle du style. Recherchez des forks activement maintenus ayant implémenté des pipelines d'entraînement/inférence.
*   **[VITS (Dépôt de recherche)](https://github.com/jaywalnut310/vits) :** Architecture bout en bout populaire. De nombreux forks et implémentations existent.
*   **[Coqui TTS (Archivé)](https://github.com/coqui-ai/TTS) :** Était une bibliothèque très populaire et complète. Bien qu'archivée, sa base de code et ses concepts restent influents. De nombreux forks actifs peuvent exister.
*   **[ESPnet](https://github.com/espnet/espnet) :** Boîte à outils de traitement de la parole bout en bout, comprenant des recettes TTS pour divers modèles. Davantage orientée recherche.
*   **Rechercher sur GitHub :** Utilisez des mots-clés comme « TTS », « VITS training », « StyleTTS2 training », « PyTorch TTS » pour trouver des projets actuels.

### Environnement Python et Deep Learning :

*   **[Python](https://www.python.org/) :** Le langage de programmation principal.
*   **[PyTorch](https://pytorch.org/) :** La principale bibliothèque de deep learning utilisée par la plupart des frameworks TTS modernes.
*   **[TensorBoard](https://www.tensorflow.org/tensorboard) :** Essentiel pour visualiser la progression de l'entraînement (fonctionne aussi avec PyTorch).
*   **[pip](https://pip.pypa.io/en/stable/) / [uv](https://github.com/astral-sh/uv) :** Installateurs de paquets Python. `uv` est une alternative plus récente, souvent bien plus rapide.
*   **[conda](https://docs.conda.io/en/latest/) / [venv](https://docs.python.org/3/library/venv.html) :** Outils pour créer des environnements Python isolés.
*   **[Git](https://git-scm.com/) :** Système de contrôle de version, essentiel pour cloner des dépôts et gérer le code.
*   **[Hugging Face Hub](https://huggingface.co/) :** Plateforme pour partager des modèles (y compris TTS), des jeux de données et du code.

### Communautés :

*   **GitHub Discussions/Issues des frameworks TTS :** Consultez le dépôt spécifique que vous utilisez pour les questions et réponses de la communauté.
*   **Serveurs Discord :** De nombreuses communautés IA/ML (comme LAION, EleutherAI, les serveurs de frameworks spécifiques) ont des canaux dédiés au TTS.
*   **Reddit :** Des subreddits comme r/SpeechSynthesis, r/MachineLearning.

---

Ceci conclut la série principale de guides. N'oubliez pas que la création de bons modèles TTS implique souvent une itération : revisiter la préparation des données ou ajuster les paramètres d'entraînement en fonction des résultats est une pratique courante. Bonne chance !

---
**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape précédente : Packaging et partage](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | [Retour en haut](#top){: .btn .btn-primary}
