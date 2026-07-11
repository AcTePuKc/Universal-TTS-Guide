# Guide de dépannage et ressources TTS


Ce guide fournit des solutions aux problèmes courants rencontrés lors des processus de préparation des données, d'entraînement et d'inférence TTS, ainsi qu'une liste d'outils et de ressources utiles.

Si un terme de ce guide ne vous est pas familier, consultez le [glossaire](../glossary.md#glossary-of-technical-terms). Le dépannage va plus vite lorsque vous savez distinguer sans hésiter un [checkpoint](../glossary.md#glossary-checkpoint), un [fichier manifest](../glossary.md#glossary-manifest-file), [CUDA](../glossary.md#glossary-cuda) et la [VRAM](../glossary.md#glossary-vram).

---

## Dépannage des problèmes courants

Référez-vous à ce tableau lorsque vous rencontrez des problèmes. Les problèmes sont souvent liés à la qualité des données ou aux réglages de configuration.

Avant de modifier cinq réglages à la fois, vérifiez d'abord l'essentiel dans cet ordre :

1. confirmez les chemins, les noms de fichiers et la structure des dossiers
2. confirmez que le checkpoint et la configuration appartiennent bien à la même session d'entraînement
3. confirmez que les réglages audio correspondent à l'entraînement, en particulier le sampling rate
4. confirmez que l'environnement est bien celui attendu : bon environnement Python, bonnes versions de dépendances et visibilité de CUDA

Cet ordre permet d'attraper une grande partie des erreurs de niveau débutant et intermédiaire avant de passer à un débogage plus profond.

| Catégorie de problème | Problème spécifique | Causes possibles et solutions | Guide(s) pertinent(s) |
| :--- | :--- | :--- | :--- |
| **Préparation des données** | Erreurs de script pendant le découpage ou la normalisation | Chemins de fichiers incorrects ; format audio initialement non pris en charge ; dépendances manquantes (`ffmpeg`, `pydub`) ; audio extrêmement bruyant ou silencieux perturbant la détection de silence. **Vérifiez les chemins du script, installez les dépendances et ajustez les paramètres de silence.** | [1_DATA_PREPARATION.md](./1-data-preparation.md) |
| **Préparation des données** | La génération du manifest ignore de nombreux fichiers | Noms de fichiers incohérents entre audio et transcriptions ; fichiers de transcription vides ; chemins incorrects dans le script ; fichiers texte non encodés en UTF-8. **Vérifiez les noms, les chemins et assurez-vous que les fichiers texte ont du contenu et un encodage UTF-8.** | [1_DATA_PREPARATION.md](./1-data-preparation.md) |
| **Configuration de l'entraînement** | `pip install` échoue | Bibliothèques système manquantes comme `libsndfile-dev` ; version de Python incompatible ; problèmes réseau ; conflits entre paquets. **Lisez attentivement les messages d'erreur, installez les bibliothèques système, utilisez un environnement virtuel et vérifiez la documentation du framework.** | [2_TRAINING_SETUP.md](./2-training-setup.md) |
| **Configuration de l'entraînement** | PyTorch `cuda is not available` | Mauvaise version de PyTorch installée (CPU uniquement) ; version incompatible du pilote NVIDIA ou du toolkit CUDA ; GPU non détecté par l'OS. **Réinstallez PyTorch avec la bonne version CUDA depuis le site officiel et mettez à jour les pilotes.** | [2_TRAINING_SETUP.md](./2-training-setup.md) |
| **Exécution de l'entraînement** | Erreur CUDA Out-of-Memory (OOM) au démarrage ou pendant l'entraînement | `batch_size` trop grand pour la VRAM du GPU ; architecture du modèle trop complexe ; fuite de mémoire dans le framework ou dans du code personnalisé. **Réduisez `batch_size`, activez AMP/FP16 si disponible et vérifiez les mises à jour du framework.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Exécution de l'entraînement** | La training loss est `NaN` ou diverge | Learning rate trop élevé ; gradients instables ; lot de données défectueux ; problèmes de précision numérique. **Baissez le learning rate, vérifiez la qualité des données, utilisez le gradient clipping et essayez le FP32 si vous utilisez AMP/FP16.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Exécution de l'entraînement** | La training loss stagne | Learning rate trop bas ; qualité ou variété des données insuffisante ; modèle bloqué dans un minimum local ; configuration incorrecte. **Augmentez légèrement le learning rate, améliorez ou augmentez les données, vérifiez la configuration et essayez un autre optimiseur.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Exécution de l'entraînement** | La validation loss augmente tandis que la training loss diminue | Le modèle mémorise les données d'entraînement ; le jeu de validation est insuffisant ou non représentatif ; l'entraînement dure trop longtemps. **Arrêtez l'entraînement plus tôt, ajoutez des données plus variées, utilisez la régularisation et améliorez le jeu de validation.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [3_MODEL_TRAINING.md](./3-model-training.md) |
| **Qualité de l'inférence** | La sortie sonne robotique ou monotone | Entraînement insuffisant ; mauvaise prosodie dans les données ; limitations de l'architecture du modèle ; problèmes de normalisation du texte. **Entraînez plus longtemps, améliorez la variété et la qualité des données, essayez une autre architecture et assurez-vous que le texte est bien ponctué et normalisé.** | [1_DATA_PREPARATION.md](./1-data-preparation.md), [3_MODEL_TRAINING.md](./3-model-training.md), [4_INFERENCE.md](./4-inference.md) |
| **Qualité de l'inférence** | La sortie est bruyante, brouillée ou inintelligible | Mauvaise qualité des données ; le modèle n'a pas convergé ; incompatibilité entre la configuration d'entraînement et celle d'inférence ; `sampling rate` incorrect utilisé en inférence. **Nettoyez rigoureusement les données, entraînez plus longtemps, assurez une correspondance exacte entre configuration et checkpoint et vérifiez les paramètres audio.** | Tous les guides |
| **Qualité de l'inférence** | La sortie sonne comme le mauvais locuteur en fine-tuning | Modèle pré-entraîné mal chargé ; learning rate trop élevé au départ ; données ou étapes de fine-tuning insuffisantes ; incompatibilité d'ID de locuteur. **Vérifiez `pretrained_model_path` et `ignore_layers`, utilisez un learning rate plus bas et vérifiez l'ID du locuteur.** | [2_TRAINING_SETUP.md](./2-training-setup.md), [3_MODEL_TRAINING.md](./3-model-training.md), [4_INFERENCE.md](./4-inference.md) |
| **Qualité de l'inférence** | L'inférence se coupe trop tôt ou parle trop vite ou trop lentement | Limitation du modèle ; réglage d'inférence limitant la longueur maximale ; paramètre de vitesse ou `length scale` incorrect. **Consultez la documentation du framework pour les limites de longueur et les paramètres du décodeur et ajustez les contrôles de vitesse.** | [4_INFERENCE.md](./4-inference.md) |
| **Utilisation du modèle** | Impossible de charger le fichier de checkpoint | Fichier corrompu ; checkpoint utilisé avec une version incompatible du framework ou du fichier de config ; chemin incorrect. **Retéléchargez le fichier, vérifiez son intégrité, utilisez la bonne configuration et contrôlez le chemin.** | [5_PACKAGING_AND_SHARING.md](./5-packaging-and-sharing.md), [4_INFERENCE.md](./4-inference.md) |

---

## Si Vous Avez Toujours Besoin d'Aide

Lorsque vous postez dans un issue tracker, sur Discord ou sur un forum, incluez suffisamment de détails pour qu'une autre personne puisse reproduire le problème :

- le nom du framework, la branche ou la release, ainsi que les versions de Python et PyTorch
- le modèle de GPU, la quantité de VRAM et si vous exécutez sur CUDA ou CPU
- la commande exacte exécutée et le message d'erreur exact
- si le problème survient pendant la préparation des données, le démarrage de l'entraînement, le chargement du checkpoint ou l'inférence
- un petit exemple de configuration concernée, de ligne de manifest ou de texte d'entrée si c'est pertinent

Les bons rapports de bug obtiennent des réponses utiles bien plus vite qu'un vague "ça ne marche pas".

Si possible, réduisez le problème à une commande courte, une petite entrée et un message d'erreur exact. Les gens peuvent presque toujours aider plus vite lorsqu'ils n'ont pas à reconstruire tout votre projet avant de commencer.

## Ressources et outils utiles

Cette liste comprend des logiciels, bibliothèques et communautés utiles pour les projets TTS.

Considérez cette section comme une carte de départ, et non comme une liste figée de recommandations. Les dépôts TTS, les forks maintenus, les outils cloud et les modèles de tarification évoluent régulièrement, donc vérifiez l'activité et la documentation actuelles avant de vous engager dans un flux de travail.

### Traitement et analyse audio :

*   **[Audacity](https://www.audacityteam.org/) :** Éditeur audio gratuit, open source et multiplateforme. Excellent pour l'inspection manuelle, le nettoyage, l'étiquetage et le traitement de base.
*   **[FFmpeg](https://ffmpeg.org/) :** Outil en ligne de commande essentiel pour la conversion audio et vidéo, le rééchantillonnage et l'automatisation par lots.
*   **[SoX (Sound eXchange Compiled)](http://sox.sourceforge.net/) ou [Sox - Code source](https://codeberg.org/sox_ng/sox_ng/) :** Utilitaire en ligne de commande utile pour les effets, la conversion de format et l'obtention d'informations audio avec `soxi`.
*   **[pydub](https://github.com/jiaaro/pydub) :** Bibliothèque Python pour une manipulation audio simple.
*   **[librosa](https://librosa.org/doc/latest/index.html) :** Bibliothèque Python pour l'analyse audio avancée, l'extraction de caractéristiques et la visualisation.
*   **[soundfile](https://python-soundfile.readthedocs.io/en/latest/) :** Bibliothèque Python pour la lecture et l'écriture de fichiers audio.
*   **[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) :** Bibliothèque Python pour la normalisation de la sonie (LUFS).

### Transcription (ASR) :

*   **[OpenAI Whisper](https://github.com/openai/whisper) :** Modèle ASR open source de haute qualité, compatible avec de nombreuses langues. Bonne base de référence, mais la ponctuation nécessite souvent une révision.
*   **[Outils et API Google pour la transcription audio](https://ai.google.dev/) :** Google peut proposer des services ou modèles utiles pour la transcription. Les noms de produits, les limites et les niveaux gratuits changent avec le temps, donc vérifiez la documentation actuelle avant de choisir un flux précis.
*   **Services ASR cloud :**
    *   [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
    *   [AWS Transcribe](https://aws.amazon.com/transcribe/)
    *   [Azure Speech Service](https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/)
    *   *Souvent fiables, au paiement à l'usage, avec parfois des quotas gratuits initiaux.*
*   **[Hugging Face Transformers - Modèles ASR](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) :** Hub pour de nombreux modèles ASR pré-entraînés, y compris des versions affinées de Whisper et d'autres.
*   **[ElevenLabs Speech To Text (Scribe)](https://elevenlabs.io/speech-to-text) :** *Service commercial.* Connu pour sa précision élevée, mais payant et potentiellement coûteux.

### Frameworks et bases de code TTS (Exemples - Recherchez des forks ou successeurs actifs) :

*   **[StyleTTS2 (Dépôt de recherche)](https://github.com/yl4579/StyleTTS2) :** Travail influent sur le contrôle du style. Recherchez des forks activement maintenus avec des pipelines complets.
*   **[VITS (Dépôt de recherche)](https://github.com/jaywalnut310/vits) :** Architecture populaire de bout en bout. De nombreux forks et implémentations existent.
*   **[Coqui TTS (Archivé)](https://github.com/coqui-ai/TTS) :** Référence historique. Ce projet a été influent, mais pour un nouveau flux de travail les débutants devraient privilégier des projets actifs ou des forks réellement maintenus.
*   **[ESPnet](https://github.com/espnet/espnet) :** Boîte à outils de traitement de la parole comprenant des recettes TTS pour divers modèles.
*   **Rechercher sur GitHub :** Utilisez des mots-clés comme "TTS", "VITS training", "StyleTTS2 training" ou "PyTorch TTS" pour trouver des projets actuels.

### Environnement Python et Deep Learning :

*   **[Python](https://www.python.org/) :** Le langage de programmation principal.
*   **[PyTorch](https://pytorch.org/) :** La principale bibliothèque de deep learning utilisée par la plupart des frameworks TTS modernes.
*   **[TensorBoard](https://www.tensorflow.org/tensorboard) :** Essentiel pour visualiser la progression de l'entraînement.
*   **[pip](https://pip.pypa.io/en/stable/) / [uv](https://github.com/astral-sh/uv) :** Installateurs de paquets Python. `uv` est une alternative plus récente et souvent plus rapide.
*   **[conda](https://docs.conda.io/en/latest/) / [venv](https://docs.python.org/3/library/venv.html) :** Outils pour créer des environnements Python isolés.
*   **[Git](https://git-scm.com/) :** Système de contrôle de version essentiel pour cloner des dépôts et gérer le code.
*   **[Hugging Face Hub](https://huggingface.co/) :** Plateforme pour partager des modèles, jeux de données et code.

### Communautés :

*   **GitHub Discussions/Issues des frameworks TTS :** Consultez le dépôt spécifique que vous utilisez.
*   **Serveurs Discord :** De nombreuses communautés IA et ML disposent de canaux dédiés au TTS.
*   **Reddit :** Des subreddits comme `r/SpeechSynthesis` et `r/MachineLearning`.

---

Ceci conclut la série principale de guides. N'oubliez pas que la création de bons modèles TTS implique souvent une itération : revisiter la préparation des données ou ajuster les paramètres d'entraînement selon les résultats est une pratique courante.

---
