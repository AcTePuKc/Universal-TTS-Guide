# Guide de configuration de l'entraînement TTS


Une fois votre jeu de données prêt, l'étape suivante consiste à configurer l'environnement logiciel et une première configuration exploitable pour entraîner ou affiner votre modèle TTS.

Si un terme lié au matériel ou à l'entraînement n'est pas clair, consultez le [glossaire](../glossary.md#glossary-of-technical-terms). Cette page n'explique que les termes qui influencent directement les décisions de configuration.

---

## Configuration de l'environnement d'entraînement

Cette section couvre l'installation des logiciels nécessaires et l'organisation des fichiers du projet.

### Choisir et cloner un framework TTS

Si vous débutez, choisissez un framework activement maintenu, avec une installation claire et un support déjà présent pour le type d'entraînement dont vous avez besoin. Ne commencez pas par "l'architecture la plus avancée". Commencez par quelque chose que vous pouvez installer, lancer et déboguer.

-   **Choisissez un framework :** Sélectionnez une base de code adaptée à vos objectifs. Tenez compte de :
    *   **L'architecture :** VITS, StyleTTS2, Tacotron2 + vocoder, etc.
    *   **La prise en charge du fine-tuning :** Un chemin clair depuis un modèle pré-entraîné.
    *   **La prise en charge linguistique :** Si le tokenizer et la normalisation conviennent bien à votre langue.
    *   **La communauté et la maintenance :** Si le dépôt reste actif et documenté.
    *   **Les modèles pré-entraînés :** Si vous avez un bon point de départ.

### Comparaison des architectures TTS

| Architecture | Avantages | Inconvénients | Idéal pour | Configuration matérielle |
|:-------------|:----------|:--------------|:-----------|:-------------------------|
| **VITS** | End-to-end, audio de haute qualité, inférence rapide, bon pour le fine-tuning | Plus complexe, peut être instable, demande des réglages soignés | Clonage de voix à locuteur unique et projets axés sur la qualité | Entraînement : 8 Go+ de VRAM, inférence : 4 Go+ |
| **StyleTTS2** | Très bon contrôle de la voix et du style, qualité très élevée | Plus récent, plus complexe, moins de ressources pratiques | Voix expressive et contrôle du style | Entraînement : 12 Go+ de VRAM, inférence : 6 Go+ |
| **Tacotron2 + HiFi-GAN** | Plus stable, plus facile à comprendre, plus de tutoriels | Pipeline en deux étapes, qualité inférieure aux modèles récents | Projets éducatifs ou plus prévisibles | Entraînement : 6 Go+ de VRAM, inférence : 2 Go+ |
| **FastSpeech2** | Inférence rapide, plus stable que Tacotron2, bonne documentation | Exige des alignements phonémiques et un prétraitement plus complexe | Applications rapides et sortie plus contrôlée | Entraînement : 8 Go+ de VRAM, inférence : 2 Go+ |
| **YourTTS / XTTS** | Support multilingue, zero-shot, flexibilité entre langues | Configuration complexe, demande plus d'attention sur les données | Projets multilingues et scénarios cross-lingual | Entraînement : 10 Go+ de VRAM, inférence : 4 Go+ |
| **TTS basé sur la diffusion** | Très fort potentiel de qualité, prosodie plus naturelle | Inférence lente et entraînement coûteux | Génération hors ligne et recherche | Entraînement : 16 Go+ de VRAM, inférence : 8 Go+ |

**Remarque sur le matériel :**
- Il s'agit de minimums approximatifs.
- Des batch sizes plus élevés et des configurations plus grandes demanderont plus de VRAM.
- L'inférence CPU reste possible mais bien plus lente.

**Raccourci pratique :** si vous choisissez votre premier framework pour un projet réel plutôt que pour comparer des architectures, privilégiez celui qui possède la documentation d'installation la plus claire, le suivi des problèmes le plus actif et un exemple de fine-tuning proche de votre cas d'utilisation.

### Exigences matérielles selon l'échelle du projet

#### Exigences GPU par type de modèle et taille de jeu de données

| Type de modèle | Petit jeu (<10h) | Jeu moyen (10-50h) | Grand jeu (>50h) | GPU recommandés |
|:---------------|:-----------------|:-------------------|:-----------------|:----------------|
| **Tacotron2 + HiFi-GAN** | 8 Go VRAM | 12 Go VRAM | 16 Go+ VRAM | RTX 3060, RTX 2080, T4 |
| **FastSpeech2** | 8 Go VRAM | 12 Go VRAM | 16 Go+ VRAM | RTX 3060, RTX 2080, T4 |
| **VITS** | 12 Go VRAM | 16 Go VRAM | 24 Go+ VRAM | RTX 3080, RTX 3090, A5000 |
| **StyleTTS2** | 16 Go VRAM | 24 Go VRAM | 32 Go+ VRAM | RTX 3090, RTX 4090, A100 |
| **XTTS-v2** | 24 Go VRAM | 32 Go VRAM | 40 Go+ VRAM | RTX 4090, A100, A6000 |
| **TTS basé sur la diffusion** | 16 Go VRAM | 24 Go VRAM | 32 Go+ VRAM | RTX 3090, RTX 4090, A100 |

#### CPU, RAM et stockage

| Échelle | CPU | RAM | Stockage |
|:--------|:----|:----|:---------|
| **Personnel** | 4+ cœurs, 2,5 GHz+ | 16 Go | SSD 50 Go |
| **Recherche** | 8+ cœurs, 3,0 GHz+ | 32 Go | SSD 100 Go+ |
| **Production** | 16+ cœurs, 3,5 GHz+ | 64 Go+ | SSD NVMe 500 Go+ |

#### Options cloud GPU à titre indicatif*

**\*Remarque sur l'actualité :** les fournisseurs et exemples de GPU ci-dessous reflètent le paysage du cloud au moment où ce guide a été écrit. Les offres, la disponibilité et les prix changent fréquemment selon la région, les remises et le spot pricing. Utilisez ce tableau comme simple repère et vérifiez les options et les tarifs actuels sur le site du fournisseur avant de budgéter un entraînement.

| Fournisseur | Option GPU | VRAM | Coût relatif | Idéal pour |
|:------------|:-----------|:-----|:-------------|:-----------|
| **Google Colab** | T4/P100 (les niveaux gratuits peuvent varier)<br>V100/A100 (les niveaux payants peuvent varier) | 16 Go<br>16-40 Go | Faible à moyen | Tests et petits jeux |
| **Kaggle** | P100/T4 | 16 Go | Faible | Petits et moyens jeux |
| **AWS** | g4dn.xlarge (T4)<br>p3.2xlarge (V100)<br>p4d.24xlarge (A100) | 16 Go<br>16 Go<br>40 Go | Moyen à très élevé | Toute échelle |
| **GCP** | Instances T4<br>Instances A100 | 16 Go<br>40 Go | Moyen à très élevé | Toute échelle |
| **Azure** | Instances classe V100 ou A100 | 16 Go+ | Moyen à très élevé | Toute échelle |
| **Lambda Labs** | 1x RTX 3090<br>1x A100 | 24 Go<br>40 Go | Moyen | Recherche et jeux moyens |
| **Vast.ai** | Divers GPU grand public | 8-24 Go | Faible à moyen | Entraînement à budget limité |

#### Fourchettes très approximatives de durée d'entraînement

**Remarque sur le temps :** ces fourchettes varient fortement selon l'implémentation, le batch size, la propreté des données, le tokenizer, le checkpoint de départ et le fait de faire du fine-tuning ou un entraînement depuis zéro. Considérez-les comme un ordre de grandeur, pas comme un engagement.

| Modèle | Taille du jeu | GPU | Temps approximatif | Étapes jusqu'à convergence |
|:-------|:--------------|:----|:-------------------|:---------------------------|
| **Tacotron2 + HiFi-GAN** | 10 heures | RTX 3080 | 2-3 jours | 50-100K étapes |
| **FastSpeech2** | 10 heures | RTX 3080 | 2-3 jours | 150-200K étapes |
| **VITS** | 10 heures | RTX 3090 | 3-5 jours | 300-500K étapes |
| **StyleTTS2** | 10 heures | RTX 3090 | 4-7 jours | 500-800K étapes |
| **XTTS-v2** | 10 heures | RTX 4090 | 5-10 jours | 1M+ étapes |

#### Conseils pour réduire les besoins matériels

1. **Gradient accumulation :** simule des batch sizes plus grands en accumulant les gradients sur plusieurs passages forward/backward.
2. **Mixed precision training :** utilise FP16 au lieu de FP32 pour réduire jusqu'à 50 % l'utilisation de la VRAM.
3. **Gradient checkpointing :** échange de la mémoire contre du calcul en recalculant les activations pendant le passage backward.
4. **Model parallelism :** répartit les gros modèles sur plusieurs GPU.
5. **Entraînement progressif :** commencez avec des modèles ou configurations plus petits, puis augmentez progressivement la complexité.

Ces exigences devraient vous aider à planifier le matériel selon les objectifs et le budget de votre projet.

-   **Clonez le dépôt :** Une fois votre choix fait, clonez le framework avec Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY>
    ```
    *   Exemple : `git clone https://github.com/some-user/some-tts-framework.git`

### Configurer l'environnement Python et installer les dépendances

-   **Environnement virtuel :** Il est recommandé d'utiliser un environnement virtuel dédié.
    *   **Avec `venv` :**
        ```bash
        python -m venv venv_tts
        # Windows : .\venv_tts\Scripts\activate
        # Linux/macOS : source venv_tts/bin/activate
        ```
    *   **Avec `conda` :**
        ```bash
        conda create --name tts_env python=3.9
        conda activate tts_env
        ```
-   **Installer PyTorch avec CUDA :** Utilisez le [générateur officiel PyTorch](https://pytorch.org/get-started/locally/) pour faire correspondre la version CUDA, les pilotes et le gestionnaire de paquets.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    ```
-   **Installer les dépendances du framework :**
    ```bash
    pip install -r requirements.txt

    # Si vous utilisez uv :
    uv pip install -r requirements.txt
    ```
    *   **En cas d'erreur :** Vérifiez les bibliothèques système manquantes, les incompatibilités de version ou les problèmes entre CUDA et PyTorch.

#### Test minimal de l'environnement

Avant de modifier une grosse configuration ou de lancer un long entraînement, confirmez que ces commandes fonctionnent :

```bash
python --version
ffmpeg -version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__)"
```

Si l'une d'elles échoue, corrigez l'environnement d'abord. Cela coûte beaucoup moins cher que de déboguer un entraînement cassé après des heures d'attente.

### Organiser le dossier du projet

-   Une structure claire évite la confusion entre manifests, checkpoints et configurations.

    ```mermaid
    flowchart TD
        root["Project Root"] --> repo["TTS Repo Directory"]
        repo --> scripts["Scripts principaux"]
        scripts --> train["train.py"]
        scripts --> inference["inference.py"]
        repo --> config["configs/base_config.yaml"]
        repo --> requirements["requirements.txt"]
        repo --> repoMore["autres fichiers du framework"]
    ```

    ```mermaid
    flowchart TD
        root["Project Root"] --> dataset["my_tts_dataset"]
        dataset --> audio["normalized_chunks"]
        audio --> wav1["segment_00001.wav"]
        audio --> wavMore["autres fichiers .wav"]
        dataset --> transcripts["transcripts (optionnel)"]
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

-   **Chemins :** Assurez-vous que les chemins de votre configuration sont corrects par rapport à l'endroit où vous exécuterez réellement `train.py`, généralement depuis `<TTS_REPO_DIRECTORY>`.

---

## Paramétrer l'exécution de l'entraînement

Avant de lancer l'entraînement, vous avez besoin d'un fichier de configuration qui indique au framework comment entraîner le modèle avec vos données.

### 1. Trouver et copier une configuration de base

-   **Cherchez des exemples :** Parcourez le dossier `configs/` du framework choisi.
-   **Choisissez selon votre objectif :**
    *   **Fine-tuning :** Recherchez des noms comme `config_ft.yaml` ou `finetune_*.yaml` ; ces configurations attendent souvent un modèle pré-entraîné.
    *   **Entraînement depuis zéro :** Recherchez `config_base.yaml` ou `train_*.yaml`.
    *   **Taille du jeu de données :** Certaines implémentations proposent des variantes pour petits (`_sm`) ou grands (`_lg`) jeux.
-   **Copiez et renommez :** Copiez le modèle dans votre propre dossier et donnez-lui un nom explicite pour cette exécution, par exemple `my_yoruba_voice_ft_config.yaml`.
    ```bash
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```
    Sous Windows PowerShell, utilisez `Copy-Item` si vous ne travaillez pas dans Git Bash.

**Conseil débutant :** partez de l'exemple fonctionnel le plus proche déjà fourni par le framework. Ne construisez pas votre première configuration à partir de zéro.

### 2. Modifier votre fichier de configuration personnalisé

-   Ouvrez la copie de la configuration dans un éditeur de texte.
-   **Modifiez les paramètres clés :** Les noms exacts varieront d'un framework à l'autre, mais les catégories restent proches.

    ```yaml
    # --- Jeu de données et chargement ---
    # Chemins relatifs à l'endroit où vous exécutez train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt"
    val_filelist_path: "../my_tts_dataset/val_list.txt"
    # Certains frameworks peuvent aussi demander data_path ou audio_root vers le dossier audio.

    # --- Sortie et logs ---
    output_directory: "../checkpoints/my_yoruba_voice_run1"
    log_interval: 100
    validation_interval: 1000
    save_checkpoint_interval: 5000

    # --- Hyperparamètres principaux ---
    epochs: 1000
    batch_size: 16                     # Réduisez cette valeur en cas d'erreur CUDA OOM.
    learning_rate: 1e-4                # Peut nécessiter un réglage ; elle est souvent plus basse en fine-tuning.
    # lr_scheduler: "cosine_decay"
    # weight_decay: 0.01

    # --- Paramètres audio ---
    sampling_rate: 22050               # DOIT correspondre à la fréquence d'échantillonnage du jeu préparé (du Guide 1).
    # filter_length: 1024
    # hop_length: 256
    # win_length: 1024
    # n_mel_channels: 80
    # mel_fmin: 0.0
    # mel_fmax: 8000.0

    # --- Architecture du modèle ---
    # model_type: "VITS"
    # hidden_channels: 192
    # num_speakers: 1

    # --- Détails du fine-tuning (le cas échéant) ---
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth"
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```

-   **Lisez la documentation du framework :** C'est la référence exacte pour chaque paramètre.
-   **Remarque sur les termes :** dans cette configuration, un [checkpoint](../glossary.md#glossary-checkpoint) est un instantané sauvegardé du modèle, et le [sampling rate](../glossary.md#glossary-sampling-rate) doit correspondre exactement au jeu préparé.
-   **Piège fréquent au début :** pour le premier lancement, modifiez seulement l'essentiel : chemins des données, dossier de sortie, sampling rate, batch size et checkpoint de fine-tuning si nécessaire. Ne touchez pas dix réglages à la fois avant d'avoir validé que le pipeline fonctionne une fois.

### 3. Prendre en compte le matériel et le jeu de données

-   **VRAM GPU :** la [VRAM](../glossary.md#glossary-vram) est la mémoire de la carte graphique. `batch_size` est le réglage principal pour contrôler cette mémoire. Commencez avec une valeur recommandée et réduisez-la en cas d'erreur « CUDA out of memory ».
-   **Taille du jeu vs nombre d'epochs :**
    *   **Petits jeux (<20h) :** peuvent nécessiter moins d'époques (par exemple 300-1500), mais demandent un suivi attentif de la [validation loss](../glossary.md#glossary-validation-loss) et des échantillons pour éviter l'[overfitting](../glossary.md#glossary-overfitting). Envisagez des learning rates plus faibles.
    *   **Grands jeux (>50h) :** peuvent bénéficier de davantage d'époques (1000+) pour apprendre pleinement les motifs des données.
-   **CPU :** même si le GPU effectue l'essentiel du travail, un processeur multicœur correct est nécessaire pour charger et prétraiter les données.
-   **Stockage :** prévoyez de l'espace pour le jeu de données, l'environnement Python, le code du framework et surtout les checkpoints, qui peuvent occuper des centaines de Mo ou plusieurs Go.

### 4. Outils de suivi (TensorBoard)

-   La plupart des frameworks TTS modernes s'intègrent avec [TensorBoard](https://www.tensorflow.org/tensorboard).
-   La configuration contient souvent des options comme `use_tensorboard: True` ou `log_directory`.
-   Pendant l'entraînement, vous pouvez généralement exécuter `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (par exemple `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) dans un autre terminal pour suivre les courbes de loss, le learning rate et les échantillons de validation synthétisés.
-   Si TensorBoard reste vide, vérifiez d'abord que le framework écrit bien les fichiers d'événements dans le dossier attendu. Un tableau vide est souvent simplement un mauvais chemin de logs.

---

Avec l'environnement prêt et la configuration adaptée à vos données, vous pouvez maintenant passer à l'entraînement réel du modèle.

## Avant de continuer

- [ ] Votre environnement Python est activé et les dépendances du framework s'installent sans erreur.
- [ ] `torch.cuda.is_available()` renvoie `True` si vous comptez entraîner sur GPU.
- [ ] `ffmpeg` et les bibliothèques système nécessaires sont installés et visibles dans le PATH.
- [ ] Les chemins de votre configuration pointent vers de vrais manifests, checkpoints et dossiers de sortie.
- [ ] Le `sampling_rate` de la configuration correspond exactement au jeu de données préparé.
