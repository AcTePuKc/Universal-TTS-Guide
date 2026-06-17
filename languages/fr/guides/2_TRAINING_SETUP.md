# Guide 2 : Configuration et paramétrage de l'environnement d'entraînement

**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape précédente : Préparation des données](./1_DATA_PREPARATION.md){: .btn .btn-primary} | [Étape suivante : Entraînement du modèle](./3_MODEL_TRAINING.md){: .btn .btn-primary}

Une fois votre jeu de données préparé, l'étape suivante consiste à configurer l'environnement logiciel nécessaire et à paramétrer les options pour votre exécution d'entraînement spécifique.

---

## 3. Configuration de l'environnement d'entraînement

Cette section couvre l'installation des logiciels requis et l'organisation de vos fichiers de projet.

### 3.1. Choisir et cloner un framework TTS

-   **Sélectionner un framework :** Choisissez une base de code TTS adaptée à vos objectifs. Tenez compte de facteurs comme :
    *   **Architecture :** VITS, StyleTTS2, Tacotron2+Vocoder, etc. Les architectures plus récentes offrent souvent une meilleure qualité.
    *   **Prise en charge du fine-tuning :** Le framework prend-il explicitement en charge le fine-tuning à partir de modèles pré-entraînés ? C'est souvent plus facile que l'entraînement à partir de zéro.
    *   **Prise en charge linguistique :** Vérifiez si le modèle/tokenizer gère bien votre langue cible.
    *   **Communauté et maintenance :** Le dépôt est-il activement maintenu ? Existe-t-il des discussions communautaires ou des canaux de support ?
    *   **Modèles pré-entraînés :** Le framework fournit-il des modèles pré-entraînés adaptés comme point de départ pour le fine-tuning ?

#### Comparaison des architectures TTS

Lors de la sélection d'une architecture TTS, prenez en compte ces options populaires et leurs caractéristiques :

| Architecture | Avantages | Inconvénients | Idéal pour | Configuration matérielle requise |
|:-------------|:-----|:-----|:---------|:---------------------|
| **VITS** | • Bout en bout (pas de vocoder séparé)<br>• Audio de haute qualité<br>• Inférence rapide<br>• Bon pour le fine-tuning | • Complexe à comprendre<br>• Peut être instable pendant l'entraînement<br>• Nécessite un réglage minutieux des hyperparamètres | • Clonage vocal à locuteur unique<br>• Projets nécessitant une sortie de haute qualité<br>• Lorsque vous disposez de plus de 5 heures de données | • Entraînement : 8 Go+ de VRAM<br>• Inférence : 4 Go+ de VRAM |
| **StyleTTS2** | • Excellent contrôle de la voix et du style<br>• Qualité à la pointe de la technologie<br>• Bon pour l'émotion/la prosodie | • Plus récent, implémentations potentiellement moins stables<br>• Architecture plus complexe<br>• Moins de ressources communautaires | • Projets nécessitant un contrôle du style<br>• Synthèse vocale expressive<br>• Multi-locuteurs avec transfert de style | • Entraînement : 12 Go+ de VRAM<br>• Inférence : 6 Go+ de VRAM |
| **Tacotron2 + HiFi-GAN** | • Bien établi, stable<br>• Plus facile à comprendre<br>• Plus de tutoriels disponibles<br>• Composants séparés pour un débogage plus facile | • Pipeline en deux étapes (plus lent)<br>• Qualité généralement inférieure aux modèles plus récents<br>• Plus sujet aux échecs d'attention sur les textes longs | • Projets éducatifs<br>• Lorsque la stabilité prime sur la qualité<br>• Environnements à faibles ressources | • Entraînement : 6 Go+ de VRAM<br>• Inférence : 2 Go+ de VRAM |
| **FastSpeech2** | • Non autorégressif (inférence plus rapide)<br>• Plus stable que Tacotron2<br>• Bonne documentation | • Nécessite des alignements de phonèmes<br>• Prétraitement plus complexe<br>• Qualité moindre que VITS/StyleTTS2 | • Applications en temps réel<br>• Lorsque la vitesse d'inférence est critique<br>• Sortie plus contrôlée | • Entraînement : 8 Go+ de VRAM<br>• Inférence : 2 Go+ de VRAM |
| **YourTTS (variante VITS)** | • Prise en charge multilingue<br>• Clonage vocal zero-shot<br>• Bon pour le transfert linguistique | • Configuration d'entraînement complexe<br>• Nécessite une préparation soignée des données<br>• Peut nécessiter des jeux de données plus volumineux | • Projets multilingues<br>• Clonage vocal interlingue<br>• Lorsque la flexibilité linguistique est nécessaire | • Entraînement : 10 Go+ de VRAM<br>• Inférence : 4 Go+ de VRAM |
| **TTS basé sur la diffusion** | • Potentiel de qualité le plus élevé<br>• Prosodie plus naturelle<br>• Meilleure gestion des mots rares | • Inférence très lente<br>• Entraînement extrêmement gourmand en calcul<br>• Plus récent, moins établi | • Génération hors ligne<br>• Lorsque la qualité prime sur la vitesse<br>• Projets de recherche | • Entraînement : 16 Go+ de VRAM<br>• Inférence : 8 Go+ de VRAM |

**Remarque sur la configuration matérielle requise :**
- Ce sont des minimums approximatifs ; des batch sizes ou des configurations de modèle plus importants nécessiteront plus de VRAM
- Les durées d'entraînement varient considérablement : VITS/StyleTTS2 nécessitent généralement plus d'epochs que Tacotron2
- L'inférence sur CPU est possible pour tous les modèles mais sera nettement plus lente

### 1.3. Configuration matérielle détaillée requise

Choisir le bon matériel est essentiel pour réussir l'entraînement d'un modèle TTS. Voici un détail des exigences pour différents scénarios :

#### Exigences GPU par type de modèle et taille de jeu de données

| Type de modèle | Petit jeu de données (<10h) | Jeu de données moyen (10-50h) | Grand jeu de données (>50h) | Modèles de GPU recommandés |
|:-----------|:---------------------|:------------------------|:---------------------|:-----------------------|
| **Tacotron2 + HiFi-GAN** | 8 Go de VRAM | 12 Go de VRAM | 16 Go+ de VRAM | RTX 3060, RTX 2080, T4 |
| **FastSpeech2** | 8 Go de VRAM | 12 Go de VRAM | 16 Go+ de VRAM | RTX 3060, RTX 2080, T4 |
| **VITS** | 12 Go de VRAM | 16 Go de VRAM | 24 Go+ de VRAM | RTX 3080, RTX 3090, A5000 |
| **StyleTTS2** | 16 Go de VRAM | 24 Go de VRAM | 32 Go+ de VRAM | RTX 3090, RTX 4090, A100 |
| **XTTS-v2** | 24 Go de VRAM | 32 Go de VRAM | 40 Go+ de VRAM | RTX 4090, A100, A6000 |
| **TTS basé sur la diffusion** | 16 Go de VRAM | 24 Go de VRAM | 32 Go+ de VRAM | RTX 3090, RTX 4090, A100 |

#### CPU et mémoire système

| Échelle d'entraînement | Exigences CPU | RAM système | Stockage |
|:---------------|:-----------------|:-----------|:--------|
| **Loisir/Personnel** | 4+ cœurs, 2,5 GHz+ | 16 Go | SSD 50 Go |
| **Recherche** | 8+ cœurs, 3,0 GHz+ | 32 Go | SSD 100 Go+ |
| **Production** | 16+ cœurs, 3,5 GHz+ | 64 Go+ | SSD NVMe 500 Go+ |

#### Options de GPU cloud et coûts approximatifs

| Fournisseur cloud | Option GPU | VRAM | Coût approx./heure | Idéal pour |
|:---------------|:-----------|:-----|:------------------|:---------|
| **Google Colab** | T4/P100 (Gratuit)<br>V100/A100 (Pro) | 16 Go<br>16-40 Go | Gratuit<br>10-15 $ | Expérimentation, petits jeux de données |
| **Kaggle** | P100/T4 | 16 Go | Gratuit (heures limitées) | Petits à moyens jeux de données |
| **AWS** | g4dn.xlarge (T4)<br>p3.2xlarge (V100)<br>p4d.24xlarge (A100) | 16 Go<br>16 Go<br>40 Go | 0,50-0,75 $<br>3,00-3,50 $<br>20,00-32,00 $ | Toute échelle, production |
| **GCP** | n1-standard-8 + T4<br>a2-highgpu-1g (A100) | 16 Go<br>40 Go | 0,35-0,50 $<br>3,80-4,50 $ | Toute échelle, production |
| **Azure** | NC6s_v3 (V100)<br>NC24ads_A100_v4 | 16 Go<br>80 Go | 3,00-3,50 $<br>16,00-24,00 $ | Toute échelle, production |
| **Lambda Labs** | 1x RTX 3090<br>1x A100 | 24 Go<br>40 Go | 1,10 $<br>1,99 $ | Recherche, jeux de données moyens |
| **Vast.ai** | Divers GPU grand public | 8-24 Go | 0,20-1,00 $ | Entraînement à budget maîtrisé |

#### Estimations de durée d'entraînement

| Modèle | Taille du jeu de données | GPU | Durée d'entraînement approximative | Epochs jusqu'à convergence |
|:------|:-------------|:----|:--------------------------|:----------------------|
| **Tacotron2 + HiFi-GAN** | 10 heures | RTX 3080 | 2-3 jours | 50-100K steps |
| **FastSpeech2** | 10 heures | RTX 3080 | 2-3 jours | 150-200K steps |
| **VITS** | 10 heures | RTX 3090 | 3-5 jours | 300-500K steps |
| **StyleTTS2** | 10 heures | RTX 3090 | 4-7 jours | 500-800K steps |
| **XTTS-v2** | 10 heures | RTX 4090 | 5-10 jours | 1M+ steps |

#### Conseils d'optimisation pour réduire la configuration matérielle requise

1. **Accumulation de gradient (Gradient Accumulation)** : Simulez des batch sizes plus importants en accumulant les gradients sur plusieurs passes avant/arrière
2. **Entraînement en précision mixte (Mixed Precision Training)** : Utilisez le FP16 au lieu du FP32 pour réduire l'utilisation de la VRAM jusqu'à 50 %
3. **Gradient Checkpointing** : Échangez du calcul contre de la mémoire en recalculant les activations lors de la passe arrière
4. **Parallélisme de modèle (Model Parallelism)** : Répartissez les grands modèles sur plusieurs GPU
5. **Entraînement progressif (Progressive Training)** : Commencez par des modèles/configurations plus petits et augmentez progressivement la complexité

Ces exigences devraient vous aider à planifier vos besoins matériels en fonction des objectifs spécifiques de votre projet et de vos contraintes budgétaires.
-   **Cloner le dépôt :** Une fois choisi, clonez le dépôt de code du framework à l'aide de Git.
    ```bash
    git clone <URL_OF_YOUR_CHOSEN_TTS_REPO>
    cd <TTS_REPO_DIRECTORY> # Naviguez dans le répertoire cloné
    ```
    *   Exemple : `git clone https://github.com/some-user/some-tts-framework.git`

### 3.2. Configurer l'environnement Python et installer les dépendances

-   **Environnement virtuel (Recommandé) :** Créez et activez un environnement virtuel Python dédié pour isoler les dépendances et éviter les conflits avec d'autres projets ou les paquets Python du système.
    *   **Avec `venv` (intégré) :**
        ```bash
        python -m venv venv_tts  # Crée un environnement nommé 'venv_tts'
        # Activez-le :
        # Windows : .\venv_tts\Scripts\activate
        # Linux/macOS : source venv_tts/bin/activate
        ```
    *   **Avec `conda` :**
        ```bash
        conda create --name tts_env python=3.9 # Ou la version Python souhaitée
        conda activate tts_env
        ```
-   **Installer PyTorch avec CUDA :** C'est essentiel pour l'accélération GPU. Visitez le [guide d'installation officiel de PyTorch](https://pytorch.org/get-started/locally/) et sélectionnez les options correspondant à votre OS, votre gestionnaire de paquets (`pip` ou `conda`), votre plateforme de calcul (version de CUDA) et la version de PyTorch souhaitée. **Assurez-vous que vos pilotes NVIDIA installés sont compatibles avec la version de CUDA choisie.**
    ```bash
    # Exemple de commande avec pip pour CUDA 11.8 (vérifiez le site de PyTorch pour les commandes actuelles !)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Vérifiez l'installation :
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
    # Devrait afficher la version de PyTorch, True, et votre version de CUDA en cas de succès.
    ```
-   **Installer les dépendances du framework :** La plupart des frameworks listent leurs dépendances dans un fichier `requirements.txt`. Installez-les avec `pip` (ou `uv`, qui est souvent plus rapide).
    ```bash
    # Naviguez d'abord vers le répertoire du framework si vous n'y êtes pas déjà
    # Avec pip :
    pip install -r requirements.txt

    # Avec uv (si installé : pip install uv) :
    uv pip install -r requirements.txt
    ```
    *   **Dépannage :** Soyez attentif à toute erreur d'installation. Elle peut indiquer des bibliothèques système manquantes (comme `libsndfile`), des versions de paquets incompatibles, ou des problèmes avec votre configuration CUDA/PyTorch. Consultez la documentation du framework pour les prérequis spécifiques.

### 3.3. Organiser votre dossier de projet

-   Une structure de dossiers bien organisée facilite la gestion de votre projet. Placez votre jeu de données préparé (ou créez un lien symbolique vers celui-ci) dans ou à côté du code du framework. Une structure courante ressemble à ceci :

    ```bash
    <YOUR_PROJECT_ROOT>/
    ├── <TTS_REPO_DIRECTORY>/         # Le code du framework cloné
    │   ├── train.py                 # Script d'entraînement principal (le nom peut varier)
    │   ├── inference.py             # Script d'inférence (le nom peut varier)
    │   ├── configs/                 # Répertoire des fichiers de configuration
    │   │   └── base_config.yaml     # Exemple de config du framework
    │   ├── requirements.txt
    │   └── ... (autres fichiers du framework)
    │
    ├── my_tts_dataset/              # Votre jeu de données préparé du Guide 1
    │   ├── normalized_chunks/       # Fichiers audio finaux
    │   │   ├── segment_00001.wav
    │   │   └── ...
    │   ├── transcripts/             # Optionnel : fichiers texte s'ils ne sont pas directement dans le manifest
    │   ├── train_list.txt           # Manifest d'entraînement
    │   └── val_list.txt             # Manifest de validation
    │
    ├── checkpoints/                 # Créez ce répertoire : où les modèles seront enregistrés
    │   └── my_custom_model/         # Sous-répertoire pour une exécution d'entraînement spécifique
    │
    └── my_configs/                  # Optionnel : placez vos configs personnalisées ici
        └── my_training_run_config.yaml
    ```
-   **Chemins :** Assurez-vous que les chemins spécifiés ultérieurement dans votre fichier de configuration (pour les jeux de données, les sorties) sont corrects par rapport à l'endroit où vous *exécuterez* le script `train.py` (généralement depuis le `<TTS_REPO_DIRECTORY>`).

---

## 4. Paramétrer l'exécution de l'entraînement

Avant de lancer l'entraînement, vous devez créer un fichier de configuration qui indique au framework *comment* entraîner le modèle, en utilisant *vos* données spécifiques.

### 4.1. Trouver et copier une configuration de base

-   **Localiser des exemples :** Explorez le répertoire `configs/` au sein du framework TTS. Recherchez des fichiers de configuration (`.yaml`, `.json`, ou similaires) servant de modèles.
-   **Choisir judicieusement :** Sélectionnez un fichier de configuration qui correspond à votre objectif :
    *   **Fine-tuning :** Recherchez des noms comme `config_ft.yaml`, `finetune_*.yaml`. Ils supposent souvent que vous fournirez un modèle pré-entraîné.
    *   **Entraînement à partir de zéro :** Recherchez des noms comme `config_base.yaml`, `train_*.yaml`.
    *   **Taille du jeu de données :** Certains frameworks peuvent proposer des configs réglées pour les petits (`_sm`) ou grands (`_lg`) jeux de données.
-   **Copier et renommer :** Copiez le fichier modèle choisi vers un nouvel emplacement (par exemple, votre propre répertoire `my_configs/` ou dans le répertoire `configs/` du framework) et donnez-lui un nom descriptif pour votre exécution spécifique (par exemple, `my_yoruba_voice_ft_config.yaml`).
    ```bash
    # Exemple : Copier une config de fine-tuning
    cp <TTS_REPO_DIRECTORY>/configs/base_finetune_config.yaml my_configs/my_yoruba_voice_ft_config.yaml
    ```

### 4.2. Éditer votre fichier de configuration personnalisé

-   Ouvrez votre fichier de configuration nouvellement copié (`my_yoruba_voice_ft_config.yaml`) dans un éditeur de texte.
-   **Modifier les paramètres clés :** Examinez et modifiez soigneusement les paramètres. Les noms des paramètres **varieront considérablement** d'un framework à l'autre, mais les catégories courantes incluent :

    ```yaml
    # --- Jeu de données et chargement des données ---
    # Chemins relatifs à l'endroit où vous exécutez train.py
    train_filelist_path: "../my_tts_dataset/train_list.txt" # Chemin vers votre manifest d'entraînement
    val_filelist_path: "../my_tts_dataset/val_list.txt"   # Chemin vers votre manifest de validation
    # Certains frameworks peuvent nécessiter 'data_path' ou 'audio_root' pointant vers le répertoire audio à la place/en plus.

    # --- Sortie et journalisation ---
    output_directory: "../checkpoints/my_yoruba_voice_run1" # TRÈS IMPORTANT : où les modèles, logs, échantillons sont enregistrés. Créez ce répertoire de base si nécessaire.
    log_interval: 100                  # Fréquence (en steps/batches) d'affichage des logs
    validation_interval: 1000          # Fréquence (en steps/batches) d'exécution de la validation
    save_checkpoint_interval: 5000     # Fréquence (en steps/batches) de sauvegarde des checkpoints du modèle

    # --- Hyperparamètres principaux d'entraînement ---
    epochs: 1000                       # Nombre total de passes sur les données d'entraînement. À ajuster selon la taille du jeu de données et la convergence.
    batch_size: 16                     # Nombre d'échantillons traités en parallèle par GPU. DIMINUEZ en cas d'erreurs CUDA OOM. AUGMENTEZ pour un entraînement plus rapide si la VRAM le permet.
    learning_rate: 1e-4                # Taux d'apprentissage initial. Peut nécessiter un réglage (par ex. plus bas pour le fine-tuning : 5e-5 ou 1e-5).
    # lr_scheduler: "cosine_decay"     # Planification du learning rate (par ex. step decay, exponential decay) - dépend du framework
    # weight_decay: 0.01               # Paramètre de régularisation

    # --- Paramètres audio ---
    sampling_rate: 22050               # CRITIQUE : DOIT correspondre au sampling rate de votre jeu de données préparé (du Guide 1).
    # Autres paramètres audio (dépendent souvent de l'architecture du modèle) :
    # filter_length: 1024              # Taille de la FFT pour la STFT
    # hop_length: 256                  # Pas (hop size) pour la STFT
    # win_length: 1024                 # Taille de fenêtre pour la STFT
    # n_mel_channels: 80               # Nombre de bandes mel
    # mel_fmin: 0.0                    # Fréquence mel minimale
    # mel_fmax: 8000.0                 # Fréquence mel maximale (souvent sampling_rate / 2)

    # --- Architecture du modèle ---
    # model_type: "VITS"               # Type d'architecture du modèle
    # hidden_channels: 192             # Taille des couches internes
    # num_speakers: 1                  # Définir à >1 pour les jeux de données multi-locuteurs (doit correspondre à la préparation des données)

    # --- Spécificités du fine-tuning (si applicable) ---
    # Définir 'True' ou fournir un chemin lors du fine-tuning
    fine_tuning: True
    pretrained_model_path: "/path/to/downloaded/base_model.pth" # Chemin vers le checkpoint pré-entraîné de départ.
    # Optionnel : Spécifier les couches à ignorer/réinitialiser si nécessaire
    # ignore_layers: ["speaker_embedding.weight", "decoder.output_layer.weight"]
    ```
-   **Lire la documentation du framework :** Consultez la documentation spécifique de votre framework TTS choisi pour comprendre ce que fait chaque paramètre de son fichier de configuration.

### 4.3. Considérations matérielles et relatives au jeu de données

-   **VRAM du GPU :** Le `batch_size` est le principal levier pour contrôler l'utilisation de la mémoire GPU. Commencez par une valeur recommandée (par exemple, 16 ou 32) et diminuez-la si vous rencontrez des erreurs « CUDA out of memory » au démarrage de l'entraînement. Des batch sizes plus importants conduisent généralement à une convergence plus rapide mais nécessitent plus de VRAM.
-   **Taille du jeu de données vs Epochs :**
    *   **Petits jeux de données (< 20h) :** Peuvent nécessiter moins d'epochs (par exemple, 300-1500) mais demandent un suivi attentif via la validation loss/les échantillons pour éviter l'overfitting (lorsque le modèle mémorise les données d'entraînement mais performe mal sur un nouveau texte). Envisagez des learning rates plus faibles.
    *   **Grands jeux de données (> 50h) :** Peuvent bénéficier de davantage d'epochs (1000+) pour apprendre pleinement les motifs présents dans les données.
-   **CPU :** Bien que le GPU fasse le gros du travail, un CPU multicœur correct est nécessaire pour le chargement et le prétraitement des données, qui peuvent autrement devenir un goulot d'étranglement.
-   **Stockage :** Assurez-vous d'avoir assez d'espace disque pour le jeu de données, l'environnement Python, le code du framework, et surtout les checkpoints sauvegardés, qui peuvent devenir volumineux (de centaines de Mo à plusieurs Go par checkpoint).

### 4.4. Outils de suivi (TensorBoard)

-   La plupart des frameworks TTS modernes s'intègrent avec [TensorBoard](https://www.tensorflow.org/tensorboard) pour visualiser la progression de l'entraînement.
-   Le fichier de configuration comporte souvent des paramètres relatifs à la journalisation (par exemple, `use_tensorboard: True`, `log_directory`).
-   Pendant l'entraînement, vous pouvez généralement lancer TensorBoard en exécutant `tensorboard --logdir <YOUR_OUTPUT_DIRECTORY>` (par exemple, `tensorboard --logdir ../checkpoints/my_yoruba_voice_run1`) dans un terminal séparé. Cela vous permet de suivre les courbes de loss, les learning rates et potentiellement d'écouter les échantillons de validation synthétisés dans votre navigateur web.

---

Avec votre environnement configuré et votre fichier de configuration adapté à vos données et objectifs, vous êtes maintenant prêt à démarrer le processus d'entraînement proprement dit du modèle.

**Étape suivante :** [Entraînement du modèle](./3_MODEL_TRAINING.md){: .btn .btn-primary} | 
[Retour en haut](#top){: .btn .btn-primary}
