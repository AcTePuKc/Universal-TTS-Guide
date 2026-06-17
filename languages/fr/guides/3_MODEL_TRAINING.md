# Guide 3 : Entraînement et fine-tuning du modèle

**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape précédente : Configuration de l'entraînement](./2_TRAINING_SETUP.md){: .btn .btn-primary} | [Étape suivante : Inférence](./4_INFERENCE.md){: .btn .btn-primary}

Vous avez préparé vos données et configuré votre environnement d'entraînement. Il est maintenant temps d'entraîner (ou d'affiner) votre modèle de synthèse vocale. Cette phase implique l'exécution du script d'entraînement, le suivi de sa progression et la compréhension de la gestion du processus.

---

## 5. Lancer l'entraînement

Cette section détaille comment lancer, suivre et gérer le processus d'entraînement.

### 5.1. Lancer le script d'entraînement

-   **Naviguer vers le bon répertoire :** Ouvrez votre terminal ou invite de commande et naviguez dans le répertoire principal du dépôt du framework TTS cloné (le répertoire contenant le script `train.py` ou son équivalent).
-   **Activer l'environnement virtuel :** Assurez-vous que votre environnement virtuel Python dédié (par exemple, `venv_tts`, `tts_env`) est activé.
    ```bash
    # Exemple d'activation (ajustez le chemin/nom selon les besoins)
    # Windows : ..\venv_tts\Scripts\activate
    # Linux/macOS : source ../venv_tts/bin/activate
    # Conda : conda activate tts_env
    ```
-   **Exécuter la commande d'entraînement :** Lancez le script d'entraînement du framework, en le pointant vers votre fichier de configuration personnalisé créé au Guide 2. La structure exacte de la commande varie d'un framework à l'autre. Les schémas courants incluent :
    ```bash
    # Schéma courant 1 : Utilisation de l'argument --config
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Schéma courant 2 : Utilisation de -c pour la config et -m pour le nom du répertoire de modèle/sortie
    # (Vérifiez si votre output_directory dans la config est remplacé par -m)
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Schéma courant 3 : Spécifier directement le répertoire de checkpoint
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```
    *   **Entraînement multi-GPU :** Si vous disposez de plusieurs GPU et que le framework prend en charge l'entraînement distribué (consultez sa documentation), vous pourriez utiliser des commandes impliquant `torchrun` ou `python -m torch.distributed.launch`. Exemple :
        ```bash
        # Exemple utilisant torchrun (ajustez nproc_per_node à votre nombre de GPU)
        torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
        ```

### 5.2. Suivre la progression de l'entraînement

-   **Sortie console :** Le terminal où vous avez lancé l'entraînement affichera des informations de progression. Recherchez :
    *   **Initialisation :** Messages indiquant que le modèle est en cours de construction, que les chargeurs de données sont en cours de préparation et potentiellement qu'un modèle pré-entraîné est en cours de chargement (pour le fine-tuning).
    *   **Epochs/Steps :** Progression actuelle de l'entraînement (par exemple, `Epoch: [1/1000]`, `Step: [500/100000]`).
    *   **Valeurs de loss :** Métriques cruciales indiquant la qualité de l'apprentissage du modèle. Attendez-vous à voir `train_loss` (loss sur le batch courant) et, périodiquement, `validation_loss` (loss sur le jeu de validation inédit). Les deux devraient généralement diminuer au fil du temps. Des composantes de loss spécifiques (comme `mel_loss`, `duration_loss`, `kl_loss`) peuvent aussi être rapportées selon l'architecture du modèle.
    *   **Learning Rate :** Le learning rate actuel peut être affiché, surtout si un scheduler le réduit au fil du temps.
    *   **Horodatages/Vitesse :** Temps pris par step ou par epoch.
-   **TensorBoard (Fortement recommandé) :** S'il est activé dans votre config, utilisez TensorBoard pour un suivi visuel.
    *   **Lancement :** Ouvrez un *nouveau* terminal (gardez celui de l'entraînement en cours d'exécution), activez le même environnement virtuel, et lancez :
        ```bash
        # Pointez logdir vers le output_directory spécifié dans votre config
        tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
        ```
    *   **Accès :** Ouvrez l'URL fournie par TensorBoard (généralement `http://localhost:6006/`) dans votre navigateur web.
    *   **Visualisation :** Vous pouvez voir des graphiques des losses d'entraînement et de validation au fil du temps, les planifications du learning rate et potentiellement d'autres métriques.
    *   **Écouter les échantillons audio :** De nombreux frameworks journalisent périodiquement des échantillons audio synthétisés à partir du jeu de validation vers TensorBoard (vérifiez l'onglet `AUDIO`). Les écouter est la *meilleure* façon d'évaluer qualitativement l'amélioration du modèle et d'identifier des problèmes comme le bruit, les erreurs de prononciation ou une sortie robotique.
-   **Répertoire de sortie :** Vérifiez le `output_directory` que vous avez spécifié dans votre config (`../checkpoints/my_yoruba_voice_run1`). Il devrait contenir :
    *   Les checkpoints du modèle sauvegardés (fichiers `.pth`, `.pt`, `.ckpt`).
    *   Les fichiers journaux (`train.log`, etc.).
    *   Des copies du fichier de configuration utilisé.
    *   Les fichiers d'événements TensorBoard (généralement dans un sous-répertoire `logs` ou `events`).
    *   Éventuellement des échantillons audio synthétisés.

### 5.3. Comprendre les checkpoints

-   **Ce qu'ils sont :** Les checkpoints sont des instantanés de l'état du modèle (tous ses poids appris et potentiellement l'état de l'optimiseur) sauvegardés à intervalles spécifiques pendant l'entraînement.
-   **Pourquoi ils sont importants :**
    *   **Reprise de l'entraînement :** Permettent de continuer l'entraînement s'il est interrompu (en raison de plantages, de coupures de courant ou d'un arrêt manuel).
    *   **Évaluation de la progression :** Vous pouvez utiliser des checkpoints de différentes étapes pour synthétiser de l'audio et voir comment le modèle a évolué.
    *   **Sélection du meilleur modèle :** La validation loss aide à identifier les bons checkpoints, mais le *meilleur* modèle est souvent choisi en écoutant l'audio synthétisé à partir de plusieurs checkpoints prometteurs proches de la validation loss la plus basse. Parfois, un checkpoint légèrement antérieur sonne mieux que celui avec la loss absolument la plus basse.
-   **Fréquence de sauvegarde :** Configurez le `save_checkpoint_interval` dans votre config. Sauvegarder trop souvent consomme de l'espace disque ; sauvegarder trop rarement risque de faire perdre une progression significative en cas de plantage. Sauvegarder tous les quelques milliers de steps ou une fois par epoch est courant. De nombreux frameworks sauvegardent aussi automatiquement le « meilleur » checkpoint en fonction de la validation loss.

### 5.4. Reprendre un entraînement interrompu

-   Si votre entraînement s'arrête de manière inattendue ou que vous l'arrêtez manuellement, vous pouvez généralement reprendre à partir du dernier checkpoint sauvegardé.
-   Trouvez le chemin vers le dernier fichier de checkpoint dans votre répertoire de sortie (par exemple, `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` ou `latest_checkpoint.pth`).
-   Utilisez l'argument de reprise du framework lors du relancement du script d'entraînement. Le nom de l'argument varie :
    ```bash
    # Exemple utilisant --resume_checkpoint
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

    # Exemple utilisant --restore_path ou --resume_path
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
    ```
-   Le script devrait charger les poids du modèle et l'état de l'optimiseur depuis le checkpoint et continuer l'entraînement à partir de ce point.

### 5.5. Quand arrêter l'entraînement

-   **Limite d'epochs :** L'entraînement s'arrête automatiquement lorsque le nombre maximal d'`epochs` spécifié dans la config est atteint.
-   **Arrêt précoce (Early Stopping) :** Surveillez la `validation_loss`. Si elle cesse de diminuer et commence à augmenter de manière constante pendant une période prolongée (par exemple, sur plusieurs intervalles de validation), le modèle pourrait commencer à faire de l'overfitting. Vous pourriez envisager d'arrêter l'entraînement manuellement autour du point où la validation loss était la plus basse.
-   **Évaluation qualitative :** Écoutez régulièrement les échantillons audio de validation générés dans TensorBoard ou synthétisez manuellement des échantillons à partir des checkpoints récents. Arrêtez l'entraînement lorsque vous êtes satisfait de la qualité et de la stabilité audio, même si la loss diminue encore légèrement. Un entraînement supplémentaire pourrait ne pas apporter d'améliorations perceptibles, voire dégrader la qualité.

---

## 6. Fine-tuning vs Entraînement à partir de zéro

### 6.1. Choisir votre approche

Au démarrage d'un projet TTS, l'une des décisions les plus importantes est de savoir s'il faut affiner (fine-tuning) un modèle existant ou en entraîner un nouveau à partir de zéro. Ce tableau vous aide à décider quelle approche convient le mieux à votre situation spécifique :

| Facteur | Fine-tuning | Entraînement à partir de zéro |
|:-------|:------------|:----------------------|
| **Taille du jeu de données** | Fonctionne bien avec de plus petits jeux de données (5-20 heures)<br>Peut produire de bons résultats avec aussi peu que 1-2 heures pour certaines voix | Nécessite généralement de plus grands jeux de données (30h+)<br>Moins de 20 heures conduit souvent à une qualité médiocre |
| **Similarité de la voix** | Idéal lorsque votre voix cible est similaire aux voix des données d'entraînement du modèle pré-entraîné | Nécessaire lorsque votre voix cible est très unique ou significativement différente des modèles pré-entraînés disponibles |
| **Langue** | Fonctionne bien pour le fine-tuning au sein de la même langue<br>Peut fonctionner en interlingue avec une préparation soignée | Requis pour les langues sans modèles pré-entraînés disponibles<br>Meilleur pour capturer la phonétique propre à une langue |
| **Durée d'entraînement** | Bien plus rapide (jours au lieu de semaines)<br>Nécessite moins d'epochs pour converger | Durée d'entraînement nettement plus longue<br>Peut nécessiter 2 à 5 fois plus d'epochs |
| **Configuration matérielle requise** | Exigences GPU similaires mais pour moins longtemps<br>Peut souvent utiliser des batch sizes plus petits | Nécessite un accès GPU soutenu pendant de plus longues périodes<br>Peut davantage bénéficier de configurations multi-GPU |
| **Potentiel de qualité** | Peut atteindre rapidement une excellente qualité<br>Peut hériter des limitations du modèle de base | Flexibilité et potentiel de qualité maximaux<br>Aucune contrainte d'un entraînement antérieur |
| **Stabilité** | Processus d'entraînement généralement plus stable<br>Moins sujet à l'effondrement ou à la non-convergence | Plus sensible aux hyperparamètres<br>Risque plus élevé d'instabilité de l'entraînement |

#### Quand choisir le fine-tuning

Le fine-tuning est généralement recommandé lorsque :
- Vous disposez de données limitées (moins de 20 heures)
- Vous avez besoin de résultats plus rapides
- Votre voix/langue cible est raisonnablement similaire aux modèles pré-entraînés disponibles
- Vous disposez de ressources de calcul limitées
- Vous débutez dans l'entraînement TTS (le fine-tuning est plus tolérant)

#### Quand choisir l'entraînement à partir de zéro

L'entraînement à partir de zéro est préférable lorsque :
- Vous disposez de données abondantes (30h+)
- Votre voix cible est très unique ou possède des caractéristiques non représentées dans les modèles pré-entraînés
- Vous travaillez avec une langue mal prise en charge par les modèles existants
- Vous avez besoin d'un contrôle maximal sur tous les aspects du modèle
- Vous avez accès à des ressources de calcul importantes
- Vous construisez un modèle de fondation que d'autres affineront

### 6.2. Spécificités du fine-tuning

Le fine-tuning tire parti d'un modèle pré-entraîné puissant et l'adapte à votre jeu de données spécifique (locuteur, langue, style). Il est généralement plus rapide et nécessite moins de données que l'entraînement à partir de zéro.

#### L'objectif

-   Transférer les capacités générales de synthèse vocale (comme la compréhension de la correspondance texte-son, la prosodie de base) du grand jeu de données sur lequel le modèle de base a été entraîné, tout en spécialisant l'identité vocale et potentiellement l'accent/le style pour correspondre à votre jeu de données plus petit et spécifique.

### 6.2. Différences clés de configuration (Rappel de la configuration)

-   **`pretrained_model_path` :** Vous DEVEZ fournir le chemin vers le fichier de checkpoint du modèle pré-entraîné dans votre configuration.
-   **`fine_tuning: True` :** Assurez-vous que tout indicateur signalant le mode fine-tuning est activé si le framework le requiert.
-   **Learning Rate :** Commencez avec un learning rate *plus bas* que celui généralement utilisé pour l'entraînement à partir de zéro (par exemple, `1e-5`, `2e-5`, `5e-5`). Un learning rate élevé peut détruire les informations précieuses apprises par le modèle pré-entraîné.
-   **Batch Size :** Peut souvent être similaire à l'entraînement à partir de zéro, à ajuster selon la VRAM.
-   **Epochs :** Le nombre d'epochs requis pour le fine-tuning est généralement nettement inférieur à celui de l'entraînement à partir de zéro, mais dépend tout de même de la taille du jeu de données et de la qualité souhaitée. Surveillez de près la validation loss et les échantillons audio.

### 6.3. Stratégies possibles (Dépendantes du framework)

-   **Fine-tuning complet du réseau :** L'approche par défaut consiste souvent à mettre à jour les poids de l'ensemble du réseau, mais avec un learning rate faible.
-   **Gel de couches (Freezing Layers) :** Certains frameworks permettent de geler des parties du réseau (par exemple, l'encodeur de texte ou le prédicteur de durée) initialement et de n'entraîner que des composants spécifiques (comme les speaker embeddings ou le décodeur). Cela peut parfois aider à préserver les forces du modèle de base tout en adaptant des aspects spécifiques. Consultez la documentation de votre framework pour les options `--freeze_layers` ou similaires.
-   **Ignorer des couches :** Lors du chargement du modèle pré-entraîné, vous pourriez vouloir `ignore_layers` (ou `reinitialize_layers`) comme la couche de sortie finale ou la couche de speaker embedding, surtout si votre jeu de données a un nombre de locuteurs différent de celui du modèle pré-entraîné.

### 6.4. Suivre le fine-tuning

-   **Amélioration initiale rapide :** Vous devriez voir la validation loss chuter relativement rapidement au début, à mesure que le modèle s'adapte à la voix cible.
-   **Se concentrer sur la qualité audio :** Prêtez une attention particulière aux échantillons de validation synthétisés. L'identité vocale se déplace-t-elle vers votre locuteur cible ? La parole est-elle claire et stable ? Le fine-tuning relève souvent davantage de la qualité perceptive que de l'atteinte de la valeur de loss minimale absolue.

---

## 7. Guide complet de dépannage

L'entraînement de modèles TTS peut être difficile, avec de nombreux problèmes potentiels. Cette section fournit des solutions aux problèmes courants que vous pourriez rencontrer.

### 7.1. Messages d'erreur courants et solutions

| Message d'erreur | Causes possibles | Solutions |
|:--------------|:----------------|:----------|
| `CUDA out of memory` | • Batch size trop grand<br>• Modèle trop grand pour le GPU<br>• Fuite de mémoire | • Réduire le batch size<br>• Activer le gradient checkpointing<br>• Utiliser l'entraînement en précision mixte<br>• Réduire la longueur de séquence |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | • Type de données incorrect dans le jeu de données<br>• Types de tenseurs incompatibles | • Vérifier le prétraitement des données<br>• S'assurer que tous les tenseurs ont le bon dtype<br>• Ajouter une conversion de type explicite |
| `ValueError: too many values to unpack` | • Incohérence entre les sorties du modèle et les attentes de la fonction de loss<br>• Format de données incorrect | • Vérifier la structure de sortie du modèle<br>• Vérifier l'implémentation de la fonction de loss<br>• Déboguer les sorties du chargeur de données |
| `FileNotFoundError: [Errno 2] No such file or directory` | • Chemins incorrects dans la config<br>• Fichiers de données manquants | • Vérifier tous les chemins de fichiers<br>• Vérifier l'intégrité du fichier manifest<br>• S'assurer que les données sont téléchargées/extraites |
| `KeyError: 'speaker_id'` | • Information de locuteur manquante<br>• Format de jeu de données incorrect | • Vérifier le format du jeu de données<br>• Vérifier le fichier de mappage des locuteurs<br>• Ajouter l'information de locuteur au manifest |
| `Loss is NaN` | • Learning rate trop élevé<br>• Initialisation instable<br>• Explosion du gradient | • Réduire le learning rate<br>• Ajouter un gradient clipping<br>• Vérifier les divisions par zéro<br>• Normaliser les données d'entrée |
| `ModuleNotFoundError: No module named 'X'` | • Dépendance manquante<br>• Problème d'environnement | • Installer le paquet manquant<br>• Vérifier l'environnement virtuel<br>• Vérifier les versions des paquets |
| `RuntimeError: expected scalar type Float but found Double` | • Types de tenseurs incohérents | • Ajouter `.float()` aux tenseurs<br>• Vérifier le prétraitement des données<br>• Standardiser le dtype dans tout le modèle |

### 7.2. Problèmes de qualité de l'entraînement

| Symptôme | Causes possibles | Solutions |
|:--------|:----------------|:----------|
| **Audio robotique/grésillant** | • Problèmes de vocoder<br>• Entraînement insuffisant<br>• Mauvais prétraitement audio | • Entraîner le vocoder plus longtemps<br>• Vérifier la normalisation audio<br>• Vérifier la cohérence du sampling rate |
| **Mots sautés/répétés** | • Problèmes d'attention<br>• Entraînement instable<br>• Données insuffisantes | • Utiliser une guided attention loss<br>• Ajouter plus de variété de données<br>• Réduire le learning rate<br>• Vérifier les longs silences dans les données |
| **Prononciation incorrecte** | • Problèmes de normalisation du texte<br>• Erreurs de phonèmes<br>• Incompatibilité linguistique | • Améliorer le prétraitement du texte<br>• Utiliser une entrée basée sur les phonèmes<br>• Ajouter un dictionnaire de prononciation |
| **Perte de l'identité du locuteur** | • Overfitting au locuteur dominant<br>• Speaker embeddings faibles<br>• Données de locuteur insuffisantes | • Équilibrer les données de locuteur<br>• Augmenter la dimension du speaker embedding<br>• Utiliser une speaker adversarial loss |
| **Convergence lente** | • Problèmes de learning rate<br>• Mauvaise initialisation<br>• Jeu de données complexe | • Essayer différentes planifications de LR<br>• Utiliser le transfert d'apprentissage<br>• Simplifier le jeu de données au départ |
| **Entraînement instable** | • Variance des batches<br>• Valeurs aberrantes dans le jeu de données<br>• Problèmes d'optimiseur | • Utiliser l'accumulation de gradient<br>• Nettoyer les échantillons aberrants<br>• Essayer différents optimiseurs |

### 7.3. Problèmes spécifiques aux frameworks

#### Coqui TTS
```
# Erreur : "RuntimeError: Error in applying gradient to param_name"
# Solution : Vérifiez les valeurs NaN dans votre jeu de données ou réduisez le learning rate
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # À exécuter avant l'entraînement pour déboguer

# Erreur : "ValueError: Tacotron training requires `r` > 1"
# Solution : Définissez correctement le reduction factor dans la config
# Exemple de correction dans config.json :
"r": 2  # Essayez des valeurs entre 2 et 5
```

#### ESPnet
```
# Erreur : "TypeError: forward() missing 1 required positional argument: 'feats'"
# Solution : Vérifiez le formatage des données et assurez-vous que les feats sont fournis
# Déboguer le chargement des données :
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS/StyleTTS
```
# Erreur : "RuntimeError: expected scalar type Half but found Float"
# Solution : Assurez une précision cohérente dans tout le modèle
# Ajoutez à votre script d'entraînement :
model = model.half()  # Si vous utilisez la précision mixte
# OU
model = model.float()  # Si vous n'utilisez pas la précision mixte
```

### 7.4. Problèmes matériels et d'environnement

1. **Fragmentation de la mémoire GPU**
   - **Symptôme** : Erreurs OOM après plusieurs heures d'entraînement malgré une VRAM suffisante
   - **Solution** : Redémarrer périodiquement l'entraînement depuis un checkpoint, utiliser des batches plus petits

2. **Goulots d'étranglement CPU**
   - **Symptôme** : L'utilisation du GPU fluctue ou reste faible
   - **Solution** : Augmenter num_workers dans le DataLoader, utiliser un stockage plus rapide, pré-mettre en cache les jeux de données

3. **Goulots d'étranglement d'E/S disque**
   - **Symptôme** : L'entraînement se bloque périodiquement pendant le chargement des données
   - **Solution** : Utiliser un stockage SSD, augmenter le prefetch factor, mettre en cache le jeu de données en RAM

4. **Conflits d'environnement**
   - **Symptôme** : Plantages mystérieux ou erreurs d'import
   - **Solution** : Utiliser des environnements isolés (conda/venv), vérifier la compatibilité CUDA/PyTorch

### 7.5. Stratégies de débogage

1. **Isoler le problème**
   ```bash
   # Tester le chargement des données séparément
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"
   
   # Tester la passe avant avec des données factices
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Simplifier pour identifier les problèmes**
   - Entraîner sur un tout petit sous-ensemble (10-20 échantillons)
   - Désactiver temporairement l'augmentation de données
   - Essayer d'abord avec un seul locuteur

3. **Visualiser les sorties intermédiaires**
   - Tracer les alignements d'attention
   - Visualiser les spectrogrammes mel à différentes étapes
   - Surveiller les normes de gradient

4. **Activer la journalisation détaillée**
   ```bash
   # Ajoutez à votre script d'entraînement
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

5. **Utiliser le profilage TensorBoard**
   ```python
   # Ajoutez à votre code d'entraînement
   from torch.profiler import profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Votre passe avant
   print(prof.key_averages().table())
   ```

---

L'entraînement étant lancé et suivi, l'étape suivante, après avoir sélectionné un bon checkpoint, consiste à utiliser le modèle pour générer de la parole sur un nouveau texte.

**Étape suivante :** [Inférence](./4_INFERENCE.md){: .btn .btn-primary} | 
[Retour en haut](#top){: .btn .btn-primary}
