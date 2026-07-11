# Guide d'entraînement et de fine-tuning de modèles TTS


Vos données et votre environnement sont prêts. Il est maintenant temps d'entraîner ou d'affiner votre modèle TTS, de suivre sa progression et de gérer les checkpoints de manière sûre.

Si un terme lié à l'entraînement n'est pas clair, consultez le [glossaire](../glossary.md#glossary-of-technical-terms). Cette page n'explique que les termes qui influencent directement le lancement, le suivi ou le débogage de l'entraînement.

---

## Lancer l'entraînement

Cette section explique comment lancer, suivre et gérer l'entraînement.

### Lancer le script d'entraînement

-   **Placez-vous dans le bon dossier :** Ouvrez un terminal et allez dans le répertoire principal du framework TTS qui contient `train.py` ou l'équivalent.
-   **Activez l'environnement virtuel :** Vérifiez que le même environnement Python que pendant la préparation est bien actif.

    ```bash
    # Exemple d'activation
    # Windows : ..\venv_tts\Scripts\activate
    # Linux/macOS : source ../venv_tts/bin/activate
    # Conda : conda activate tts_env
    ```

-   **Lancez l'entraînement avec votre config personnalisée :** la structure exacte de la commande dépend du framework. Voici des modèles courants :

    ```bash
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml

    # Variante avec nom de run
    # Vérifiez si -m remplace output_directory dans votre config.
    python train.py -c ../my_configs/my_yoruba_voice_ft_config.yaml -m my_yoruba_voice_run1

    # Variante avec dossier de checkpoints indiqué directement
    python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --checkpoint_path ../checkpoints/my_yoruba_voice_run1
    ```

-   **Entraînement multi-GPU :** Si vous avez plusieurs GPU et que le framework prend en charge l'entraînement distribué, consultez sa documentation et utilisez éventuellement `torchrun`.

    ```bash
    # Exemple avec torchrun ; adaptez nproc_per_node au nombre de GPU
    torchrun --nproc_per_node=2 train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml
    ```

#### Première vérification du démarrage complet

Avant de laisser tourner un entraînement pendant des heures ou des jours, vérifiez dans les premières minutes :

- le script dépasse bien le démarrage et charge de vrais batchs
- le dossier de sortie commence à recevoir des logs, checkpoints ou fichiers d'événements
- les loss apparaissent comme des nombres normaux, pas comme `NaN` ou `inf`
- la mémoire GPU se stabilise au lieu de grimper jusqu'au crash immédiat

Si le job échoue ici, corrigez cela d'abord. Cinq premières minutes cassées deviennent souvent une journée perdue.

### Suivre la progression

-   **Sortie console :** Surveillez :
    *   l'initialisation du modèle et des data loaders
    *   l'epoch ou le step courant
    *   `train_loss` et `validation_loss`
    *   le learning rate
    *   le temps par step ou par epoch
-   **TensorBoard :** S'il est activé dans votre config, lancez-le dans un autre terminal.

    ```bash
    tensorboard --logdir ../checkpoints/my_yoruba_voice_run1
    ```

    Ouvrez l'URL affichée par TensorBoard (généralement `http://localhost:6006/`). Vous pourrez y suivre les courbes de loss, le learning rate et, si le framework les enregistre, les échantillons de validation synthétisés.

-   **Dossier de sortie :** Il devrait contenir des checkpoints (`.pth`, `.pt` ou `.ckpt`), des logs, une copie de la config, des fichiers d'événements TensorBoard et parfois des échantillons audio synthétisés.

#### À quoi ressemble un bon début de progression

Sur un premier run sain, vous cherchez généralement à voir :

- un entraînement sans crash immédiat, sans `NaN` et sans mémoire incontrôlée
- `train_loss` et `validation_loss` qui diminuent au lieu d'exploser
- des échantillons de validation de plus en plus clairs et stables
- des checkpoints plus tardifs qui sonnent mieux que les tout premiers, même s'ils restent imparfaits

Ne vous focalisez pas sur un seul chiffre de loss. En TTS, l'écoute compte autant que les métriques.

### Comprendre les checkpoints

-   Les checkpoints sont des instantanés de l'état du modèle et souvent aussi de l'optimiseur.
-   Ils servent à :
    *   **reprendre un entraînement interrompu**
    *   **comparer différentes étapes**
    *   **choisir le meilleur modèle**
-   **Fréquence de sauvegarde :** Réglez un `save_checkpoint_interval` raisonnable. Sauvegarder trop souvent consomme du disque ; trop rarement augmente le risque de perdre du progrès utile. De nombreux frameworks enregistrent aussi automatiquement le checkpoint « best » selon la validation loss.

### Reprendre un entraînement interrompu

Si l'entraînement s'arrête de manière inattendue, vous pouvez souvent reprendre depuis le dernier checkpoint :

- Trouvez le checkpoint le plus récent dans le dossier de sortie, par exemple `../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth` ou `latest_checkpoint.pth`.
- Utilisez l'argument de reprise du framework ; son nom peut varier :

```bash
python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --resume_checkpoint ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth

python train.py --config ../my_configs/my_yoruba_voice_ft_config.yaml --restore_path ../checkpoints/my_yoruba_voice_run1/ckpt_step_50000.pth
```

Le script doit charger les poids et l'état de l'optimiseur, puis reprendre au même point de l'entraînement.

### Quand arrêter l'entraînement

-   **Limite d'epochs :** l'entraînement s'arrête lorsque le nombre maximal d'`epochs` de la config est atteint.
-   **Arrêt précoce :** si la validation loss cesse de baisser et augmente régulièrement sur plusieurs intervalles de validation, le modèle peut commencer à subir de l'overfitting.
-   **Évaluation à l'écoute :** écoutez régulièrement les échantillons de validation et arrêtez lorsque la qualité et la stabilité sont suffisantes pour votre objectif, même si la loss continue à diminuer légèrement.

Il arrive qu'un checkpoint légèrement plus tôt sonne mieux que celui ayant la loss minimale absolue.

---

## Fine-tuning ou entraînement à partir de zéro

### Choisir l'approche

Quand vous démarrez un projet TTS, l'une des décisions les plus importantes est de choisir entre le fine-tuning d'un modèle existant et l'entraînement d'un nouveau modèle depuis zéro. Ce tableau vous aide à choisir l'approche la plus adaptée à votre situation :

| Facteur | Fine-tuning | Entraînement à partir de zéro |
|:--------|:------------|:------------------------------|
| **Taille du jeu de données** | Fonctionne bien avec des jeux plus petits (5 à 20 heures)<br>Peut parfois donner des résultats utiles avec 1 à 2 heures pour certaines voix | Demande généralement des jeux plus grands (30+ heures)<br>En dessous de 20 heures, la qualité est souvent plus faible |
| **Similarité de la voix** | Convient mieux si la voix cible ressemble aux voix du modèle de base | Préférable si la voix cible est très particulière ou très différente |
| **Langue** | Fonctionne bien dans la même langue<br>Peut aussi marcher entre langues avec une préparation soignée | Nécessaire lorsqu'il n'existe pas de bon modèle de base pour la langue<br>Capture mieux la phonétique spécifique |
| **Temps d'entraînement** | Bien plus rapide (quelques jours au lieu de semaines)<br>Nécessite moins d'epochs pour converger | Demande beaucoup plus de temps<br>Peut nécessiter 2 à 5 fois plus d'epochs |
| **Exigences matérielles** | Besoins GPU comparables mais sur une durée plus courte<br>Supporte souvent des batch sizes plus petits | Exige un accès GPU soutenu sur une plus longue période<br>Profite davantage des configurations multi-GPU |
| **Potentiel de qualité** | Peut atteindre rapidement une très bonne qualité<br>Peut hériter des limites du modèle de base | Offre un maximum de flexibilité et de potentiel qualitatif<br>Ne dépend pas des contraintes d'un entraînement précédent |
| **Stabilité** | Généralement plus stable<br>Moins susceptible de s'effondrer ou de ne pas converger | Plus sensible aux hyperparamètres<br>Présente un risque plus élevé d'instabilité |

#### Quand choisir le fine-tuning

- Vous avez peu de données
- Vous avez besoin de résultats plus vite
- Votre voix ou langue ressemble à un modèle pré-entraîné disponible
- Vous avez des ressources de calcul limitées
- Vous débutez en entraînement TTS, car le fine-tuning est souvent plus tolérant

#### Quand choisir l'entraînement à partir de zéro

- Vous avez beaucoup de données (30+ heures)
- La voix cible est très différente ou possède des caractéristiques peu représentées dans les modèles de base
- Aucun bon modèle de base n'existe pour votre langue
- Vous avez besoin d'un contrôle maximal sur tous les aspects du modèle
- Vous avez accès à des ressources de calcul importantes
- Vous construisez un modèle de base que d'autres pourront ensuite affiner

### Particularités du fine-tuning

Le fine-tuning s'appuie sur un modèle de base puissant et l'adapte à votre dataset spécifique, qu'il s'agisse d'une voix, d'une langue ou d'un style. Il est généralement plus rapide et nécessite moins de données qu'un entraînement depuis zéro.

#### L'objectif

- Transférer les capacités générales de synthèse vocale du modèle de base, comme la relation texte-audio et une prosodie de base, tout en adaptant l'identité vocale, et parfois l'accent ou le style, à votre dataset plus réduit.

### Différences clés de configuration

- **`pretrained_model_path` :** Vous devez impérativement renseigner le chemin vers le checkpoint du modèle pré-entraîné dans votre configuration.
- **`fine_tuning: True` :** Activez tout indicateur de mode fine-tuning si votre framework l'exige.
- **Learning rate :** Commencez avec un learning rate plus faible que pour un entraînement depuis zéro, par exemple `1e-5`, `2e-5` ou `5e-5`. Un learning rate trop élevé peut détruire l'information utile apprise par le modèle de base.
- **Batch size :** Il peut souvent rester proche de celui d'un entraînement classique, à ajuster selon la VRAM disponible.
- **Epochs :** Le fine-tuning nécessite généralement moins d'epochs qu'un entraînement depuis zéro, mais cela dépend toujours de la taille du dataset et de la qualité visée. Surveillez attentivement la validation loss et les échantillons audio.

### Stratégies de fine-tuning

- **Fine-tuning complet du réseau :** L'approche la plus courante consiste à mettre à jour tout le réseau avec un learning rate faible.
- **Gel de couches :** Certains frameworks permettent de geler temporairement une partie du réseau, comme l'encodeur texte ou le prédicteur de durée, et de n'entraîner que certains composants. Consultez la documentation pour des options comme `--freeze_layers`.
- **Ignorer ou réinitialiser des couches :** Lors du chargement du modèle pré-entraîné, il peut être utile d'utiliser `ignore_layers` ou `reinitialize_layers` pour la sortie finale ou les speaker embeddings, surtout si votre dataset comporte un nombre différent de speakers.

### Ce qu'il faut surveiller pendant le fine-tuning

- **Amélioration rapide au début :** La validation loss devrait baisser assez vite au début de l'adaptation.
- **Qualité perçue :** Écoutez les échantillons générés. La voix doit progressivement se rapprocher du locuteur cible sans perdre en clarté ni en stabilité.
- **Stabilité :** Surveillez les artefacts, répétitions ou dégradations qui apparaissent si l'entraînement dure trop longtemps.

Le fine-tuning est souvent davantage une question de qualité perçue que de poursuite de la loss minimale absolue.

---

## Guide de dépannage pendant l'entraînement

L'entraînement de modèles TTS peut être complexe et provoquer différents problèmes. Les sections suivantes donnent des indications pour les cas les plus courants.

### Erreurs fréquentes et ce qu'elles signifient souvent

| Erreur | Causes possibles | Solutions |
|:-------|:-----------------|:----------|
| `CUDA out of memory` | Batch size trop grand<br>Modèle trop lourd pour le GPU<br>Fuite ou pression mémoire | Réduisez le batch size<br>Activez le gradient checkpointing<br>Utilisez mixed precision<br>Réduisez la longueur des séquences |
| `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long` | Mauvais type de données dans le dataset<br>Tenseurs incompatibles | Vérifiez le prétraitement<br>Assurez-vous que tous les tenseurs ont le bon dtype<br>Ajoutez des conversions de type explicites |
| `ValueError: too many values to unpack` | Désaccord entre les sorties du modèle et la loss<br>Format de données incorrect | Vérifiez la structure des sorties du modèle<br>Contrôlez l'implémentation de la loss<br>Déboguez les sorties du data loader |
| `FileNotFoundError: [Errno 2] No such file or directory` | Mauvais chemins dans la configuration<br>Fichiers de données manquants | Vérifiez tous les chemins<br>Contrôlez l'intégrité des manifests<br>Assurez-vous que toutes les données ont été téléchargées ou extraites |
| `KeyError: 'speaker_id'` | Information speaker manquante<br>Format de dataset incorrect | Vérifiez le format du dataset<br>Contrôlez le fichier de mapping des speakers<br>Ajoutez l'information speaker au manifest |
| `Loss is NaN` | Learning rate trop élevé<br>Initialisation instable<br>Explosion des gradients | Réduisez le learning rate<br>Ajoutez du gradient clipping<br>Vérifiez les divisions par zéro<br>Normalisez les données d'entrée |
| `ModuleNotFoundError: No module named 'X'` | Dépendance manquante<br>Problème d'environnement | Installez le paquet manquant<br>Vérifiez l'environnement virtuel<br>Contrôlez les versions des paquets |
| `RuntimeError: expected scalar type Float but found Double` | Types de tenseurs incohérents | Ajoutez `.float()` aux tenseurs<br>Vérifiez le prétraitement<br>Uniformisez le dtype dans tout le modèle |

### Problèmes de qualité

| Symptôme | Causes possibles | Solutions |
|:---------|:-----------------|:----------|
| **Audio robotique ou bourdonnant** | Problèmes de vocoder<br>Entraînement insuffisant<br>Prétraitement audio de mauvaise qualité | Entraînez le vocoder plus longtemps<br>Vérifiez la normalisation audio<br>Confirmez la cohérence du sampling rate |
| **Mots sautés ou répétés** | Problèmes d'attention<br>Entraînement instable<br>Données insuffisantes | Utilisez guided attention loss<br>Ajoutez plus de variété dans les données<br>Réduisez le learning rate<br>Recherchez de longs silences dans le dataset |
| **Prononciation incorrecte** | Problèmes de normalisation de texte<br>Erreurs phonémiques<br>Désaccord de langue | Améliorez le prétraitement texte<br>Utilisez une entrée basée sur les phonèmes<br>Ajoutez un dictionnaire de prononciation |
| **Perte d'identité du locuteur** | Surapprentissage sur le speaker dominant<br>Speaker embeddings faibles<br>Peu de données par speaker | Rééquilibrez les données speaker<br>Augmentez la dimension des speaker embeddings<br>Revoyez la stratégie multi-speaker |
| **Convergence lente** | Problèmes de learning rate<br>Mauvaise initialisation<br>Dataset complexe | Testez d'autres stratégies de learning rate<br>Utilisez le transfer learning<br>Simplifiez le dataset au début |
| **Entraînement instable** | Forte variance entre batchs<br>Outliers dans le dataset<br>Problèmes d'optimiseur | Utilisez le gradient accumulation<br>Nettoyez les échantillons atypiques<br>Essayez un autre optimiseur |

### Problèmes de framework et d'environnement

#### Coqui TTS

```bash
# Error: "RuntimeError: Error in applying gradient to param_name"
# Solution : recherchez des valeurs NaN dans votre dataset ou réduisez le learning rate
python -c "import torch; torch.autograd.set_detect_anomaly(True)"  # À lancer avant l'entraînement pour déboguer
```

```bash
# Error: "ValueError: Tacotron training requires `r` > 1"
# Solution : définissez correctement le reduction factor dans la config
# Exemple de correction dans config.json :
"r": 2  # Essayez des valeurs entre 2 et 5
```

#### ESPnet

```bash
# Error: "TypeError: forward() missing 1 required positional argument: 'feats'"
# Solution : vérifiez le format des données et assurez-vous que feats est bien fourni
# Débogage du chargement des données :
python -c "from espnet2.train.dataset import ESPnetDataset; dataset = ESPnetDataset(...); print(dataset[0])"
```

#### VITS / StyleTTS

```python
# Error: "RuntimeError: expected scalar type Half but found Float"
# Solution : gardez une précision cohérente dans tout le modèle
# Ajoutez à votre script d'entraînement :
model = model.half()  # Si vous utilisez mixed precision
# OU
model = model.float()  # Si vous n'utilisez pas mixed precision
```

### Problèmes matériels et d'environnement

1. **Fragmentation de la mémoire GPU**
   - **Symptôme :** erreurs OOM après plusieurs heures alors que la VRAM devrait suffire
   - **Solution :** redémarrez périodiquement depuis un checkpoint et essayez des batchs plus petits

2. **Goulot d'étranglement du CPU**
   - **Symptôme :** utilisation GPU faible ou irrégulière
   - **Solution :** augmentez `num_workers` dans le DataLoader, utilisez un stockage plus rapide et précachez le dataset si possible

3. **Goulot d'étranglement disque / I/O**
   - **Symptôme :** pauses périodiques pendant le chargement des données
   - **Solution :** utilisez un SSD, augmentez le prefetch factor ou cachez le dataset en RAM

4. **Conflits d'environnement**
   - **Symptôme :** plantages étranges ou erreurs d'import difficiles à expliquer
   - **Solution :** utilisez des environnements isolés, vérifiez la compatibilité CUDA/PyTorch et évitez de mélanger d'anciennes installations

5. **Activez les logs détaillés :** si nécessaire, ajoutez ceci au script d'entraînement :

   ```python
   # À ajouter au script d'entraînement
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

6. **Utilisez le profiling TensorBoard :** profilez le temps CPU/GPU pour localiser les goulots d'étranglement :

   ```python
   # À ajouter au code d'entraînement
   from torch.profiler import profile, record_function
   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       with record_function("model_inference"):
           # Votre forward pass
           pass
   print(prof.key_averages().table())
   ```

### Stratégies de débogage

1. **Isolez le problème**

   ```bash
   # Tester le chargement des données séparément
   python -c "from your_framework import DataLoader; loader = DataLoader(...); next(iter(loader))"

   # Tester un forward pass avec des données factices
   python -c "import torch; from your_model import Model; model = Model(); x = torch.randn(1, 100); model(x)"
   ```

2. **Simplifiez pour identifier la cause**
   - Entraînez sur un sous-ensemble très petit et propre
   - Désactivez temporairement l'augmentation de données
   - Utilisez une configuration plus légère si le framework le permet

3. **Inspectez les artefacts intermédiaires**
   - Regardez les alignements d'attention, mel spectrograms, logs et échantillons de validation
   - Vérifiez si le problème apparaît dès le début ou seulement après plusieurs checkpoints

4. **Ajoutez plus de visibilité**
   - Activez une journalisation plus détaillée si disponible
   - Sauvegardez davantage d'échantillons intermédiaires
   - Utilisez `torch.autograd.set_detect_anomaly(True)` uniquement pendant le débogage, pas comme réglage permanent

---

Une fois l'entraînement lancé et suivi correctement, l'étape suivante consiste à choisir un bon checkpoint et à utiliser le modèle pour générer de la parole à partir d'un nouveau texte.
