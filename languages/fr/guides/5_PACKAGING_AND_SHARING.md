# Guide 5 : Packaging et partage de votre modèle TTS

**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape précédente : Inférence](./4_INFERENCE.md){: .btn .btn-primary} |  | [Étape suivante : Dépannage et ressources](./6_TROUBLESHOOTING_AND_RESOURCES.md){: .btn .btn-primary} | 


Vous avez entraîné un modèle et pouvez générer de la parole avec lui. Félicitations ! Pour garantir que votre modèle sera utilisable à l'avenir (par vous-même ou par d'autres) et pour faciliter la reproductibilité, un packaging et une documentation appropriés sont essentiels.

---

## 9. Packaging de votre modèle entraîné

Considérez votre modèle entraîné non pas comme un simple fichier `.pth`, mais comme un package complet contenant tout ce qui est nécessaire pour le comprendre et l'utiliser.

### 9.1. Organiser les fichiers de votre modèle

Créez une structure de répertoires propre et autonome pour chaque modèle entraîné distinct ou version significative. Cela facilite la recherche ultérieure de tous les éléments.

**Exemple de structure :**

```
my_tts_model_packages/
└── yoruba_male_v1.0/         # Nom descriptif de ce package de modèle
    ├── checkpoints/          # Répertoire des poids du modèle
    │   ├── best_model.pth    # Checkpoint avec la validation loss la plus basse (ou la meilleure qualité perçue)
    │   └── last_model.pth    # Checkpoint de la toute fin de l'entraînement (optionnel, mais parfois utile)
    │
    ├── config.yaml           # Le fichier de configuration EXACT utilisé pour entraîner CE checkpoint
    │
    ├── training_info.md      # Optionnel : un fichier avec des logs/notes d'entraînement détaillés
    │   ├── train_list.txt    # Copie du fichier manifest d'entraînement utilisé
    │   └── val_list.txt      # Copie du fichier manifest de validation utilisé
    │
    ├── samples/              # Répertoire avec des exemples audio générés par ce modèle
    │   ├── sample_short_sentence.wav
    │   ├── sample_question.wav
    │   └── sample_longer_paragraph.wav
    │
    └── README.md             # Essentiel : documentation lisible par l'humain pour ce package de modèle spécifique
```

**Composants clés expliqués :**

*   **`checkpoints/`** : Contient les poids réels du modèle. Incluez toujours le checkpoint jugé « meilleur » (que ce soit par la loss ou par les tests d'écoute). Inclure le checkpoint final est aussi une bonne pratique.
*   **`config.yaml` (ou `.json`)** : Absolument critique. Ce fichier définit l'architecture et les paramètres du modèle requis pour charger et utiliser correctement le checkpoint. Sans lui, le checkpoint est souvent inutilisable. Assurez-vous qu'il s'agit de la config *exacte* utilisée pour les checkpoints inclus.
*   **`training_info.md` / Manifests (Optionnel mais recommandé) :** Conserver les manifests aide à suivre exactement sur quelles données le modèle a été entraîné. Un `training_info.md` peut contenir des notes sur l'exécution d'entraînement (durée, matériel utilisé, métriques finales, observations).
*   **`samples/`** : Incluez quelques exemples audio variés générés par le `best_model.pth`. Cela démontre rapidement l'identité vocale, la qualité et les caractéristiques du modèle.
*   **`README.md`** : Le manuel d'utilisation de ce package de modèle spécifique. Voir la section suivante.

### 9.2. Rédiger un bon README.md de modèle

Ce README est spécifique à *ce package de modèle*, et non au guide global du projet. Il doit indiquer à quiconque (y compris à vous-même dans le futur) tout ce qu'il faut savoir pour utiliser le modèle.

**Modèle minimal :**

```markdown
# Package de modèle TTS : Voix masculine yoruba v1.0

## Description du modèle
- **Voix :** Voix masculine adulte et claire parlant le yoruba.
- **Qualité des données source :** Entraîné sur environ 25 heures d'enregistrements de diffusion radio propres.
- **Langue(s) :** Yoruba (principalement). Peut avoir une gestion limitée des emprunts anglais selon les données d'entraînement.
- **Style d'élocution :** Style formel, narratif/de diffusion.
- **Architecture du modèle :** [Préciser le framework/l'architecture, par exemple StyleTTS2, VITS]
- **Version :** 1.0

## Détails de l'entraînement
- **Basé sur :** Affiné à partir de [préciser le modèle de base, par exemple un modèle LibriTTS pré-entraîné] OU entraîné à partir de zéro.
- **Données d'entraînement :** Voir les fichiers `train_list.txt` et `val_list.txt` inclus. Nombre total d'heures : ~25h.
- **Configuration d'entraînement clé :** Voir le fichier `config.yaml` inclus.
- **Taux d'échantillonnage :** 22050 Hz (l'audio d'entrée doit correspondre à ce taux pour certains frameworks).
- **Temps d'entraînement :** Environ 48 heures sur 1x NVIDIA RTX 3090.
- **Informations sur le checkpoint :** `best_model.pth` sélectionné en fonction de la validation loss la plus basse à l'étape [XXXXX].

## Comment l'utiliser pour l'inférence
1.  **Prérequis :** Assurez-vous d'avoir installé le framework [préciser le nom du framework TTS, par exemple StyleTTS2], compatible avec cette version du modèle.
2.  **Configuration :** Utilisez le fichier `config.yaml` inclus.
3.  **Checkpoint :** Chargez le fichier `checkpoints/best_model.pth`.
4.  **Texte d'entrée :** Fournissez un texte brut en entrée. Une normalisation du texte correspondant aux données d'entraînement (par exemple, l'expansion des nombres) peut améliorer les résultats.
5.  **Identifiant de locuteur (le cas échéant) :** Il s'agit d'un modèle à locuteur unique. Utilisez l'identifiant de locuteur `[préciser l'ID utilisé, par exemple main_speaker]` si le framework l'exige, sinon il peut ne pas être nécessaire.
6.  **Sortie attendue :** L'audio sera généré à un taux d'échantillonnage de 22050 Hz.

## Échantillons audio
Écoutez des exemples générés par ce modèle :
- [Phrase courte](./samples/sample_short_sentence.wav)
- [Question](./samples/sample_question.wav)
- [Paragraphe plus long](./samples/sample_longer_paragraph.wav)

## Limitations connues / Notes
- Les performances peuvent se dégrader sur des textes très différents du domaine de la diffusion radio.
- Ne modélise pas explicitement les émotions nuancées.
- [Ajouter toute autre observation pertinente]

## Licence
- **Poids du modèle :** [Préciser la licence, par exemple CC BY-NC-SA 4.0, usage recherche/non commercial uniquement, licence MIT - Soyez précis !]
- **Données source :** [Mentionner les restrictions de licence des données source si elles affectent l'utilisation du modèle, par exemple « Entraîné sur des données propriétaires, modèle à usage interne uniquement. »] **Consultez la licence de vos données d'entraînement !**
```

### 9.3. Conseils de versionnement des modèles

Traitez vos modèles entraînés comme des versions logicielles.

*   **Utiliser le versionnement sémantique (Recommandé) :** Utilisez des noms comme `model_v1.0`, `model_v1.1`, `model_v2.0`.
    *   Incrémentez la version PATCH (v1.0 -> v1.0.1) pour des corrections mineures/réentraînements avec les mêmes données/config.
    *   Incrémentez la version MINEURE (v1.0 -> v1.1) pour des améliorations, un réentraînement avec plus de données, des ajustements de config significatifs.
    *   Incrémentez la version MAJEURE (v1.0 -> v2.0) pour des changements d'architecture majeurs ou un réentraînement complet avec des données/objectifs fondamentaux différents.
*   **Mettre à jour les README :** Lors de la création d'une nouvelle version, mettez à jour son README pour refléter les changements par rapport à la version précédente.
*   **Conserver les anciennes versions :** N'éliminez pas immédiatement les versions antérieures. Parfois, un modèle précédent peut être plus performant sur certains types de texte, ou vous pourriez avoir besoin de revenir en arrière si une nouvelle version introduit des régressions. Si le stockage le permet, archivez-les.

### 9.4. Considérations relatives au partage et à la distribution

Si vous prévoyez de partager votre modèle :

*   **Packaging :** Créez une archive compressée (par exemple, `.zip`, `.tar.gz`) de l'ensemble du répertoire du package de modèle (contenant les checkpoints, la config, le README, les échantillons, etc.).
*   **Plateformes d'hébergement :**
    *   **Hugging Face Hub (Models) :** Excellente plateforme pour partager des modèles, comprend le versionnement, les model cards (utilisez le contenu de votre README !) et potentiellement des widgets d'inférence. Facile pour les autres à découvrir et utiliser.
    *   **GitHub Releases :** Adapté aux modèles plus petits, attachez l'archive zip à un tag de release dans le dépôt de votre projet.
    *   **Stockage cloud (Google Drive, Dropbox, S3) :** Simple pour le partage direct, mais moins facile à découvrir et dépourvu de fonctionnalités de versionnement. Assurez-vous que les permissions des liens sont correctement définies.
*   **Licences (CRITIQUE) :**
    *   **Votre modèle :** Choisissez une licence pour les *poids* du modèle que vous distribuez (par exemple, MIT, Apache 2.0 pour les licences permissives ; CC BY-NC-SA pour un partage non commercial).
    *   **Dépendance aux données :** **Point crucial : la licence de vos données d'entraînement dicte souvent la manière dont vous pouvez licencier votre modèle entraîné.** Si vous avez entraîné sur des données avec une licence non commerciale, vous ne pouvez généralement pas publier votre modèle sous une licence commerciale permissive. Si vous avez entraîné sur des données protégées par le droit d'auteur sans autorisation, vous ne pouvez probablement pas partager le modèle publiquement du tout. **Vérifiez toujours les licences de vos sources de données.**
    *   **Licence du framework :** Le code du framework TTS lui-même possède sa propre licence, distincte de celle de votre modèle.
    *   **Indiquer clairement les conditions d'utilisation :** Utilisez le `README.md` au sein de votre package de modèle pour indiquer clairement l'utilisation prévue (par exemple, recherche uniquement, non commercial, libre pour tout usage) et les conditions de licence.

---

Un packaging et une documentation appropriés de vos modèles les rendent nettement plus précieux et utilisables, que ce soit pour vos propres projets futurs ou pour la collaboration et le partage au sein de la communauté.

**Étape suivante :** [Dépannage et ressources](./6_TROUBLESHOOTING_AND_RESOURCES.md){: .btn .btn-primary} | 
[Retour en haut](#top){: .btn .btn-primary}
