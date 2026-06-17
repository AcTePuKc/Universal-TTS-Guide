# Guide universel d'entraînement de modèles TTS et de préparation de jeux de données

## Langues disponibles

- [English](../../README.md) (Original)
- [Español](../es/README.md)
- **Français** (Actuel)
- [Italiano](../it/README.md)
- [Български](../bg/README.md)

*Vous souhaitez contribuer à une traduction ? Consultez le [Guide de traduction](../../README.md#translation-guide) ci-dessous.*

## Introduction

Bienvenue ! Ce guide complet propose un processus universel pour préparer vos propres jeux de données vocaux et entraîner un modèle de synthèse vocale (TTS, Text-to-Speech) personnalisé. Que vous disposiez d'un petit jeu de données (par exemple, 10 heures) ou d'un plus volumineux (100 heures et plus), ces étapes vous aideront à organiser correctement vos données et à mener à bien le processus d'entraînement pour la plupart des frameworks TTS modernes.

**Objectif :** Vous donner les moyens d'effectuer un fine-tuning ou d'entraîner un modèle TTS sur une voix ou une langue spécifique à partir de vos propres paires audio-texte.

**Ce que couvre ce guide :**
Ce guide est divisé en plusieurs parties, couvrant l'intégralité du flux de travail, de la planification à l'utilisation de votre modèle entraîné :

1.  **Planification :** Considérations initiales avant de démarrer votre projet.
2.  **Préparation des données :** Acquisition, traitement et structuration des données audio et textuelles.
3.  **Configuration de l'entraînement :** Préparation de votre environnement et configuration des paramètres d'entraînement.
4.  **Entraînement du modèle :** Lancement, suivi et fine-tuning du modèle TTS.
5.  **Inférence :** Utilisation de votre modèle entraîné pour synthétiser de la parole.
6.  **Packaging et partage :** Organisation et documentation de votre modèle pour une utilisation future ou une distribution.
7.  **Dépannage et ressources :** Problèmes courants et outils utiles.

---

## 0. Avant de commencer : planifier votre jeu de données

Avant de collecter des données, prenez en compte ces points cruciaux pour vous assurer que votre projet est bien défini et réalisable :

1.  **Locuteur :** S'agira-t-il d'un seul locuteur ou de plusieurs locuteurs ? Les jeux de données à locuteur unique sont plus simples pour débuter avec un fine-tuning ou un entraînement initial. Les modèles multi-locuteurs nécessitent un équilibrage soigneux des données et une gestion des identifiants de locuteur (speaker ID).
2.  **Source des données :** Où obtiendrez-vous l'audio ? (Livres audio, podcasts, archives radiophoniques, données vocales enregistrées professionnellement, vos propres enregistrements). **Point crucial : assurez-vous de disposer des droits ou des licences nécessaires pour utiliser ces données afin d'entraîner des modèles.**
3.  **Qualité audio :** Visez la meilleure qualité possible. Privilégiez des enregistrements propres avec un minimum de bruit de fond, de réverbération, de musique ou de paroles superposées. La cohérence des conditions d'enregistrement est très bénéfique.
4.  **Langue et domaine :** Quelle(s) langue(s) le modèle parlera-t-il ? Quel est le style de parole ou le domaine (par exemple, narration, conversationnel, lecture de bulletins d'information) ? Le modèle sera plus performant sur des textes similaires à ses données d'entraînement.
5.  **Volume de données cible :** Quelle quantité de données prévoyez-vous de collecter ou d'utiliser ?
    *   **~1 à 5 heures :** Peut suffire pour un *clonage* vocal de base avec un modèle pré-entraîné performant, mais la qualité peut être limitée.
    *   **~5 à 20 heures :** Généralement considéré comme le minimum pour un *fine-tuning* correct d'une voix spécifique sur un modèle pré-entraîné.
    *   **50 à 100 heures et plus :** Préférable pour entraîner des modèles robustes ou des modèles dépendant moins des poids pré-entraînés, en particulier pour les langues moins courantes.
    *   **1000 heures et plus :** Nécessaire pour entraîner des modèles polyvalents de haute qualité, en grande partie à partir de zéro.
6.  **Sampling rate :** Décidez tôt d'un sampling rate (fréquence d'échantillonnage) cible (par exemple, 16000 Hz, 22050 Hz, 44100 Hz, 48000 Hz). Des fréquences plus élevées capturent davantage de détails mais nécessitent plus de stockage et de calcul. **Toutes vos données d'entraînement DOIVENT utiliser de manière cohérente la fréquence choisie.** 22050 Hz constitue un compromis courant pour de nombreux modèles TTS.

---

## Vue d'ensemble du processus et navigation

Ce guide est décomposé en modules ciblés. Suivez les liens ci-dessous pour obtenir des étapes détaillées sur chaque phase :

1.  **➡️ [Préparation des données](./guides/1_DATA_PREPARATION.md)**
    *   Couvre l'acquisition, le nettoyage, la segmentation, la normalisation de l'audio, la transcription du texte et la création des fichiers manifest nécessaires à l'entraînement. Comprend la checklist cruciale de qualité des données.

2.  **➡️ [Configuration de l'entraînement](./guides/2_TRAINING_SETUP.md)**
    *   Vous guide dans la configuration de votre environnement Python, l'installation des dépendances (comme PyTorch avec CUDA), le choix d'un framework TTS et la configuration des paramètres d'entraînement dans votre fichier de configuration.

3.  **➡️ [Entraînement du modèle](./guides/3_MODEL_TRAINING.md)**
    *   Explique comment lancer le script d'entraînement, suivre sa progression (loss, validation), reprendre un entraînement interrompu et fournit des conseils spécifiques pour le fine-tuning de modèles existants.

4.  **➡️ [Inférence](./guides/4_INFERENCE.md)**
    *   Détaille comment utiliser le checkpoint de votre modèle entraîné pour synthétiser de la parole à partir d'un nouveau texte, y compris le traitement d'une phrase unique, le traitement par lots et les considérations multi-locuteurs.

5.  **➡️ [Packaging et partage](./guides/5_PACKAGING_AND_SHARING.md)**
    *   Fournit les bonnes pratiques pour organiser les fichiers de votre modèle entraîné (checkpoints, configs, échantillons), les documenter avec un README, les versionner et les préparer pour le partage ou l'archivage.

6.  **➡️ [Dépannage et ressources](./guides/6_TROUBLESHOOTING_AND_RESOURCES.md)** 
    *   Propose des solutions aux problèmes courants rencontrés lors de l'entraînement et de l'inférence, et répertorie des outils, bibliothèques et communautés externes utiles.

---

## Conclusion

En suivant ces guides, vous acquerrez une compréhension complète du flux de travail pour préparer des données et entraîner vos propres modèles de synthèse vocale. N'oubliez pas qu'une préparation méticuleuse des données est le fondement d'une voix de haute qualité, et que le processus d'entraînement implique souvent un raffinement itératif.

Maintenant, passez à la section pertinente en fonction de l'étape où vous en êtes dans le cycle de vie de votre projet. Bonne chance pour la création de vos voix personnalisées ! 🚀

## Contribuer 

Les contributions visant à améliorer ce guide sont les bienvenues ! Que vous trouviez des fautes de frappe, des inexactitudes, que vous ayez des suggestions pour des explications plus claires, que vous souhaitiez ajouter des informations sur des outils ou frameworks spécifiques, ou que vous ayez des idées pour de nouvelles sections, votre contribution est précieuse.

N'hésitez pas à :

*   **Ouvrir une Issue :** Pour signaler des erreurs, suggérer des améliorations ou discuter de changements potentiels.
*   **Soumettre une Pull Request :** Pour des corrections ou ajouts concrets. Veuillez essayer de vous assurer que vos modifications sont claires et s'alignent avec la structure et le ton général du guide.

Nous apprécions tout effort visant à rendre ce guide plus précis, complet et utile pour la communauté !

## Glossaire des termes techniques

Ce glossaire explique les principaux termes techniques utilisés tout au long des guides afin d'aider les débutants à comprendre la terminologie :

- **ASR (Automatic Speech Recognition)** : Technologie qui convertit le langage parlé en texte écrit ; utilisée pour transcrire les données audio (reconnaissance automatique de la parole).
- **Batch Size** : Le nombre d'exemples d'entraînement traités ensemble lors d'une passe avant/arrière (forward/backward) ; affecte la vitesse d'entraînement et l'utilisation de la mémoire (taille de lot).
- **Checkpoint** : Un instantané sauvegardé des poids d'un modèle pendant ou après l'entraînement, vous permettant de reprendre l'entraînement ou d'utiliser le modèle pour l'inférence.
- **CUDA** : La plateforme de calcul parallèle de NVIDIA qui permet l'accélération GPU pour les tâches de deep learning.
- **dBFS (Decibels relative to Full Scale)** : Une unité de mesure des niveaux audio dans les systèmes numériques, où 0 dBFS représente le niveau maximal possible (décibels par rapport à la pleine échelle).
- **Diffusion Models** : Une classe de modèles génératifs qui ajoutent progressivement puis suppriment du bruit des données ; certains systèmes TTS récents utilisent cette approche (modèles de diffusion).
- **FFT (Fast Fourier Transform)** : Un algorithme qui convertit les signaux du domaine temporel en représentations du domaine fréquentiel ; fondamental pour le traitement audio (transformée de Fourier rapide).
- **Fine-tuning** : Le processus consistant à prendre un modèle pré-entraîné et à poursuivre son entraînement sur un jeu de données plus petit et spécifique afin de l'adapter à une nouvelle voix ou langue.
- **LUFS (Loudness Units relative to Full Scale)** : Une mesure standardisée de la sonie perçue, plus représentative de l'audition humaine que les mesures de crête.
- **Manifest File** : Un fichier texte qui répertorie les fichiers audio et leurs transcriptions correspondantes, utilisé pour indiquer au script d'entraînement où trouver les données (fichier manifest).
- **Mel Spectrogram** : Une représentation visuelle de l'audio qui approxime la perception auditive humaine en utilisant l'échelle mel ; couramment utilisée comme représentation intermédiaire dans les systèmes TTS (spectrogramme mel).
- **Overfitting** : Lorsqu'un modèle apprend trop bien les données d'entraînement, y compris leur bruit et leurs valeurs aberrantes, ce qui entraîne de mauvaises performances sur de nouvelles données (surapprentissage).
- **Sampling Rate** : Le nombre d'échantillons audio par seconde (mesuré en Hz) ; des fréquences plus élevées capturent davantage de détails audio mais nécessitent plus de stockage et de puissance de traitement (fréquence d'échantillonnage).
- **STFT (Short-Time Fourier Transform)** : Une technique qui détermine le contenu fréquentiel de sections locales d'un signal au fur et à mesure qu'il évolue dans le temps (transformée de Fourier à court terme).
- **TTS (Text-to-Speech)** : Technologie qui convertit le texte écrit en sortie vocale parlée (synthèse vocale).
- **Validation Loss** : Une métrique qui mesure l'erreur d'un modèle sur un jeu de données de validation (données non utilisées pour l'entraînement) ; aide à détecter l'overfitting (perte de validation).
- **VRAM (Video RAM)** : La mémoire d'une carte graphique ; les modèles de deep learning et leurs calculs intermédiaires y sont stockés pendant l'entraînement.
- **Vocoder** : Un composant de certains systèmes TTS qui convertit les caractéristiques acoustiques (comme les spectrogrammes mel) en formes d'onde (audio réel).

## Guide de traduction

Nous accueillons favorablement les traductions de ce guide afin de le rendre accessible à un public plus large. Si vous souhaitez contribuer à une traduction, veuillez suivre ces étapes :

1. **Forkez le dépôt** vers votre propre compte GitHub
2. **Créez la structure de répertoires nécessaire** pour votre langue :
   ```
   languages/[language_code]/
   ├── README.md
   └── guides/
       ├── 1_DATA_PREPARATION.md
       ├── 2_TRAINING_SETUP.md
       └── ... (tous les fichiers de guide)
   ```
   Où `[language_code]` est le code à deux lettres ISO 639-1 de votre langue (par exemple, `es` pour l'espagnol)

3. **Traduisez le contenu** en commençant par le README.md, puis les fichiers de guide individuels
   - Conservez la même structure de fichiers et le même formatage Markdown
   - Laissez tous les exemples de code inchangés (ils doivent rester en anglais)
   - Traduisez tout le texte explicatif, les en-têtes et les commentaires

4. **Mettez à jour les liens de navigation** pour qu'ils pointent vers les bons fichiers dans le répertoire de votre langue

5. **Soumettez une Pull Request** avec votre traduction

**Remarques importantes pour les traducteurs :**
- Les termes techniques peuvent être difficiles à traduire. En cas de doute, vous pouvez conserver le terme anglais suivi d'une brève explication dans votre langue.
- Essayez de conserver le même ton et le même niveau de détail technique que l'original.
- Si vous trouvez des erreurs ou des points à améliorer dans le contenu anglais original lors de la traduction, veuillez ouvrir une issue distincte pour les traiter.

## [Licence](../../LICENCE.md)
Le contenu de ce guide est sous licence [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). Vous êtes libre de partager et d'adapter le contenu tant que vous attribuez le crédit approprié. Le contenu est également protégé par le droit d'auteur 2025 AcTePuKc et toute nouvelle contribution sera soumise à la même licence.
