# Guide 1 : Préparation des données pour l'entraînement TTS

**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape suivante : Configuration de l'entraînement](./2_TRAINING_SETUP.md){: .btn .btn-primary} | 

Ce guide couvre la première phase critique de tout projet TTS : la préparation de données audio et textuelles de haute qualité et correctement formatées. La qualité de votre jeu de données a un impact direct sur la qualité de votre modèle TTS final.

---

## 1. Étapes de préparation du jeu de données

Suivez ces étapes de manière systématique pour transformer l'audio brut en un jeu de données prêt pour l'entraînement.

### 1.1. Acquisition audio et traitement initial

-   **Rassembler l'audio :** Collectez vos fichiers audio bruts (les formats courants incluent WAV, MP3, FLAC, OGG, M4A). Assurez-vous de disposer des droits d'utilisation de cet audio.
-   **Convertir en WAV :** La plupart des frameworks TTS attendent le format WAV. Utilisez des outils comme `ffmpeg` ou des bibliothèques audio (`pydub`, `soundfile`) pour convertir votre audio. Visez un encodage WAV standard comme PCM 16 bits.
    ```bash
    # Exemple utilisant ffmpeg pour convertir un MP3 en WAV
    ffmpeg -i input_audio.mp3 output_audio.wav
    ```
-   **Standardiser les canaux (Mono) :** Les modèles TTS s'entraînent généralement sur de l'audio mono (un seul canal). Convertissez les pistes stéréo en mono.
    ```bash
    # Exemple utilisant ffmpeg pour convertir un WAV stéréo en WAV mono
    ffmpeg -i stereo_input.wav -ac 1 mono_output.wav
    ```
    *   `-ac 1` : Définit le nombre de canaux audio à 1.
-   **Rééchantillonner l'audio :** Assurez-vous que tous les fichiers audio ont **exactement le même sampling rate**. Choisissez votre fréquence cible en fonction des objectifs de votre projet et de la compatibilité du framework (par exemple, 16000 Hz, 22050 Hz, 48000 Hz). 22050 Hz est courant pour de nombreux modèles.
    ```bash
    # Exemple utilisant ffmpeg pour rééchantillonner à 22050 Hz
    ffmpeg -i input.wav -ar 22050 resampled_output.wav
    ```
    *   `-ar 22050` : Définit le sampling rate audio (échantillons par seconde).

### 1.2 Nettoyage audio avancé (Suppression du bruit/de la musique) - *Optionnel mais recommandé*

-   **Objectif :** Supprimer les sons de fond indésirables comme le bruit (ronflement, sifflement, ventilateurs), la musique, la réverbération ou d'autres voix interférentes de votre audio source, afin d'isoler autant que possible la voix du locuteur cible. Cette étape est cruciale si votre audio source n'est pas de qualité studio.
-   **Pourquoi ?** Les modèles TTS apprennent à partir de l'audio qu'on leur fournit. Si l'audio contient du bruit de fond ou de la musique, la voix TTS résultante héritera probablement de ces caractéristiques, sonnant bruyante ou « boueuse ». Un audio plus propre conduit à une voix TTS plus propre.

-   **Outils et techniques :**
    *   **Outils de séparation de sources par IA (Recommandés pour la musique/voix) :** Ces outils utilisent des modèles d'IA pour séparer l'audio en différentes pistes (voix, musique, batterie, basse, autres).
        *   **[Ultimate Vocal Remover (UVR)](https://ultimatevocalremover.com/)** : Une application GUI populaire, gratuite et open source qui donne accès à divers modèles de séparation par IA à la pointe de la technologie. Elle est excellente pour supprimer la musique de fond ou isoler les dialogues.
            *   **Modèles (comme ceux mentionnés) :** UVR vous permet d'utiliser différents modèles d'IA. `MDX-Inst-HQ3` est l'un de ces modèles souvent efficace pour séparer les voix des instruments (d'où « Inst »). D'autres modèles MDX, les modèles Demucs (comme `htdemucs`) et potentiellement des modèles comme Mel-Roformer (s'ils sont intégrés ou disponibles de manière autonome) sont conçus pour des tâches similaires, chacun avec des forces et des faiblesses légèrement différentes. L'expérimentation est essentielle. Choisissez des modèles axés sur l'**isolation vocale**.
        *   **Autres outils :** Des services en ligne (par exemple, Lalal.ai) ou d'autres logiciels autonomes peuvent utiliser des modèles sous-jacents similaires (souvent des variantes de Demucs ou Spleeter).
    *   **Outils traditionnels de réduction du bruit :** Souvent présents dans les stations de travail audio numériques (DAW) ou les éditeurs audio.
        *   **[Audacity](https://www.audacityteam.org/) :** Contient des effets intégrés de réduction du bruit (nécessite d'échantillonner un profil de bruit). Peut être efficace contre le bruit de fond constant (comme le sifflement ou le ronflement).
        *   **Plugins commerciaux (par exemple, Izotope RX, Waves Clarity) :** Offrent des outils plus sophistiqués d'isolation du bruit, de la réverbération et de la voix alimentés par l'IA, mais sont payants.
    *   **Édition spectrale :** Suppression manuelle des sons indésirables dans un éditeur spectral (comme Adobe Audition, Izotope RX, Acon Digital Acoustica). Puissant mais très chronophage.

-   **Considérations relatives au flux de travail :**
    *   **Quand l'appliquer :** Il est généralement recommandé d'appliquer le nettoyage à vos **fichiers audio plus longs *avant* le découpage (Étape 1.3 ci-dessous)**. Cela permet aux modèles d'IA de travailler avec plus de contexte et peut être plus efficace que de traiter des milliers de petits segments. Cependant, si le nettoyage introduit trop d'artefacts sur les fichiers longs, vous pourriez essayer de nettoyer individuellement les segments problématiques ultérieurement.
    *   **Processus :**
        1.  Chargez votre fichier WAV standardisé (issu de l'Étape 1.1) dans l'outil choisi (par exemple, UVR).
        2.  Sélectionnez un modèle d'isolation vocale approprié (par exemple, un modèle vocal MDX ou Demucs).
        3.  Traitez l'audio pour générer une piste « voix uniquement ».
        4.  **Écoutez attentivement :** Évaluez de manière critique la piste vocale séparée. Vérifiez :
            *   **Les artefacts :** La séparation par IA peut parfois introduire des sons « aqueux », des glitches ou des parties de la voix supprimées par erreur.
            *   **Le bruit/la musique résiduels :** Avec quelle efficacité le son indésirable a-t-il été supprimé ?
        5.  **Itérez :** Vous devrez peut-être essayer différents modèles, ajuster les paramètres au sein de l'outil, ou même appliquer une seconde passe de réduction du bruit (par exemple, en utilisant la réduction du bruit d'Audacity sur les voix séparées par l'IA) pour obtenir les meilleurs résultats.
    *   **Enregistrer la sortie :** Enregistrez la piste vocale nettoyée sous forme d'un nouveau fichier WAV (par exemple, `original_file_cleaned.wav`). Utilisez ces fichiers nettoyés comme entrée pour l'étape *suivante* (Découpage).

-   **Mises en garde :**
    *   **Des artefacts sont possibles :** Un nettoyage agressif peut dégrader le naturel de la voix cible. Visez un équilibre entre la suppression du bruit et la préservation de la qualité vocale.
    *   **Coût de calcul :** Les modèles de séparation par IA peuvent être gourmands en calcul et prendre un temps considérable, en particulier sur les fichiers audio longs et sans GPU puissant.


### 1.3. Découpage audio (Division en segments)

-   **Objectif :** Diviser les longs fichiers audio (comme les chapitres d'un livre audio ou les épisodes de podcast) en segments plus courts et gérables. La durée idéale d'un segment se situe généralement entre **2 et 15 secondes**.
-   **Pourquoi découper ?**
    *   Aligne la durée audio sur les longueurs de phrases typiques.
    *   Rend la transcription réalisable (transcrire des fichiers de plusieurs heures est difficile).
    *   Aide à gérer la mémoire pendant l'entraînement.
    *   Permet de filtrer les segments inadaptés (par exemple, silence pur, bruit, musique).
-   **Méthode :** Utilisez des outils qui détectent le silence pour diviser l'audio. `pydub` est une bibliothèque Python populaire pour cela.

    ```python
    # Exemple utilisant pydub pour la division basée sur le silence
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import os

    input_file = "resampled_mono_audio.wav" # Utilisez la sortie de l'étape 1.1
    output_dir = "audio_chunks"             # Créez ce répertoire
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading audio file: {input_file}")
    sound = AudioSegment.from_wav(input_file)
    print("Audio loaded. Splitting based on silence...")

    chunks = split_on_silence(
        sound,
        min_silence_len=500,    # Durée minimale de silence en millisecondes pour déclencher une division. À ajuster selon les besoins.
        silence_thresh=-40,     # Seuil de silence en dBFS (décibels par rapport à la pleine échelle). Des valeurs plus basses (par ex. -50) détectent des silences plus faibles. À ajuster selon le niveau de bruit de fond de votre audio.
        keep_silence=200        # Optionnel : Quantité de silence (en ms) à laisser au début/à la fin de chaque segment. Aide à éviter les coupures abruptes.
    )

    print(f"Found {len(chunks)} potential chunks before duration filtering.")

    # --- Filtrage et exportation ---
    min_duration_sec = 2.0  # Durée minimale du segment en secondes
    max_duration_sec = 15.0 # Durée maximale du segment en secondes
    target_sr = 22050       # Assure que les segments conservent le bon sampling rate (pydub gère généralement cela)

    exported_count = 0
    for i, chunk in enumerate(chunks):
        duration_sec = len(chunk) / 1000.0
        if min_duration_sec <= duration_sec <= max_duration_sec:
            # Assure que le segment utilise le sampling rate cible si nécessaire (pydub essaie de le préserver)
            # chunk = chunk.set_frame_rate(target_sr) # Généralement inutile si la source était correctement échantillonnée
            
            chunk_filename = f"segment_{exported_count:05d}.wav" # Utilise un remplissage pour faciliter le tri
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            print(f"Exporting chunk {i} ({duration_sec:.2f}s) to {chunk_path}")
            chunk.export(chunk_path, format="wav")
            exported_count += 1
        else:
             print(f"Skipping chunk {i} due to duration: {duration_sec:.2f}s")


    print(f"\nExported {exported_count} chunks meeting duration criteria ({min_duration_sec}-{max_duration_sec}s) to '{output_dir}'.")
    ```
-   **Vérification :** Écoutez un échantillon des segments générés. Les divisions sont-elles logiques ? La parole est-elle coupée ? Ajustez `min_silence_len` et `silence_thresh` et relancez si nécessaire. Une division manuelle ou un affinement des divisions dans un éditeur audio (comme Audacity) peut être nécessaire pour les audios délicats.

### 1.4. Normalisation du volume

-   **Objectif :** Garantir que tous les segments audio ont un niveau de volume cohérent. Cela empêche les segments trop faibles ou trop forts d'affecter l'entraînement de manière disproportionnée.
-   **Méthodes :**
    *   **Normalisation de crête (Peak) :** Ajuste l'audio pour que le point le plus fort atteigne un niveau spécifique (par exemple, -3.0 dBFS). Simple, mais ne garantit pas une sonie *perçue* cohérente.
    *   **Normalisation de la sonie (LUFS) :** Ajuste l'audio pour atteindre un niveau de sonie perçue cible (par exemple, -23 LUFS est courant pour la diffusion). Généralement préférée car elle reflète mieux l'audition humaine. Nécessite des bibliothèques comme `pyloudnorm`.
-   **Appliquer de manière cohérente :** Appliquez la méthode de normalisation choisie à *tous* les segments créés à l'étape précédente. Enregistrez les fichiers normalisés dans un **nouveau répertoire** (par exemple, `normalized_chunks`) pour conserver les originaux intacts.

    ```python
    # Exemple utilisant pydub pour la normalisation de CRÊTE (PEAK)
    from pydub import AudioSegment
    import os
    import glob

    input_chunk_dir = "audio_chunks"
    output_norm_dir = "normalized_chunks"
    os.makedirs(output_norm_dir, exist_ok=True)
    
    target_dBFS = -3.0 # Amplitude de crête cible

    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    print(f"Normalizing chunks from '{input_chunk_dir}' to '{output_norm_dir}' with target peak {target_dBFS} dBFS.")
    
    wav_files = glob.glob(os.path.join(input_chunk_dir, "*.wav"))
    
    for i, wav_file in enumerate(wav_files):
        filename = os.path.basename(wav_file)
        output_path = os.path.join(output_norm_dir, filename)
        
        try:
            sound = AudioSegment.from_wav(wav_file)
            # N'applique le gain que si le son n'est pas silencieux (dBFS n'est pas -inf)
            if sound.dBFS > -float('inf'):
              normalized_sound = match_target_amplitude(sound, target_dBFS)
              normalized_sound.export(output_path, format="wav")
            else:
              print(f"Skipping silent file: {filename}")
              # Optionnellement, copier les fichiers silencieux ou les gérer selon les besoins
              # shutil.copy(wav_file, output_path) 
            
            if (i + 1) % 50 == 0: # Affiche la progression
                 print(f"Processed {i+1}/{len(wav_files)} files...")
                 
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nNormalization complete. Normalized files saved in '{output_norm_dir}'.")
    ```
    *   **Remarque :** Pour la normalisation LUFS, vous utiliseriez une bibliothèque comme `pyloudnorm`, en itérant sur les fichiers de manière similaire.

### 1.5. Transcription : création des paires de texte

-   **Objectif :** Obtenir une transcription textuelle précise pour *chaque segment audio normalisé*. Le texte doit représenter *exactement* ce qui est dit dans l'audio.
-   **Méthodes :**
    *   **Reconnaissance automatique de la parole (ASR) :** Idéale pour les grands jeux de données. Utilisez des modèles ASR de haute qualité.
    *   **[OpenAI Whisper](https://github.com/openai/whisper) :** Excellente option multilingue et open source. S'exécute localement (GPU recommandé) ou via API. *Remarque : Bien que puissant pour la précision des mots, la ponctuation et la capitalisation de Whisper peuvent nécessiter une révision et une correction soigneuses lors de l'étape de nettoyage.* Divers modèles Whisper affinés par la communauté (souvent trouvés sur Hugging Face) peuvent offrir des améliorations.
    *   **[Modèles Google Gemini](https://ai.google.dev/) (par exemple, via API ou AI Studio) :** Des modèles comme Gemini Pro ou Flash peuvent effectuer la transcription audio. Nécessitent souvent que l'audio soit dans des formats spécifiques et peuvent être plus performants sur des segments plus courts (s'alignant bien avec l'étape de pré-découpage). Vérifiez les offres API actuelles et les éventuels niveaux gratuits.
    *   **Services cloud :** Google Cloud Speech-to-Text, AWS Transcribe, Azure Speech Service proposent des API robustes, souvent avec une tarification à l'usage et potentiellement des niveaux gratuits au départ.
    *   **Autres modèles :** Explorez les [modèles Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition) pour d'autres modèles ASR open source ou affinés spécifiques à votre langue.
    *   **Transcription manuelle :** La plus précise mais très chronophage. Convient aux jeux de données petits et de grande valeur ou pour *corriger les sorties ASR*.
    *   **Transcriptions existantes :** Si votre audio source est accompagné de transcriptions alignées (par exemple, certains livres audio, archives de diffusion), vous aurez peut-être besoin de scripts pour les analyser et les aligner avec vos segments.
-   **Format de sortie :** Créez un fichier `.txt` pour chaque fichier `.wav` correspondant dans votre répertoire `normalized_chunks`. Les noms de fichiers doivent correspondre exactement (par exemple, `normalized_chunks/segment_00001.wav` nécessite `transcripts/segment_00001.txt`).
-   **Nettoyage et normalisation du texte :** **C'est crucial !**
    *   **Supprimer le non-vocal :** Supprimez les horodatages (comme `[00:01:05]`), les étiquettes de locuteur (« SPEAKER A: », « John Doe: »), les balises d'événements sonores (`[laughter]`, `[music]`), les commentaires de transcription.
    *   **Gérer les mots de remplissage :** Décidez de conserver ou de supprimer les mots de remplissage courants (« uh », « um », « ah »). Les conserver peut rendre le TTS plus naturel mais peut aussi introduire des hésitations indésirables. Les supprimer conduit à une parole plus propre et plus directe. La cohérence est essentielle.
    *   **Ponctuation :** Assurez une ponctuation cohérente et appropriée. Les virgules, points et points d'interrogation aident le modèle à apprendre la prosodie. Évitez une ponctuation excessive ou non standard.
    *   **Nombres, acronymes, symboles :** Développez-les en mots (par exemple, « 101 » -> « one hundred one », « USA » -> « U S A » ou « United States of America », « % » -> « percent »). La façon de les développer dépend de la manière dont vous souhaitez que le TTS les prononce. Créez un dictionnaire/jeu de règles de normalisation si nécessaire.
    *   **Casse :** Convertissez généralement le texte en une casse cohérente (par exemple, minuscules) à moins que votre framework/tokenizer TTS ne gère la casse de manière appropriée. Consultez la documentation du framework.
    *   **Caractères spéciaux :** Supprimez ou remplacez les caractères susceptibles de perturber le tokenizer (par exemple, les emojis, les caractères de contrôle).

    ```
    # Exemple de structure :
    my_tts_dataset/
    ├── normalized_chunks/
    │   ├── segment_00001.wav
    │   ├── segment_00002.wav
    │   └── ...
    └── transcripts/
        ├── segment_00001.txt  # Contient "Hello world."
        ├── segment_00002.txt  # Contient "This is a test sentence."
        └── ...
    ```

### 1.6. Structuration des données et création du fichier manifest

-   **Objectif :** Créer des fichiers d'index (manifests) qui indiquent au script d'entraînement TTS où trouver les fichiers audio et leurs transcriptions correspondantes.
-   **Format du manifest :** Le format le plus courant est un fichier texte brut où chaque ligne représente une paire audio-texte, séparée par un délimiteur (généralement une barre verticale `|`).
    ```
    path/to/audio_chunk.wav|The corresponding transcription text|speaker_id
    ```
    *   `path/to/audio_chunk.wav` : Chemin relatif vers le fichier audio normalisé depuis le répertoire où le script d'entraînement sera exécuté.
    *   `The corresponding transcription text` : Le texte nettoyé et normalisé issu du fichier `.txt`.
    *   `speaker_id` : Un identifiant pour le locuteur (par exemple, `speaker0`, `mary_smith`). Pour les jeux de données à locuteur unique, utilisez le même ID pour toutes les lignes. Pour les jeux de données multi-locuteurs, utilisez des ID uniques pour chaque locuteur distinct.
-   **Division des données (Entraînement/Validation) :** Divisez vos données en un jeu d'entraînement (utilisé pour mettre à jour les poids du modèle) et un jeu de validation (utilisé pour surveiller les performances sur des données inédites et éviter l'overfitting). Une division courante est de 90 à 98 % pour l'entraînement et de 2 à 10 % pour la validation. **Point crucial : assurez-vous, dans la mesure du possible, que des segments provenant *du même enregistrement long d'origine* ne se retrouvent pas à la fois dans les jeux d'entraînement et de validation, afin d'éviter les fuites de données.** En cas de division aléatoire, mélangez d'abord.
-   **Script de génération des manifests :**

    ```python
    import os
    import random

    # --- Configuration ---
    dataset_name = "my_tts_dataset"
    normalized_audio_dir = os.path.join(dataset_name, "normalized_chunks")
    transcripts_dir = os.path.join(dataset_name, "transcripts")
    output_dir = dataset_name # Où les fichiers manifest seront enregistrés

    train_manifest_path = os.path.join(output_dir, "train_list.txt")
    val_manifest_path = os.path.join(output_dir, "val_list.txt")

    speaker_id = "main_speaker" # Utilisez un ID cohérent pour les jeux de données à locuteur unique
                                # Pour le multi-locuteurs, déterminez l'ID selon le nom de fichier ou la source
    val_split_ratio = 0.05    # 5 % pour le jeu de validation
    random_seed = 42          # Pour des divisions reproductibles
    # ---------------------

    manifest_entries = []
    print("Reading audio and transcript files...")

    # Itère sur les fichiers audio normalisés
    wav_files = sorted([f for f in os.listdir(normalized_audio_dir) if f.endswith(".wav")])

    for wav_filename in wav_files:
        base_filename = os.path.splitext(wav_filename)[0]
        txt_filename = base_filename + ".txt"
        
        audio_path = os.path.join(normalized_audio_dir, wav_filename)
        # Utilisez os.path.relpath si votre script d'entraînement s'exécute depuis une racine différente
        # relative_audio_path = os.path.relpath(audio_path, start=training_script_dir) 
        relative_audio_path = audio_path # En supposant que le script s'exécute depuis la racine contenant 'my_tts_dataset'

        transcript_path = os.path.join(transcripts_dir, txt_filename)

        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                # Nettoyage de base : supprime les caractères pipe, supprime les espaces superflus
                transcript = transcript.replace('|', ' ').strip()
                transcript = ' '.join(transcript.split()) # Normalise les espaces

                if transcript: # S'assure que la transcription n'est pas vide après nettoyage
                    manifest_entries.append(f"{relative_audio_path}|{transcript}|{speaker_id}")
                else:
                    print(f"Warning: Empty transcript for {wav_filename}. Skipping.")
            except Exception as e:
                print(f"Error reading or processing transcript {txt_filename}: {e}. Skipping.")
        else:
            print(f"Warning: Missing transcript file {txt_filename} for {wav_filename}. Skipping.")

    print(f"Found {len(manifest_entries)} valid audio-transcript pairs.")

    # Mélange et division
    random.seed(random_seed)
    random.shuffle(manifest_entries)

    split_idx = int(len(manifest_entries) * (1 - val_split_ratio))
    train_entries = manifest_entries[:split_idx]
    val_entries = manifest_entries[split_idx:]

    # Écrit les fichiers manifest
    try:
        with open(train_manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_entries))
        print(f"Successfully wrote {len(train_entries)} entries to {train_manifest_path}")

        with open(val_manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_entries))
        print(f"Successfully wrote {len(val_entries)} entries to {val_manifest_path}")
    except Exception as e:
        print(f"Error writing manifest files: {e}")

    ```

---

## 2. Checklist de qualité des données

Avant de passer à la configuration de l'entraînement, examinez rigoureusement votre jeu de données préparé à l'aide de cette checklist. Corriger les problèmes maintenant vous fera gagner un temps considérable par la suite.

| Aspect                  | Vérification                                                          | Pourquoi est-ce important ?                                | Action en cas d'échec                                                                  |
| :---------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Complétude audio**    | Tous les fichiers `.wav` listés dans les manifests existent-ils réellement ? | L'entraînement plantera si des fichiers sont manquants.    | Régénérez les manifests ; vérifiez les chemins ; assurez-vous qu'aucun fichier n'a été supprimé par accident. |
| **Correspondance des transcriptions** | Chaque `.wav` a-t-il un `.txt`/une transcription correspondant(e) et précis(e) ? | Des paires incohérentes enseignent au modèle des associations incorrectes. | Vérifiez les noms de fichiers ; révisez la sortie ASR ; corrigez manuellement les transcriptions. |
| **Durée audio**         | La plupart des segments sont-ils dans la plage souhaitée (par ex. 2-15 s) ? Peu de valeurs aberrantes ? | Les segments très courts/longs peuvent déstabiliser l'entraînement. | Relancez le découpage avec des paramètres ajustés ; filtrez manuellement les valeurs aberrantes des manifests. |
| **Qualité audio**       | Écoutez des échantillons aléatoires : Faible bruit de fond ? Pas de musique/réverbération/écho ? | Garbage In, Garbage Out. Le modèle apprend le bruit.       | Améliorez l'audio source ; appliquez une réduction du bruit (avec précaution !) ; filtrez les mauvais segments. |
| **Cohérence du locuteur** | Pour le locuteur unique : Est-ce toujours la voix cible ? Pas d'autres locuteurs ? | Évite la dilution ou l'instabilité de la voix.             | Révisez/filtrez manuellement les segments ; vérifiez les limites de découpage.        |
| **Format et spécifications** | Tout en WAV ? Sampling rate **identique** ? Canaux mono ? PCM 16 bits ? | Les incohérences provoquent des erreurs ou de mauvaises performances. | Relancez les étapes de conversion/rééchantillonnage (Section 1.1). Vérifiez les spécifications par lots avec des outils en ligne de commande comme `ffprobe` ou `soxi` (qui fait partie du paquet [SoX](http://sox.sourceforge.net/)). Exemple : `soxi -r *.wav` pour vérifier les fréquences. |
| **Niveaux de volume**   | Écoutez des échantillons aléatoires : Les volumes sont-ils relativement cohérents ? | Des variations de volume drastiques peuvent entraver l'apprentissage. | Relancez la normalisation (Section 1.3) ; vérifiez les paramètres de normalisation.   |
| **Propreté de la transcription** | Pas d'horodatages, d'étiquettes de locuteur ? Mots de remplissage gérés de manière cohérente ? Ponctuation standard ? Nombres/symboles développés ? | Garantit que le texte correspond proprement aux sons/à la prosodie de la parole. | Relancez les scripts de nettoyage de texte ; effectuez une révision et une correction manuelles. |
| **Format du manifest**  | Structure `path|text|speaker_id` correcte ? Chemins valides ? Pas de lignes superflues ? | Les erreurs de parseur empêcheront le chargement des données. | Vérifiez le délimiteur (`|`) ; validez les chemins relatifs à l'emplacement du script d'entraînement ; vérifiez l'encodage (UTF-8 de préférence). |
| **Division Entraînement/Validation** | Les fichiers de validation sont-ils vraiment inédits pendant l'entraînement ? Pas de chevauchement ? | Des données qui se chevauchent donnent des scores de validation trompeurs. | Assurez-vous d'un mélange aléatoire avant la division ; vérifiez la logique de division. |

**Astuce :** Utilisez des outils comme `soxi` (de SoX) ou `ffprobe` pour vérifier par lots les propriétés audio (sampling rate, canaux, durée). Écrivez de petits scripts pour vérifier l'existence des fichiers et le formatage de base du manifest.

### 2.1. Scripts de vérification pratiques

Voici quelques scripts pratiques pour vous aider à vérifier la qualité de votre jeu de données :

#### Vérifier les propriétés audio (Sampling Rate, Canaux, Durée)

```bash
#!/bin/bash
# verify_audio.sh - Vérifie les propriétés audio de tous les fichiers WAV
# Usage : ./verify_audio.sh /path/to/audio/directory

AUDIO_DIR="$1"
echo "Checking audio files in $AUDIO_DIR..."

# Vérifie si SoX est installé
if ! command -v soxi &> /dev/null; then
    echo "SoX not found. Please install it first (e.g., 'apt-get install sox' or 'brew install sox')."
    exit 1
fi

# Initialise les compteurs et les tableaux
total_files=0
non_mono=0
wrong_rate=0
too_short=0
too_long=0
target_rate=22050  # Modifiez ceci selon votre sampling rate cible
min_duration=1.0   # Durée minimale en secondes
max_duration=15.0  # Durée maximale en secondes

# Traite tous les fichiers WAV
find "$AUDIO_DIR" -name "*.wav" | while read -r file; do
    total_files=$((total_files + 1))
    
    # Récupère les propriétés audio
    channels=$(soxi -c "$file")
    rate=$(soxi -r "$file")
    duration=$(soxi -d "$file" | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
    
    # Vérifie les propriétés
    if [ "$channels" -ne 1 ]; then
        echo "WARNING: Non-mono file: $file (channels: $channels)"
        non_mono=$((non_mono + 1))
    fi
    
    if [ "$rate" -ne "$target_rate" ]; then
        echo "WARNING: Wrong sampling rate: $file (rate: $rate Hz, expected: $target_rate Hz)"
        wrong_rate=$((wrong_rate + 1))
    fi
    
    if (( $(echo "$duration < $min_duration" | bc -l) )); then
        echo "WARNING: File too short: $file (duration: ${duration}s, minimum: ${min_duration}s)"
        too_short=$((too_short + 1))
    fi
    
    if (( $(echo "$duration > $max_duration" | bc -l) )); then
        echo "WARNING: File too long: $file (duration: ${duration}s, maximum: ${max_duration}s)"
        too_long=$((too_long + 1))
    fi
    
    # Affiche la progression tous les 100 fichiers
    if [ $((total_files % 100)) -eq 0 ]; then
        echo "Processed $total_files files..."
    fi
done

# Affiche le résumé
echo "===== SUMMARY ====="
echo "Total files checked: $total_files"
echo "Non-mono files: $non_mono"
echo "Files with wrong sampling rate: $wrong_rate"
echo "Files too short (<${min_duration}s): $too_short"
echo "Files too long (>${max_duration}s): $too_long"

if [ $((non_mono + wrong_rate + too_short + too_long)) -eq 0 ]; then
    echo "All files passed basic checks!"
else
    echo "Some issues were found. Please review the warnings above."
fi
```

#### Vérifier l'intégrité du fichier manifest

```python
#!/usr/bin/env python3
# verify_manifest.py - Vérifie que tous les fichiers du manifest existent et ont des transcriptions correspondantes
# Usage : python verify_manifest.py path/to/manifest.txt

import os
import sys
from pathlib import Path

def verify_manifest(manifest_path):
    """Vérifie que tous les fichiers audio et transcriptions du manifest existent et sont valides."""
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        return False
    
    print(f"Verifying manifest: {manifest_path}")
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    
    # Statistiques
    total_entries = 0
    missing_audio = 0
    empty_transcripts = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_entries += 1
            
            # Analyse la ligne (en supposant un format séparé par des pipes : audio_path|transcript|speaker_id)
            parts = line.split('|')
            if len(parts) < 2:
                print(f"Line {line_num}: Invalid format. Expected at least 'audio_path|transcript'")
                continue
            
            audio_path = parts[0]
            transcript = parts[1]
            
            # Vérifie si le chemin audio est relatif et le résout
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(base_dir, audio_path)
            
            # Vérifie si le fichier audio existe
            if not os.path.exists(audio_path):
                print(f"Line {line_num}: Audio file not found: {audio_path}")
                missing_audio += 1
            
            # Vérifie si la transcription est vide
            if not transcript or transcript.isspace():
                print(f"Line {line_num}: Empty transcript for {audio_path}")
                empty_transcripts += 1
    
    # Affiche le résumé
    print("\n===== SUMMARY =====")
    print(f"Total entries: {total_entries}")
    print(f"Missing audio files: {missing_audio}")
    print(f"Empty transcripts: {empty_transcripts}")
    
    if missing_audio == 0 and empty_transcripts == 0:
        print("All manifest entries are valid!")
        return True
    else:
        print("Issues found in manifest. Please fix them before proceeding.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_manifest.py path/to/manifest.txt")
        sys.exit(1)
    
    success = verify_manifest(sys.argv[1])
    sys.exit(0 if success else 1)
```

#### Visualiser les spectrogrammes audio pour évaluer la qualité

Ce script vous aide à inspecter visuellement la qualité de vos fichiers audio en générant des spectrogrammes :

```python
#!/usr/bin/env python3
# generate_spectrograms.py - Crée des spectrogrammes pour évaluer la qualité audio
# Usage : python generate_spectrograms.py /path/to/audio/directory /path/to/output/directory [num_samples]

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

def generate_spectrograms(audio_dir, output_dir, num_samples=10):
    """Génère des spectrogrammes pour un échantillon aléatoire de fichiers audio."""
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Récupère tous les fichiers WAV
    wav_files = list(Path(audio_dir).glob('**/*.wav'))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return False
    
    # Échantillonne les fichiers s'il y en a plus que demandé
    if len(wav_files) > num_samples:
        wav_files = random.sample(wav_files, num_samples)
    
    print(f"Generating spectrograms for {len(wav_files)} files...")
    
    for i, wav_path in enumerate(wav_files):
        try:
            # Charge le fichier audio
            y, sr = librosa.load(wav_path, sr=None)
            
            # Crée une figure avec deux sous-graphiques
            plt.figure(figsize=(12, 8))
            
            # Trace la forme d'onde
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform: {wav_path.name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Trace le spectrogramme
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram')
            
            # Enregistre la figure
            output_path = os.path.join(output_dir, f'spectrogram_{i+1}_{wav_path.stem}.png')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            print(f"Generated: {output_path}")
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
    
    print(f"Spectrograms saved to {output_dir}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_spectrograms.py /path/to/audio/directory /path/to/output/directory [num_samples]")
        sys.exit(1)
    
    audio_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    success = generate_spectrograms(audio_dir, output_dir, num_samples)
    sys.exit(0 if success else 1)
```

Ces scripts fournissent des outils pratiques pour vérifier la qualité de votre jeu de données avant l'entraînement, vous aidant à identifier et corriger les problèmes tôt dans le processus.

---

Une fois que votre jeu de données a passé ce contrôle de qualité, vous êtes prêt à passer à la configuration de l'environnement d'entraînement.

**Étape suivante :** [Configuration de l'entraînement](./2_TRAINING_SETUP.md){: .btn .btn-primary} |
[Retour en haut](#top){: .btn .btn-primary}
