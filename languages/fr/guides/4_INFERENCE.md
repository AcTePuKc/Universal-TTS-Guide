# Guide 4 : Inférence - Générer de la parole avec votre modèle

**Navigation :** [README principal]({{ site.baseurl }}/languages/fr/){: .btn .btn-primary} | [Étape précédente : Entraînement du modèle](./3_MODEL_TRAINING.md){: .btn .btn-primary} | [Étape suivante : Packaging et partage](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 

Vous avez entraîné ou affiné avec succès un modèle TTS et sélectionné un checkpoint prometteur ! Maintenant, utilisons ce modèle pour convertir un nouveau texte en audio de parole, un processus appelé **inférence** ou **synthèse**.

---

## 7. Inférence : synthétiser la parole

Cette section explique comment exécuter le processus d'inférence à l'aide de votre modèle entraîné.

### 7.1. Localiser le script d'inférence et le meilleur checkpoint

-   **Script d'inférence :** Trouvez le script Python au sein de votre framework TTS conçu pour générer de l'audio. Les noms courants incluent `inference.py`, `synthesize.py`, `infer.py`, `tts.py`.
-   **Meilleur checkpoint :** Identifiez le chemin vers le checkpoint du modèle (`.pth`, `.pt`, `.ckpt`) que vous souhaitez utiliser. Il s'agit généralement de celui sauvegardé sous `best_model.pth` (basé sur la validation loss) ou d'un autre checkpoint que vous avez sélectionné en écoutant les échantillons de validation pendant l'entraînement. Il se trouvera dans votre répertoire de sortie d'entraînement (par exemple, `../checkpoints/my_yoruba_voice_run1/best_model.pth`).
-   **Fichier de configuration :** Vous aurez presque toujours besoin du fichier de configuration (`.yaml`, `.json`) qui a été utilisé *pendant l'entraînement* du checkpoint que vous utilisez. Le script d'inférence en a besoin pour connaître l'architecture du modèle, les paramètres audio (comme le sampling rate) et d'autres réglages. Souvent, une copie de la config est sauvegardée à côté des checkpoints.

### 7.2. Inférence basique d'une phrase unique

-   **Objectif :** Générer de l'audio pour un seul morceau de texte fourni directement via la ligne de commande.
-   **Structure de la commande :** Les arguments exacts varieront, mais une commande typique ressemble à ceci :

    ```bash
    # Activez d'abord votre environnement virtuel !
    # Exemple de commande :
    python inference.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --text "Hello, this is a test of my custom trained voice." \
      --output_wav_path ./output_sample.wav
      # Arguments optionnels/dépendants du framework ci-dessous :
      # --speaker_id "main_speaker"  # Nécessaire pour les modèles multi-locuteurs
      # --device "cuda"              # Pour spécifier l'utilisation du GPU (souvent par défaut)
    ```
-   **Arguments clés :**
    *   `--config` ou `-c` : Chemin vers le fichier de configuration d'entraînement.
    *   `--checkpoint_path` ou `--model_path` ou `-m` : Chemin vers le fichier de checkpoint du modèle.
    *   `--text` ou `-t` : La phrase d'entrée que vous souhaitez synthétiser. Pensez à la mettre entre guillemets.
    *   `--output_wav_path` ou `--out_path` ou `-o` : Le chemin et le nom de fichier souhaités pour le fichier WAV généré.
    *   `--speaker_id` ou `--spk` : **Requis** si vous avez entraîné un modèle multi-locuteurs. Fournissez l'identifiant de locuteur exact utilisé dans vos fichiers manifest pour la voix souhaitée. Pour les modèles à locuteur unique, cela peut être optionnel ou ignoré.
    *   `--device` : Souvent optionnel, par défaut `cuda` si disponible, sinon `cpu`. L'inférence est bien plus rapide sur GPU.

-   **Exécution :** Lancez la commande. Elle chargera le modèle, traitera le texte, générera la forme d'onde audio et la sauvegardera dans le fichier de sortie spécifié. Écoutez le fichier WAV de sortie pour vérifier la qualité.

### 7.3. Inférence par lots (Synthèse à partir d'un fichier)

-   **Objectif :** Générer de l'audio pour plusieurs phrases listées dans un fichier texte, en sauvegardant chacune dans un fichier WAV distinct.
-   **Préparer le fichier d'entrée :** Créez un fichier texte brut (par exemple, `sentences.txt`) où chaque ligne contient une phrase que vous souhaitez synthétiser :
    ```text
    Ceci est la première phrase.
    Voici une autre phrase à synthétiser.
    Le modèle devrait gérer différents signes de ponctuation, comme les questions ?
    Et aussi les exclamations !
    ```
-   **Structure de la commande :** De nombreux frameworks fournissent un script séparé ou des arguments spécifiques pour le traitement par lots.

    ```bash
    # Exemple de commande (le nom du script et les arguments peuvent varier) :
    python inference_batch.py \
      --config ../checkpoints/my_yoruba_voice_run1/config.yaml \
      --checkpoint_path ../checkpoints/my_yoruba_voice_run1/best_model.pth \
      --input_file sentences.txt \
      --output_dir ./generated_batch_audio/
      # Arguments optionnels/dépendants du framework ci-dessous :
      # --speaker_id "main_speaker"  # Nécessaire pour les modèles multi-locuteurs
      # --device "cuda"
    ```
-   **Arguments clés :**
    *   `--input_file` ou `--text_file` : Chemin vers le fichier texte contenant les phrases (une par ligne).
    *   `--output_dir` ou `--out_dir` : Chemin vers le répertoire où les fichiers WAV générés doivent être sauvegardés. Assurez-vous que ce répertoire existe ou que le script le crée. Les noms des fichiers de sortie sont souvent basés sur le numéro de ligne ou le texte d'entrée lui-même (par exemple, `output_0.wav`, `output_1.wav`).
    *   Les autres arguments (`--config`, `--checkpoint_path`, `--speaker_id`, `--device`) sont généralement les mêmes que pour l'inférence d'une phrase unique.

-   **Exécution :** Lancez la commande. Le script itérera sur chaque ligne du fichier d'entrée, synthétisera l'audio et sauvegardera les résultats dans le répertoire de sortie spécifié.

### 7.4. Inférence de modèle multi-locuteurs

-   Comme mentionné ci-dessus, si votre modèle a été entraîné sur des données de plusieurs locuteurs, vous **devez** spécifier quelle voix de locuteur vous souhaitez utiliser pendant l'inférence.
-   Utilisez l'argument `--speaker_id` (ou son équivalent), en fournissant l'ID exact qui correspond au locuteur souhaité dans vos fichiers manifest d'entraînement (par exemple, `speaker0`, `mary_smith`, `yoruba_male_spk1`).
-   Si vous omettez l'ID de locuteur pour un modèle multi-locuteurs, le script peut échouer, utiliser par défaut un locuteur spécifique (souvent le locuteur 0) ou produire des résultats moyennés/brouillés.

### 7.5. Contrôles d'inférence avancés (Dépendants du framework)

-   Certains modèles et frameworks TTS avancés offrent des contrôles supplémentaires pendant l'inférence, souvent passés comme arguments en ligne de commande ou paramètres dans une API Python :
    *   **Débit/Vitesse de parole :** Des arguments comme `--speed` ou `--length_scale` peuvent vous permettre de faire parler la voix plus vite ou plus lentement (par exemple, `1.0` est normal, `<1.0` est plus rapide, `>1.0` est plus lent).
    *   **Contrôle de la hauteur (Pitch) :** Moins courant, mais certains modèles peuvent autoriser des ajustements de la hauteur.
    *   **Contrôle du style/de l'émotion :** Si le modèle a été entraîné avec des style tokens ou des capacités d'audio de référence (comme StyleTTS2 ou les modèles avec des style embeddings), vous pourriez fournir des arguments comme `--style_text` ou `--style_wav` pour influencer la prosodie ou l'émotion de la sortie.
    *   **Réglages du vocoder (si applicable) :** Pour les anciens modèles de type Tacotron2 ou d'autres utilisant des modèles de vocoder séparés (comme HiFi-GAN, MelGAN), il peut y avoir des réglages relatifs au vocoder (par exemple, l'intensité du débruitage).
    *   **Modèles de diffusion :** Pour les modèles TTS basés sur la diffusion, des paramètres contrôlant le nombre de steps de diffusion (échangeant la qualité contre la vitesse) peuvent être disponibles.
-   **Consulter la documentation :** Référez-vous toujours à la documentation de votre framework TTS spécifique ou à l'aide du script d'inférence (`python inference.py --help`) pour voir quels contrôles sont disponibles.

### 7.6. Problèmes d'inférence potentiels

-   **CUDA Out-of-Memory (OOM) :** Même si l'entraînement a fonctionné, des phrases très longues lors de l'inférence peuvent consommer plus de mémoire. Essayez des phrases plus courtes ou vérifiez si le framework offre des options de synthèse segmentée. L'exécution sur CPU (`--device cpu`) utilise la RAM système mais est nettement plus lente.
-   **Incompatibilité modèle/config :** Utiliser un checkpoint avec le mauvais fichier de configuration est une erreur courante, conduisant à des échecs de chargement ou à une sortie parasite. Assurez-vous qu'ils correspondent à la même exécution d'entraînement.
-   **ID de locuteur incorrect :** Fournir un ID de locuteur inexistant pour les modèles multi-locuteurs provoquera des erreurs.
-   **Problèmes de qualité (Bruit, Instabilité) :** Si la qualité de sortie est médiocre, revisitez le Guide 1 (Préparation des données) et le Guide 3 (Entraînement du modèle). Cela peut indiquer des problèmes de qualité des données, un entraînement insuffisant ou le choix d'un checkpoint sous-optimal.

---

## 8. Évaluation et déploiement du modèle

### 8.1. Évaluer la qualité d'un modèle TTS

Bien que les tests d'écoute subjectifs soient la référence absolue pour l'évaluation TTS, il existe aussi des métriques objectives qui peuvent aider à quantifier les performances de votre modèle :

#### Métriques d'évaluation objectives

| Métrique | Ce qu'elle mesure | Outils/Implémentation | Interprétation |
|:-------|:-----------------|:---------------------|:---------------|
| **MOS (Mean Opinion Score)** | Qualité perçue globale | Des évaluateurs humains notent les échantillons sur une échelle de 1 à 5 | Plus c'est élevé, mieux c'est ; norme du secteur mais nécessite des évaluateurs humains |
| **PESQ (Perceptual Evaluation of Speech Quality)** | Qualité audio comparée à une référence | Disponible en Python via `pypesq` | Plage : -0,5 à 4,5 ; plus c'est élevé, mieux c'est |
| **STOI (Short-Time Objective Intelligibility)** | Intelligibilité de la parole | Disponible en Python via `pystoi` | Plage : 0 à 1 ; plus c'est élevé, mieux c'est |
| **Taux d'erreur sur les caractères/mots (CER/WER)** | Intelligibilité via ASR | Exécuter l'ASR sur la parole synthétisée et comparer au texte d'entrée | Plus c'est bas, mieux c'est ; mesure si les mots sont prononcés correctement |
| **Mel Cepstral Distortion (MCD)** | Distance spectrale par rapport à la référence | Implémentation personnalisée avec librosa | Plus c'est bas, mieux c'est ; généralement 2-8 pour les systèmes TTS |
| **F0 RMSE** | Précision de la hauteur | Implémentation personnalisée avec librosa | Plus c'est bas, mieux c'est ; mesure la précision du contour de hauteur |
| **Voicing Decision Error** | Précision voisé/non voisé | Implémentation personnalisée | Plus c'est bas, mieux c'est ; mesure si la parole/le silence est correctement placé |

#### Approche pratique d'évaluation

1. **Préparer le jeu de test** : Créez un ensemble de phrases de test variées non vues pendant l'entraînement
   ```
   # Exemple de test_sentences.txt
   Ceci est une simple phrase déclarative.
   Est-ce une phrase interrogative ?
   Ouah ! Ceci est une phrase exclamative !
   Cette phrase contient des nombres comme 123 et des symboles comme %.
   Ceci est une phrase bien plus longue qui se poursuit pendant un certain temps, testant la capacité du modèle à maintenir la cohérence et une prosodie correcte sur des énoncés plus longs comportant plusieurs propositions et locutions.
   ```

2. **Générer les échantillons** : Utilisez votre modèle pour synthétiser la parole de toutes les phrases de test

3. **Réaliser des tests d'écoute** : Faites noter les échantillons par plusieurs auditeurs sur :
   - Le naturel (échelle de 1 à 5)
   - La qualité audio/les artefacts (échelle de 1 à 5)
   - La précision de la prononciation (échelle de 1 à 5)
   - La similarité du locuteur (échelle de 1 à 5, si vous clonez une voix spécifique)

4. **Implémenter des métriques objectives** : Ce snippet Python montre comment calculer quelques métriques de base :

   ```python
   import numpy as np
   import librosa
   from pesq import pesq
   from pystoi import stoi
   import torch
   from transformers import pipeline

   def evaluate_tts_sample(generated_audio_path, reference_audio_path=None, original_text=None):
       """Évalue un échantillon TTS à l'aide de diverses métriques."""
       results = {}
       
       # Charge l'audio généré
       y_gen, sr_gen = librosa.load(generated_audio_path, sr=None)
       
       # Statistiques audio de base
       results["duration"] = librosa.get_duration(y=y_gen, sr=sr_gen)
       results["rms_energy"] = np.sqrt(np.mean(y_gen**2))
       
       # Si l'audio de référence est disponible, calcule les métriques de comparaison
       if reference_audio_path:
           y_ref, sr_ref = librosa.load(reference_audio_path, sr=sr_gen)  # Fait correspondre les sampling rates
           
           # Assure la même longueur pour la comparaison
           min_len = min(len(y_gen), len(y_ref))
           y_gen_trim = y_gen[:min_len]
           y_ref_trim = y_ref[:min_len]
           
           # PESQ (nécessite de l'audio en 16kHz ou 8kHz)
           if sr_gen in [8000, 16000]:
               try:
                   results["pesq"] = pesq(sr_gen, y_ref_trim, y_gen_trim, 'wb')
               except Exception as e:
                   results["pesq"] = f"Error: {str(e)}"
           else:
               results["pesq"] = "Requires 8kHz or 16kHz audio"
           
           # STOI
           try:
               results["stoi"] = stoi(y_ref_trim, y_gen_trim, sr_gen, extended=False)
           except Exception as e:
               results["stoi"] = f"Error: {str(e)}"
       
       # Si le texte original est disponible, effectue l'ASR et calcule WER/CER
       if original_text:
           try:
               # Charge le modèle ASR (nécessite transformers et torch)
               asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
               
               # Transcrit l'audio généré
               transcription = asr(generated_audio_path)["text"].strip().lower()
               original_text = original_text.strip().lower()
               
               results["transcription"] = transcription
               results["original_text"] = original_text
               
               # Calcul simple du taux d'erreur sur les caractères
               def cer(ref, hyp):
                   ref, hyp = ref.lower(), hyp.lower()
                   return levenshtein_distance(ref, hyp) / len(ref)
               
               def levenshtein_distance(s1, s2):
                   if len(s1) < len(s2):
                       return levenshtein_distance(s2, s1)
                   if len(s2) == 0:
                       return len(s1)
                   previous_row = range(len(s2) + 1)
                   for i, c1 in enumerate(s1):
                       current_row = [i + 1]
                       for j, c2 in enumerate(s2):
                           insertions = previous_row[j + 1] + 1
                           deletions = current_row[j] + 1
                           substitutions = previous_row[j] + (c1 != c2)
                           current_row.append(min(insertions, deletions, substitutions))
                       previous_row = current_row
                   return previous_row[-1]
               
               results["character_error_rate"] = cer(original_text, transcription)
           except Exception as e:
               results["asr_error"] = str(e)
       
       return results
   ```

### 8.2. Déployer des modèles TTS

Une fois votre modèle entraîné et évalué, vous pourriez vouloir le déployer pour une utilisation pratique. Voici quelques options de déploiement :

#### Considérations pour le déploiement en production

Lors du passage de l'expérimentation au déploiement en production, prenez en compte ces facteurs importants :

1. **Optimisation du modèle**
   - **Quantification** : Réduire la précision du modèle de FP32 à FP16 ou INT8 pour diminuer la taille et augmenter la vitesse d'inférence
   - **Élagage (Pruning)** : Supprimer les poids inutiles pour créer des modèles plus petits et plus rapides
   - **Distillation des connaissances** : Entraîner un modèle « élève » plus petit pour imiter votre modèle « enseignant » plus grand
   - **Conversion ONNX** : Convertir votre modèle PyTorch/TensorFlow au format ONNX pour de meilleures performances multiplateformes

2. **Optimisation de la latence**
   - **Traitement par lots** : Pour les applications non temps réel, traiter plusieurs requêtes par lots
   - **Synthèse en streaming** : Pour les applications temps réel, implémenter un traitement segment par segment
   - **Mise en cache** : Mettre en cache les phrases ou séquences de phonèmes fréquemment demandées
   - **Accélération matérielle** : Utiliser GPU/TPU pour le traitement parallèle ou du matériel spécialisé comme NVIDIA TensorRT

3. **Évolutivité (Scalabilité)**
   - **Conteneurisation** : Empaqueter votre modèle et ses dépendances dans des conteneurs Docker
   - **Kubernetes** : Orchestrer plusieurs conteneurs pour la haute disponibilité et l'équilibrage de charge
   - **Mise à l'échelle automatique (Auto-scaling)** : Ajuster automatiquement les ressources en fonction de la demande
   - **Systèmes de file d'attente** : Implémenter des files d'attente de requêtes (RabbitMQ, Kafka) pour gérer les pics de trafic

4. **Surveillance et maintenance**
   - **Métriques de performance** : Suivre la latence, le débit, les taux d'erreur et l'utilisation des ressources
   - **Surveillance de la qualité** : Échantillonner et évaluer périodiquement la qualité de sortie
   - **Tests A/B** : Comparer différentes versions de modèle en production
   - **Entraînement continu** : Mettre en place des pipelines pour réentraîner les modèles avec de nouvelles données

#### Exemple d'architecture de déploiement en production

```
[Applications clientes] → [Équilibreur de charge] → [Passerelle API]
                                             ↓
[Validation des requêtes] → [Limitation de débit] → [Authentification]
                                             ↓
[File d'attente des requêtes] → [Pods de workers TTS (Kubernetes)] → [Cache audio]
                         ↓                              ↑
                  [Conteneur du modèle TTS]             |
                         ↓                              |
                  [Post-traitement audio] → [Stockage audio]
```

#### Options de déploiement local

1. **Interface en ligne de commande** : L'approche la plus simple consiste à créer un script qui encapsule le code d'inférence :

   ```python
   # tts_cli.py
   import argparse
   import os
   import torch
   
   # Importez ici vos modules spécifiques au modèle
   # from your_tts_framework import load_model, synthesize_text
   
   def main():
       parser = argparse.ArgumentParser(description="Text-to-Speech CLI")
       parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
       parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
       parser.add_argument("--config", type=str, required=True, help="Path to model config")
       parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
       parser.add_argument("--speaker", type=str, default=None, help="Speaker ID for multi-speaker models")
       args = parser.parse_args()
       
       # Charge le modèle (l'implémentation dépend de votre framework)
       model = load_model(args.model, args.config)
       
       # Synthétise la parole
       audio = synthesize_text(model, args.text, speaker_id=args.speaker)
       
       # Sauvegarde l'audio
       save_audio(audio, args.output)
       print(f"Audio saved to {args.output}")
   
   if __name__ == "__main__":
       main()
   ```

2. **Interface web simple** : Créez une interface web basique avec Flask ou Gradio :

   ```python
   # app.py (exemple Flask)
   from flask import Flask, request, send_file, render_template
   import os
   import torch
   import uuid
   
   # Importez ici vos modules spécifiques au modèle
   # from your_tts_framework import load_model, synthesize_text
   
   app = Flask(__name__)
   
   # Charge le modèle au démarrage (pour une inférence plus rapide)
   MODEL_PATH = "path/to/best_model.pth"
   CONFIG_PATH = "path/to/config.yaml"
   model = load_model(MODEL_PATH, CONFIG_PATH)
   
   @app.route('/')
   def index():
       return render_template('index.html')
   
   @app.route('/synthesize', methods=['POST'])
   def synthesize():
       text = request.form['text']
       speaker_id = request.form.get('speaker_id', None)
       
       # Génère un nom de fichier unique
       output_file = f"static/audio/{uuid.uuid4()}.wav"
       os.makedirs(os.path.dirname(output_file), exist_ok=True)
       
       # Synthétise la parole
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       
       # Sauvegarde l'audio
       save_audio(audio, output_file)
       
       return {'audio_path': output_file}
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000, debug=True)
   ```

3. **Interface Gradio** (Encore plus simple) :

   ```python
   import gradio as gr
   import torch
   
   # Importez ici vos modules spécifiques au modèle
   # from your_tts_framework import load_model, synthesize_text
   
   # Charge le modèle
   MODEL_PATH = "path/to/best_model.pth"
   CONFIG_PATH = "path/to/config.yaml"
   model = load_model(MODEL_PATH, CONFIG_PATH)
   
   def tts_function(text, speaker_id=None):
       # Synthétise la parole
       audio = synthesize_text(model, text, speaker_id=speaker_id)
       sampling_rate = 22050  # Ajustez selon la fréquence de votre modèle
       return (sampling_rate, audio)
   
   # Crée l'interface Gradio
   iface = gr.Interface(
       fn=tts_function,
       inputs=[
           gr.Textbox(lines=3, placeholder="Enter text to synthesize..."),
           gr.Dropdown(choices=["speaker1", "speaker2"], label="Speaker", visible=True)  # Pour les modèles multi-locuteurs
       ],
       outputs=gr.Audio(type="numpy"),
       title="Text-to-Speech Demo",
       description="Enter text and generate speech using a custom TTS model."
   )
   
   iface.launch(server_name="0.0.0.0", server_port=7860)
   ```

#### Options de déploiement cloud

Pour une utilisation en production, envisagez ces options :

1. **Hugging Face Spaces** : Téléversez votre modèle sur Hugging Face et créez une application Gradio ou Streamlit
2. **API REST** : Encapsulez votre modèle dans une application FastAPI ou Flask et déployez-la sur des services cloud
3. **Fonctions serverless** : Pour les modèles légers, déployez en tant que fonctions serverless (AWS Lambda, Google Cloud Functions)
4. **Conteneurs Docker** : Empaquetez votre modèle et ses dépendances dans un conteneur Docker pour un déploiement cohérent

#### Optimisation des performances

Pour améliorer la vitesse et l'efficacité de l'inférence :

1. **Quantification du modèle** : Convertir les poids du modèle en précision inférieure (FP16 ou INT8)
   ```python
   # Exemple de conversion FP16 avec PyTorch
   model = model.half()  # Convertir en FP16
   ```

2. **Élagage du modèle (Pruning)** : Supprimer les poids inutiles pour créer des modèles plus petits
3. **Conversion ONNX** : Convertir les modèles PyTorch au format ONNX pour une inférence plus rapide
   ```python
   # Exemple d'export ONNX
   import torch.onnx
   
   # Exporte le modèle
   torch.onnx.export(model,               # modèle en cours d'exécution
                     dummy_input,         # entrée du modèle (ou un tuple pour plusieurs entrées)
                     "model.onnx",        # où sauvegarder le modèle
                     export_params=True,  # stocke les poids des paramètres entraînés dans le fichier du modèle
                     opset_version=11,    # la version ONNX vers laquelle exporter le modèle
                     do_constant_folding=True)  # optimisation
   ```

4. **Traitement par lots** : Traiter plusieurs entrées textuelles à la fois pour un débit plus élevé
5. **Mise en cache** : Mettre en cache les sorties fréquemment demandées pour éviter une régénération

Maintenant que vous pouvez générer de la parole à l'aide de votre modèle entraîné, l'étape logique suivante consiste à organiser correctement les fichiers de votre modèle pour une utilisation future, un partage ou un déploiement.

**Étape suivante :** [Packaging et partage](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} | 
[Retour en haut](#top){: .btn .btn-primary}
