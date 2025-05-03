# Ръководство 4: Извеждане с TTS модел

**Навигация:** [Главно README]({{ site.baseurl }}/languages/bg/){: .btn .btn-primary} | [Предишна стъпка: Обучение на модела](./3_MODEL_TRAINING.md){: .btn .btn-primary} |  |[Следваща стъпка: Пакетиране и споделяне на вашия TTS модел](./5_PACKAGING_AND_SHARING.md){: .btn .btn-primary} |

Това ръководство обхваща как да използвате вашия обучен TTS модел за синтезиране на реч от нов текст.

---

## 1. Настройка за извеждане

### 1.1. Избор на най-добрата контролна точка

След обучението, трябва да изберете най-добрата контролна точка за извеждане:

```bash
# Изброяване на наличните контролни точки
ls -la path/to/checkpoints/

# Проверка на метриките за загуба за различни контролни точки
grep "best_loss" path/to/logs/
```

Критерии за избор на най-добрата контролна точка:
-   **Загуба при валидация:** Обикновено контролната точка с най-ниска загуба при валидация е добро начало.
-   **Субективно качество:** Слушайте примери, генерирани от различни контролни точки и изберете този, който звучи най-добре.
-   **Стабилност:** Някои контролни точки с по-ниска загуба могат да имат артефакти или нестабилност.

### 1.2. Подготовка на средата за извеждане

Настройте средата за извеждане:

```bash
# Създаване на директория за изходи
mkdir -p tts_outputs

# Инсталиране на необходимите зависимости (ако не са инсталирани)
pip install numpy torch torchaudio matplotlib
```

## 2. Основно извеждане

### 2.1. Извеждане с различни рамки

#### Coqui TTS
```python
# Пример за извеждане с Coqui TTS
from TTS.utils.synthesizer import Synthesizer

synthesizer = Synthesizer(
    tts_checkpoint="path/to/best_model.pth",
    tts_config_path="path/to/config.json",
    vocoder_checkpoint="path/to/vocoder.pth",
    vocoder_config="path/to/vocoder_config.json"
)

# Синтезиране на реч
wav = synthesizer.tts("Това е тестово изречение на български език.")

# Запазване на аудиото
synthesizer.save_wav(wav, "tts_outputs/output.wav")
```

#### ESPnet
```python
# Пример за извеждане с ESPnet
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

# Зареждане на модела
tts = Text2Speech.from_pretrained(
    model_file="path/to/model.pth",
    train_config="path/to/config.yaml",
    vocoder_file="path/to/vocoder.pth"
)

# Синтезиране на реч
wav = tts("Това е тестово изречение на български език.")["wav"]

# Запазване на аудиото
sf.write("tts_outputs/output.wav", wav.numpy(), tts.fs, "PCM_16")
```

#### VITS
```python
# Пример за извеждане с VITS
import torch
import utils
from models import SynthesizerTrn

# Зареждане на модела
hps = utils.get_hparams_from_file("path/to/config.json")
model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
)
_ = model.eval()
_ = utils.load_checkpoint("path/to/best_model.pth", model)

# Обработка на текста
text = "Това е тестово изречение на български език."
text_norm = text_to_sequence(text, hps.data.text_cleaners)
text_norm = torch.LongTensor(text_norm).unsqueeze(0)

# Синтезиране на реч
with torch.no_grad():
    audio = model.infer(text_norm, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0][0]
    audio = audio.cpu().numpy()

# Запазване на аудиото
import soundfile as sf
sf.write("tts_outputs/output.wav", audio, hps.data.sampling_rate, "PCM_16")
```

### 2.2. Командни инструменти за извеждане

Много рамки предоставят командни инструменти за лесно извеждане:

```bash
# Coqui TTS CLI
tts --text "Това е тестово изречение на български език." \
    --model_path path/to/best_model.pth \
    --config_path path/to/config.json \
    --out_path tts_outputs/output.wav

# ESPnet CLI
espnet_tts --text "Това е тестово изречение на български език." \
    --model_file path/to/model.pth \
    --train_config path/to/config.yaml \
    --out_file tts_outputs/output.wav
```

## 3. Разширено извеждане

### 3.1. Контрол на прозодията

Много TTS модели позволяват контрол върху различни аспекти на прозодията:

```python
# Пример за контрол на скоростта на речта с Coqui TTS
wav = synthesizer.tts(
    "Това е тестово изречение на български език.",
    speed=1.2  # Стойности > 1.0 ускоряват речта, < 1.0 забавят
)

# Пример за контрол на височината с VITS
with torch.no_grad():
    audio = model.infer(
        text_norm,
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=0.8  # Стойности < 1.0 ускоряват речта, > 1.0 забавят
    )[0][0]
```

### 3.2. Многоговорителни модели

Ако сте обучили многоговорителен модел, можете да избирате между различни говорители:

```python
# Пример за многоговорително извеждане с Coqui TTS
wav = synthesizer.tts(
    "Това е тестово изречение на български език.",
    speaker_id=1  # ID на говорителя, който искате да използвате
)

# Пример за многоговорително извеждане с VITS
speaker_id = torch.LongTensor([1])  # ID на говорителя като тензор
with torch.no_grad():
    audio = model.infer(
        text_norm,
        sid=speaker_id,
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0
    )[0][0]
```

### 3.3. Стилов трансфер и емоционален контрол

Някои модели поддържат стилов трансфер или емоционален контрол:

```python
# Пример за стилов трансфер с StyleTTS
style_vector = style_encoder(reference_audio)
with torch.no_grad():
    audio = model.infer(
        text_norm,
        style_vector=style_vector,
        noise_scale=0.667
    )

# Пример за емоционален контрол с емоционален TTS
wav = synthesizer.tts(
    "Това е тестово изречение на български език.",
    emotion="happy"  # Емоция: happy, sad, angry, etc.
)
```

## 4. Пакетно извеждане

За генериране на множество аудио файлове наведнъж:

```python
# Пример за пакетно извеждане с Python
import pandas as pd

# Зареждане на текстове от CSV файл
texts = pd.read_csv("texts_to_synthesize.csv")

# Синтезиране на всеки текст
for idx, row in texts.iterrows():
    text = row["text"]
    output_file = f"tts_outputs/output_{idx}.wav"
    
    # Синтезиране и запазване
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, output_file)
    
    print(f"Генериран {output_file}")
```

Или с bash скрипт:

```bash
#!/bin/bash
# Пакетно извеждане с bash скрипт

# Четене на текстове от файл, по един на ред
while IFS= read -r line; do
    # Генериране на име на изходен файл
    output_file="tts_outputs/output_$(date +%s).wav"
    
    # Извикване на TTS CLI
    tts --text "$line" \
        --model_path path/to/best_model.pth \
        --config_path path/to/config.json \
        --out_path "$output_file"
    
    echo "Генериран $output_file"
    
    # Малка пауза между заявките
    sleep 1
done < "texts_to_synthesize.txt"
```

## 5. Оценка на модела

### 5.1. Субективна оценка

Субективната оценка включва човешки слушатели, оценяващи качеството на синтезираната реч:

-   **MOS (Mean Opinion Score):** Слушателите оценяват качеството по скала от 1 до 5.
-   **MUSHRA (MUltiple Stimuli with Hidden Reference and Anchor):** Слушателите сравняват няколко аудио примера едновременно.
-   **AB тест:** Слушателите избират кой от два примера предпочитат.

```python
# Пример за генериране на примери за MOS тест
test_sentences = [
    "Това е първото тестово изречение за оценка.",
    "Това е второто тестово изречение с различна структура.",
    "Третото изречение съдържа по-сложни думи и интонация."
]

for i, text in enumerate(test_sentences):
    # Генериране с вашия модел
    wav_model = synthesizer.tts(text)
    synthesizer.save_wav(wav_model, f"evaluation/model_sample_{i}.wav")
    
    # Генериране с базов модел за сравнение (ако е наличен)
    wav_baseline = baseline_synthesizer.tts(text)
    baseline_synthesizer.save_wav(wav_baseline, f"evaluation/baseline_sample_{i}.wav")
```

### 5.2. Обективна оценка

Обективните метрики могат да предоставят количествена оценка на вашия модел:

```python
# Пример за изчисляване на обективни метрики
import numpy as np
from pesq import pesq
from pystoi import stoi

# Зареждане на референтно и синтезирано аудио
ref_audio, sr = librosa.load("reference.wav", sr=16000)
syn_audio, sr = librosa.load("synthesized.wav", sr=16000)

# Уверете се, че дължините съвпадат
min_len = min(len(ref_audio), len(syn_audio))
ref_audio = ref_audio[:min_len]
syn_audio = syn_audio[:min_len]

# Изчисляване на PESQ (Perceptual Evaluation of Speech Quality)
pesq_score = pesq(sr, ref_audio, syn_audio, 'wb')  # 'wb' за широколентов режим

# Изчисляване на STOI (Short-Time Objective Intelligibility)
stoi_score = stoi(ref_audio, syn_audio, sr, extended=False)

print(f"PESQ Score: {pesq_score}")
print(f"STOI Score: {stoi_score}")
```

### 5.3. Рамка за цялостна оценка

Създайте рамка за цялостна оценка на вашия модел:

```python
# Пример за рамка за оценка
def evaluate_tts_model(model, test_texts, reference_audios=None):
    """
    Оценява TTS модел по няколко измерения.
    
    Args:
        model: Заредения TTS модел
        test_texts: Списък с тестови изречения
        reference_audios: Опционални референтни аудио файлове за сравнение
    
    Returns:
        Dictionary с резултати от оценката
    """
    results = {
        "objective_metrics": {},
        "synthesis_speed": [],
        "audio_lengths": [],
        "sample_paths": []
    }
    
    # Измерване на скоростта на синтез и генериране на примери
    for i, text in enumerate(test_texts):
        start_time = time.time()
        audio = model.synthesize(text)
        synthesis_time = time.time() - start_time
        
        # Запазване на аудиото
        output_path = f"evaluation/sample_{i}.wav"
        sf.write(output_path, audio, model.sample_rate)
        
        # Запазване на метриките
        results["synthesis_speed"].append(synthesis_time)
        results["audio_lengths"].append(len(audio) / model.sample_rate)
        results["sample_paths"].append(output_path)
        
        # Изчисляване на обективни метрики, ако са налични референтни аудио файлове
        if reference_audios and i < len(reference_audios):
            ref_audio, _ = librosa.load(reference_audios[i], sr=model.sample_rate)
            
            # Уверете се, че дължините съвпадат
            min_len = min(len(ref_audio), len(audio))
            ref_audio = ref_audio[:min_len]
            audio_trimmed = audio[:min_len]
            
            # Изчисляване на метрики
            try:
                pesq_score = pesq(model.sample_rate, ref_audio, audio_trimmed, 'wb')
                results["objective_metrics"][f"pesq_sample_{i}"] = pesq_score
            except Exception as e:
                print(f"Грешка при изчисляване на PESQ за пример {i}: {str(e)}")
    
    # Изчисляване на средни стойности
    results["avg_synthesis_speed"] = np.mean(results["synthesis_speed"])
    results["avg_audio_length"] = np.mean(results["audio_lengths"])
    
    if results["objective_metrics"]:
        results["avg_pesq"] = np.mean([v for k, v in results["objective_metrics"].items() if "pesq" in k])
    
    return results
```

## 6. Оптимизация за продукция

### 6.1. Оптимизация на модела

За по-бързо извеждане в продукционна среда:

```python
# Пример за оптимизация на модел с TorchScript
import torch

# Проследяване и компилиране на модела
traced_model = torch.jit.trace(model, example_inputs)
torch.jit.save(traced_model, "optimized_model.pt")

# Зареждане на оптимизирания модел
optimized_model = torch.jit.load("optimized_model.pt")
```

### 6.2. Квантизация

Намалете размера на модела и ускорете извеждането с квантизация:

```python
# Пример за квантизация на модел
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Запазване на квантизирания модел
torch.save(quantized_model.state_dict(), "quantized_model.pth")
```

### 6.3. Оптимизация на латентността

За приложения в реално време, оптимизирайте латентността:

```python
# Пример за оптимизация на латентността
# 1. Използвайте по-малки размери на входа/изхода
# 2. Предварително заредете модела в паметта
# 3. Използвайте пакетно извеждане, когато е възможно

# Предварително зареждане на модела
model = load_model("path/to/model.pth")
_ = model(dummy_input)  # Загряване на модела

# Пакетно извеждане за множество заявки
def batch_inference(texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Обработка на партидата наведнъж
        batch_results = model(prepare_batch(batch))
        results.extend(batch_results)
    return results
```

### 6.4. Мащабируемост

За обработка на много заявки:

```python
# Пример за мащабируема архитектура с работници
import multiprocessing as mp

def worker_process(queue_in, queue_out):
    # Зареждане на модела веднъж за работника
    model = load_model("path/to/model.pth")
    
    while True:
        # Получаване на заявка
        job_id, text = queue_in.get()
        if job_id == -1:  # Сигнал за спиране
            break
            
        # Обработка на заявката
        try:
            audio = model.synthesize(text)
            queue_out.put((job_id, audio, None))
        except Exception as e:
            queue_out.put((job_id, None, str(e)))

# Създаване на пул от работници
def create_worker_pool(num_workers=4):
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(queue_in, queue_out))
        p.start()
        workers.append(p)
    
    return queue_in, queue_out, workers
```

### 6.5. Мониторинг и поддръжка

Настройте мониторинг за вашата TTS система в продукция:

```python
# Пример за основен мониторинг
import time
import logging
from prometheus_client import Counter, Histogram, start_http_server

# Метрики
tts_requests = Counter('tts_requests_total', 'Total TTS requests')
tts_errors = Counter('tts_errors_total', 'Total TTS errors')
tts_latency = Histogram('tts_latency_seconds', 'TTS request latency')

# Функция за синтез с мониторинг
def synthesize_with_monitoring(text):
    tts_requests.inc()
    start_time = time.time()
    
    try:
        audio = model.synthesize(text)
        tts_latency.observe(time.time() - start_time)
        return audio
    except Exception as e:
        tts_errors.inc()
        logging.error(f"TTS error: {str(e)}")
        raise
```

## 7. Интеграция с други системи

### 7.1. Уеб API

Създайте REST API за вашия TTS модел:

```python
# Пример за Flask API
from flask import Flask, request, send_file
import io
import soundfile as sf

app = Flask(__name__)
model = load_tts_model("path/to/model.pth")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    speaker_id = data.get('speaker_id', 0)
    
    # Синтезиране на реч
    audio = model.synthesize(text, speaker_id=speaker_id)
    
    # Създаване на аудио файл в паметта
    buffer = io.BytesIO()
    sf.write(buffer, audio, model.sample_rate, format='WAV')
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='audio/wav',
        as_attachment=True,
        attachment_filename='synthesized.wav'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 7.2. Интеграция с чатботове

Интегрирайте TTS с чатбот система:

```python
# Пример за интеграция с чатбот
def chatbot_response(user_input):
    # Получаване на текстов отговор от чатбота
    text_response = chatbot.get_response(user_input)
    
    # Синтезиране на аудио отговор
    audio = tts_model.synthesize(text_response)
    
    # Запазване на аудиото
    audio_file = f"responses/response_{int(time.time())}.wav"
    sf.write(audio_file, audio, tts_model.sample_rate)
    
    return {
        "text": text_response,
        "audio_url": audio_file
    }
```

### 7.3. Интеграция с настолни приложения

Интегрирайте TTS в настолно приложение:

```python
# Пример за PyQt интеграция
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit
from PyQt5.QtMultimedia import QSound
import sys

class TTSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTS Application")
        self.setGeometry(100, 100, 400, 300)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setGeometry(10, 10, 380, 200)
        
        self.speak_button = QPushButton("Speak", self)
        self.speak_button.setGeometry(150, 220, 100, 30)
        self.speak_button.clicked.connect(self.speak_text)
        
        self.tts_model = load_tts_model("path/to/model.pth")
    
    def speak_text(self):
        text = self.text_edit.toPlainText()
        if text:
            audio = self.tts_model.synthesize(text)
            sf.write("temp_audio.wav", audio, self.tts_model.sample_rate)
            QSound.play("temp_audio.wav")

app = QApplication(sys.argv)
window = TTSApp()
window.show()
sys.exit(app.exec_())
```

---

## 8. Примерна архитектура за продукционно внедряване

Ето примерна архитектура за внедряване на TTS система в продукционна среда:

```
                                  +----------------+
                                  |  Load Balancer |
                                  +--------+-------+
                                           |
                 +-------------------------+-------------------------+
                 |                         |                         |
        +--------v-------+        +--------v-------+        +--------v-------+
        |  TTS API Server |        |  TTS API Server |        |  TTS API Server |
        | (Flask/FastAPI) |        | (Flask/FastAPI) |        | (Flask/FastAPI) |
        +--------+-------+        +--------+-------+        +--------+-------+
                 |                         |                         |
        +--------v-------+        +--------v-------+        +--------v-------+
        |   TTS Worker   |        |   TTS Worker   |        |   TTS Worker   |
        | (Model Serving)|        | (Model Serving)|        | (Model Serving)|
        +----------------+        +----------------+        +----------------+
                 |                         |                         |
                 +-------------------------v-------------------------+
                                           |
                                  +--------v-------+
                                  |  Redis Cache   |
                                  | (Common Texts) |
                                  +----------------+
                                           |
                                  +--------v-------+
                                  |   Monitoring   |
                                  |  (Prometheus)  |
                                  +----------------+
```

Ключови компоненти:
1. **Load Balancer**: Разпределя заявките между API сървърите
2. **TTS API Servers**: Обработват HTTP заявки и валидират входа
3. **TTS Workers**: Изпълняват действителното TTS извеждане
4. **Redis Cache**: Кешира често използвани текстове за по-бързо извеждане
5. **Monitoring**: Следи производителността и здравето на системата

---

С тези инструменти и техники, вие сте готови да използвате вашия TTS модел в различни приложения, от прости скриптове до сложни продукционни системи.

**Предишна стъпка:** [Обучение на модела](./3_MODEL_TRAINING.md){: .btn .btn-primary} | 
[Обратно към началото](#top){: .btn .btn-primary}