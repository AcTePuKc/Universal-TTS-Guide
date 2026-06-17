# Guía Universal de Entrenamiento de Modelos TTS y Preparación de Datasets

## Idiomas Disponibles

- [English](../../README.md) (Original)
- **Español** (Actual)
- [Français](../fr/README.md)
- [Italiano](../it/README.md)
- [Български](../bg/README.md)

*¿Quieres contribuir con una traducción? Consulta la [Guía de Traducción](../../README.md#translation-guide) más abajo.*

## Introducción

¡Bienvenido! Esta guía completa ofrece un proceso universal para preparar tus propios datasets de voz y entrenar un modelo de Texto a Voz (TTS) personalizado. Tanto si tienes un dataset pequeño (por ejemplo, 10 horas) como uno más grande (más de 100 horas), estos pasos te ayudarán a organizar tus datos correctamente y a recorrer el proceso de entrenamiento para la mayoría de los frameworks modernos de TTS.

**Objetivo:** Capacitarte para hacer fine-tuning o entrenar un modelo TTS con una voz o un idioma específicos usando tus propios pares de audio y texto.

**Qué Cubre Esta Guía:**
Esta guía está dividida en varias partes, que cubren todo el flujo de trabajo desde la planificación hasta el uso de tu modelo entrenado:

1.  **Planificación:** Consideraciones iniciales antes de comenzar tu proyecto.
2.  **Preparación de Datos:** Adquisición, procesamiento y estructuración de los datos de audio y texto.
3.  **Configuración del Entrenamiento:** Preparación de tu entorno y configuración de los parámetros de entrenamiento.
4.  **Entrenamiento del Modelo:** Lanzamiento, monitorización y fine-tuning del modelo TTS.
5.  **Inferencia:** Uso de tu modelo entrenado para sintetizar voz.
6.  **Empaquetado y Compartición:** Organización y documentación de tu modelo para uso o distribución futuros.
7.  **Resolución de Problemas y Recursos:** Problemas comunes y herramientas útiles.

---

## 0. Antes de Empezar: Planificando tu Dataset

Antes de recopilar datos, considera estos puntos cruciales para asegurarte de que tu proyecto esté bien definido y sea factible:

1.  **Locutor:** ¿Será un único locutor o varios locutores? Los datasets de un solo locutor son más sencillos para empezar con el fine-tuning o el entrenamiento inicial. Los modelos multi-locutor requieren un cuidadoso equilibrio de datos y gestión de los identificadores de locutor (speaker ID).
2.  **Fuente de Datos:** ¿De dónde obtendrás el audio? (Audiolibros, podcasts, archivos de radio, datos de voz grabados profesionalmente, tus propias grabaciones). **De forma crucial, asegúrate de tener los derechos o licencias necesarios para usar los datos en el entrenamiento de modelos.**
3.  **Calidad del Audio:** Busca la mayor calidad posible. Prioriza grabaciones limpias con un mínimo de ruido de fondo, reverberación, música o voces superpuestas. La consistencia en las condiciones de grabación es muy beneficiosa.
4.  **Idioma y Dominio:** ¿Qué idioma(s) hablará el modelo? ¿Cuál es el estilo de habla o el dominio (por ejemplo, narración, conversacional, lectura de noticias)? El modelo rendirá mejor con texto similar al de sus datos de entrenamiento.
5.  **Cantidad de Datos Objetivo:** ¿Cuántos datos planeas recopilar o usar?
    *   **~1-5 horas:** Puede ser suficiente para una *clonación* de voz básica si se usa un modelo preentrenado potente, pero la calidad podría ser limitada.
    *   **~5-20 horas:** Generalmente considerado el mínimo para un *fine-tuning* decente de una voz específica sobre un modelo preentrenado.
    *   **50-100+ horas:** Mejor para entrenar modelos robustos o entrenar modelos con menor dependencia de pesos preentrenados, especialmente para idiomas menos comunes.
    *   **1000+ horas:** Necesario para entrenar modelos de alta calidad y propósito general, en gran medida desde cero.
6.  **Sampling Rate (frecuencia de muestreo):** Decide un sampling rate objetivo (por ejemplo, 16000 Hz, 22050 Hz, 44100 Hz, 48000 Hz) desde el principio. Las frecuencias más altas capturan más detalle, pero requieren más almacenamiento y cómputo. **Todos tus datos de entrenamiento DEBEN usar de forma consistente la frecuencia elegida.** 22050 Hz es un equilibrio común para muchos modelos TTS.

---

## Resumen del Proceso y Navegación

Esta guía está dividida en módulos enfocados. Sigue los enlaces de abajo para ver los pasos detallados de cada fase:

1.  **➡️ [Preparación de Datos](./guides/1_DATA_PREPARATION.md)**
    *   Cubre la adquisición, limpieza, segmentación y normalización del audio, la transcripción del texto y la creación de los archivos manifest necesarios para el entrenamiento. Incluye la crucial checklist de calidad de los datos.

2.  **➡️ [Configuración del Entrenamiento](./guides/2_TRAINING_SETUP.md)**
    *   Te guía a través de la configuración de tu entorno de Python, la instalación de dependencias (como PyTorch con CUDA), la elección de un framework de TTS y la configuración de los parámetros de entrenamiento en tu archivo de configuración.

3.  **➡️ [Entrenamiento del Modelo](./guides/3_MODEL_TRAINING.md)**
    *   Explica cómo lanzar el script de entrenamiento, monitorizar su progreso (loss, validación), reanudar un entrenamiento interrumpido, y ofrece consejos específicos para el fine-tuning de modelos existentes.

4.  **➡️ [Inferencia](./guides/4_INFERENCE.md)**
    *   Detalla cómo usar el checkpoint de tu modelo entrenado para sintetizar voz a partir de texto nuevo, incluyendo frases individuales, procesamiento por lotes y consideraciones multi-locutor.

5.  **➡️ [Empaquetado y Compartición](./guides/5_PACKAGING_AND_SHARING.md)**
    *   Ofrece buenas prácticas para organizar los archivos de tu modelo entrenado (checkpoints, configuraciones, muestras), documentarlos con un README, gestionar versiones y prepararlos para compartir o archivar.

6.  **➡️ [Resolución de Problemas y Recursos](./guides/6_TROUBLESHOOTING_AND_RESOURCES.md)** 
    *   Ofrece soluciones para problemas comunes que surgen durante el entrenamiento y la inferencia, y enumera herramientas, librerías y comunidades externas útiles.

---

## Conclusión

Siguiendo estas guías, obtendrás una comprensión completa del flujo de trabajo para preparar datos y entrenar tus propios modelos de Texto a Voz. Recuerda que una preparación de datos meticulosa es la base de una voz de alta calidad, y que el proceso de entrenamiento a menudo implica un refinamiento iterativo.

Ahora, dirígete a la sección correspondiente según en qué punto del ciclo de vida de tu proyecto te encuentres. ¡Mucha suerte construyendo tus voces personalizadas! 🚀

## Contribuir 

¡Las contribuciones para mejorar esta guía son bienvenidas! Ya sea que encuentres erratas, imprecisiones, tengas sugerencias para explicaciones más claras, quieras añadir información sobre herramientas o frameworks específicos, o tengas ideas para nuevas secciones, tu aporte es valioso.

No dudes en:

*   **Abrir un Issue:** Para reportar errores, sugerir mejoras o discutir posibles cambios.
*   **Enviar un Pull Request:** Para correcciones o adiciones concretas. Por favor, intenta asegurarte de que tus cambios sean claros y estén alineados con la estructura y el tono general de la guía.

¡Agradecemos cualquier esfuerzo por hacer esta guía más precisa, completa y útil para la comunidad!

## Glosario de Términos Técnicos

Este glosario explica los términos técnicos clave usados a lo largo de las guías para ayudar a los recién llegados a entender la terminología:

- **ASR (Automatic Speech Recognition)**: Tecnología que convierte el lenguaje hablado en texto escrito; se utiliza para transcribir datos de audio (Reconocimiento Automático del Habla).
- **Batch Size**: El número de ejemplos de entrenamiento que se procesan juntos en un mismo paso hacia adelante/atrás (forward/backward); afecta a la velocidad de entrenamiento y al uso de memoria (tamaño de lote).
- **Checkpoint**: Una instantánea guardada de los pesos de un modelo durante o después del entrenamiento, que permite reanudar el entrenamiento o usar el modelo para inferencia.
- **CUDA**: La plataforma de computación paralela de NVIDIA que permite la aceleración por GPU para tareas de deep learning.
- **dBFS (Decibels relative to Full Scale)**: Una unidad de medida para los niveles de audio en sistemas digitales, donde 0 dBFS representa el nivel máximo posible (decibelios relativos a la escala completa).
- **Diffusion Models**: Una clase de modelos generativos que gradualmente añaden y luego eliminan ruido de los datos; algunos sistemas TTS recientes usan este enfoque (modelos de difusión).
- **FFT (Fast Fourier Transform)**: Un algoritmo que convierte señales del dominio del tiempo a representaciones del dominio de la frecuencia; fundamental para el procesamiento de audio (Transformada Rápida de Fourier).
- **Fine-tuning**: El proceso de tomar un modelo preentrenado y seguir entrenándolo con un dataset más pequeño y específico para adaptarlo a una nueva voz o idioma (ajuste fino).
- **LUFS (Loudness Units relative to Full Scale)**: Una medida estandarizada de la sonoridad percibida, más representativa de la audición humana que las mediciones de pico.
- **Manifest File**: Un archivo de texto que enumera los archivos de audio y sus transcripciones correspondientes, usado para indicarle al script de entrenamiento dónde encontrar los datos (archivo manifest).
- **Mel Spectrogram**: Una representación visual del audio que aproxima la percepción auditiva humana usando la escala mel; comúnmente usado como representación intermedia en sistemas TTS (espectrograma mel).
- **Overfitting**: Cuando un modelo aprende demasiado bien los datos de entrenamiento, incluyendo su ruido y valores atípicos, lo que resulta en un mal rendimiento con datos nuevos (sobreajuste).
- **Sampling Rate**: El número de muestras de audio por segundo (medido en Hz); las frecuencias más altas capturan más detalle del audio pero requieren más almacenamiento y potencia de procesamiento (frecuencia de muestreo).
- **STFT (Short-Time Fourier Transform)**: Una técnica que determina el contenido de frecuencia de secciones locales de una señal a medida que cambia en el tiempo (Transformada de Fourier de Tiempo Corto).
- **TTS (Text-to-Speech)**: Tecnología que convierte texto escrito en salida de voz hablada (Texto a Voz).
- **Validation Loss**: Una métrica que mide el error de un modelo sobre un dataset de validación (datos no usados para el entrenamiento); ayuda a detectar el overfitting (pérdida de validación).
- **VRAM (Video RAM)**: Memoria de una tarjeta gráfica; los modelos de deep learning y sus cálculos intermedios se almacenan aquí durante el entrenamiento.
- **Vocoder**: Un componente de algunos sistemas TTS que convierte características acústicas (como los espectrogramas mel) en formas de onda (audio real).

## Guía de Traducción

Damos la bienvenida a las traducciones de esta guía para hacerla accesible a una audiencia más amplia. Si deseas contribuir con una traducción, sigue estos pasos:

1. **Haz un fork del repositorio** a tu propia cuenta de GitHub
2. **Crea la estructura de directorios necesaria** para tu idioma:
   ```
   languages/[language_code]/
   ├── README.md
   └── guides/
       ├── 1_DATA_PREPARATION.md
       ├── 2_TRAINING_SETUP.md
       └── ... (todos los archivos de la guía)
   ```
   Donde `[language_code]` es el código de dos letras ISO 639-1 de tu idioma (por ejemplo, `es` para español)

3. **Traduce el contenido** comenzando por el README.md y luego los archivos individuales de la guía
   - Mantén la misma estructura de archivos y el mismo formato Markdown
   - Mantén todos los ejemplos de código sin cambios (deben permanecer en inglés)
   - Traduce todo el texto explicativo, los encabezados y los comentarios

4. **Actualiza los enlaces de navegación** para que apunten a los archivos correctos dentro del directorio de tu idioma

5. **Envía un Pull Request** con tu traducción

**Notas Importantes para los Traductores:**
- Los términos técnicos pueden ser difíciles de traducir. En caso de duda, puedes mantener el término en inglés seguido de una breve explicación en tu idioma.
- Intenta mantener el mismo tono y nivel de detalle técnico que el original.
- Si encuentras errores o áreas de mejora en el contenido original en inglés mientras traduces, por favor abre un issue separado para abordarlos.

## [Licencia](../../LICENCE.md)
El contenido de esta guía está licenciado bajo la [Licencia Internacional Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Eres libre de compartir y adaptar el material siempre que proporciones la atribución adecuada. El contenido también está protegido bajo el copyright de 2025 AcTePuKc y cualquier nueva contribución estará sujeta a la misma licencia.
