<a id="translation-guide"></a>
# Guía de Traducción {#translation-guide}

## Estructura actual de MkDocs

Los archivos ingleses de `docs/` son la fuente canónica. Las traducciones disponibles están en `docs/bg/`, `docs/es/`, `docs/fr/` y `docs/it/`. Una nueva traducción debe usar `docs/[language_code]/index.md`, los seis archivos de `guides/` y también `glossary.md`, `licence.md` y `contributing-translations.md`. Usa los archivos Markdown y los enlaces `.md`; MkDocs genera las URL finales.

Damos la bienvenida a las traducciones de esta guía para hacerla accesible a una audiencia más amplia. Si deseas contribuir con una traducción, sigue estos pasos:

1. **Haz un fork del repositorio** a tu propia cuenta de GitHub
2. **Crea la estructura de directorios necesaria** para tu idioma:
   ```
   docs/[language_code]/
   ├── index.md
   └── guides/
       ├── 1-data-preparation.md
       ├── 2-training-setup.md
       └── ... (todos los archivos de la guía)
   ```
   Donde `[language_code]` es el código de dos letras ISO 639-1 de tu idioma (por ejemplo, `es` para español)

3. **Traduce el contenido** comenzando por el index.md y luego los archivos individuales de la guía
   - Mantén la misma estructura de archivos y el mismo formato Markdown
   - Mantén todos los ejemplos de código sin cambios (deben permanecer en inglés)
   - Traduce todo el texto explicativo, los encabezados y los comentarios

4. **Actualiza los enlaces de navegación** para que apunten a los archivos correctos dentro del directorio de tu idioma

5. **Envía un Pull Request** con tu traducción

**Notas Importantes para los Traductores:**
- Los términos técnicos pueden ser difíciles de traducir. En caso de duda, puedes mantener el término en inglés seguido de una breve explicación en tu idioma.
- Intenta mantener el mismo tono y nivel de detalle técnico que el original.
- Si encuentras errores o áreas de mejora en el contenido original en inglés mientras traduces, por favor abre un issue separado para abordarlos.
