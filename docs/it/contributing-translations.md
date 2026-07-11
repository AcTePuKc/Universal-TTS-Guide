<a id="translation-guide"></a>
# Guida alla Traduzione {#translation-guide}

## Struttura attuale di MkDocs

I file inglesi in `docs/` sono la fonte canonica. Le traduzioni disponibili si trovano in `docs/bg/`, `docs/es/`, `docs/fr/` e `docs/it/`. Una nuova traduzione deve usare `docs/[language_code]/index.md`, i sei file in `guides/` e anche `glossary.md`, `licence.md` e `contributing-translations.md`. Usa i file Markdown e i link `.md`; MkDocs genera gli URL finali.

Accogliamo con piacere le traduzioni di questa guida per renderla accessibile a un pubblico più ampio. Se desideri contribuire con una traduzione, segui questi passaggi:

1. **Effettua il fork del repository** sul tuo account GitHub
2. **Crea la struttura di directory necessaria** per la tua lingua:
   ```
   docs/[language_code]/
   ├── index.md
   └── guides/
       ├── 1-data-preparation.md
       ├── 2-training-setup.md
       └── ... (all guide files)
   ```
   Dove `[language_code]` è il codice ISO 639-1 a due lettere per la tua lingua (ad esempio `es` per lo spagnolo)

3. **Traduci il contenuto** iniziando dall'index.md e poi dai singoli file delle guide
   - Mantieni la stessa struttura dei file e la stessa formattazione Markdown
   - Mantieni invariati tutti gli esempi di codice (dovrebbero rimanere in inglese)
   - Traduci tutto il testo esplicativo, le intestazioni e i commenti

4. **Aggiorna i link di navigazione** in modo che puntino ai file corretti all'interno della directory della tua lingua

5. **Invia una Pull Request** con la tua traduzione

**Note Importanti per i Traduttori:**
- I termini tecnici possono essere difficili da tradurre. In caso di dubbio, puoi mantenere il termine inglese seguito da una breve spiegazione nella tua lingua.
- Cerca di mantenere lo stesso tono e lo stesso livello di dettaglio tecnico dell'originale.
- Se mentre traduci trovi errori o aree di miglioramento nel contenuto originale in inglese, apri una issue separata per affrontarli.
