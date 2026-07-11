# Translation Guide

English is the canonical source for the tutorial structure. Update the English page first, then propagate the same structure, sections, examples, and links to every translation. Synchronize one guide at a time so a large content update can be reviewed and tested safely.

## Available Languages

- **English** — current canonical source
- [Български](bg/index.md) — translated by AcTePuKc
- [Español](es/index.md), [Français](fr/index.md), and [Italiano](it/index.md) — contributed by [@therealpan](https://github.com/therealpan)

The language selector in the site header is generated from these language folders. A new language must be added to `docs/javascripts/site-navigation.js` as well as to the documentation tree.

Every language must use this exact file tree:

```text
docs/<language-code>/
├── index.md
├── glossary.md
├── licence.md
├── contributing-translations.md
└── guides/
    ├── 1-data-preparation.md
    ├── 2-training-setup.md
    ├── 3-model-training.md
    ├── 4-inference.md
    ├── 5-packaging-and-sharing.md
    └── 6-troubleshooting-and-resources.md
```

Current examples are `docs/bg/`, `docs/es/`, `docs/fr/`, and `docs/it/`. Use one of these as a reference for the directory shape, but use the English files in `docs/` as the content source.

The translated home page must mirror the English home page's three-part structure: title/introduction, “Start here”, and “Choose your path”. The six guide files must stay in the same order and retain the same major headings, diagrams, checklists, and code examples. Translate prose and headings, but do not rename files or change link targets without a specific reason.

The glossary, licence, and translation manual are pages in their own language directory; do not keep them only inside a README. Their first heading must be the localized page title so MkDocs does not generate an English title above it.

The repository uses MkDocs Material. Published documentation comes from the Markdown files under `docs/`; use those source files and their `.md` links. MkDocs generates the final site URLs. For a new language, add its code and localized labels to `docs/javascripts/site-navigation.js` so the top navigation, sidebar, table of contents, and language selector can resolve that language.

For example, a Bulgarian page uses:

```text
docs/bg/index.md
docs/bg/guides/1-data-preparation.md
docs/bg/glossary.md
```

and a Spanish page uses:

```text
docs/es/index.md
docs/es/guides/1-data-preparation.md
docs/es/glossary.md
```

Before opening a translation pull request, compare the language folder with the English folder and confirm that all ten files exist, that the guide sections are in the same order, that glossary anchors used by Guide 6 exist, and that `mkdocs build --strict --clean` passes.

## Contribution Steps

1. **Fork the repository** to your own GitHub account.
2. **Create the language folder** under `docs/<language-code>/` using the exact tree above. The language code should be the ISO 639-1 two-letter code where one exists.
3. **Translate the English source** one page at a time. Preserve Markdown structure, diagrams, checklists, code examples, command flags, file names, and glossary anchor IDs. Code examples and comments may remain in English when translating them would make them less useful.
4. **Update navigation support** in `docs/javascripts/site-navigation.js` with the language label, guide titles, and localized interface labels.
5. **Build and check the site locally**, then submit a pull request with the translation.

## Notes for Translators

- Technical terms can be difficult to translate. When in doubt, keep the English term and add a short explanation in the target language.
- Keep the same tone, scope, and level of technical detail as the English source.
- If the English source contains an error or needs a content improvement, open a separate issue instead of silently changing the meaning in one translation.
- When English changes, update the corresponding translated page rather than creating a new alternative file outside the language folder.

## Translation Priorities

1. Preserve meaning before style.
2. Keep glossary-linked technical terms consistent across pages.
3. Preserve Mermaid diagrams, checklists, warnings, and code examples.
4. Report broken Markdown or broken links immediately instead of silently rewording around them.
