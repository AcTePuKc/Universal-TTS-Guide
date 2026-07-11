# Universal TTS Guide

Beginner-friendly documentation for preparing speech datasets, training and fine-tuning text-to-speech models, running inference, packaging models, and troubleshooting common problems.

## Documentation

- [Read the published guide](https://actepukc.github.io/Universal-TTS-Guide/)
- [Start with the documentation source](docs/index.md)
- [Guide 1: Data Preparation](docs/guides/1-data-preparation.md)
- [Guide 2: Training Setup](docs/guides/2-training-setup.md)
- [Guide 3: Model Training](docs/guides/3-model-training.md)
- [Guide 4: Inference](docs/guides/4-inference.md)
- [Guide 5: Packaging and Sharing](docs/guides/5-packaging-and-sharing.md)
- [Guide 6: Troubleshooting and Resources](docs/guides/6-troubleshooting-and-resources.md)
- [Glossary](docs/glossary.md)
- [Translation guide](docs/contributing-translations.md)
- [Licence](LICENCE.md)

The site is available in English, Bulgarian, Spanish, French, and Italian. Each language follows the same guide structure so translations can be compared and maintained consistently.

## Repository layout

- `docs/` contains the MkDocs source tree.
- `docs/bg/`, `docs/es/`, `docs/fr/`, and `docs/it/` contain localized pages.
- `mkdocs.yml` configures the documentation site, navigation, languages, and themes.
- `.github/workflows/mkdocs-gh-pages.yml` builds and deploys the site to GitHub Pages.
- `site/` is generated output and is intentionally ignored by Git.

## Contributing

Corrections, clearer explanations, new examples, and translation improvements are welcome. Please keep the shared structure intact, leave executable commands and code identifiers in English, and follow the [translation guide](docs/contributing-translations.md) when adding or updating a language.

## Licence

The guide is released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See [LICENCE.md](LICENCE.md) for the full notice.
