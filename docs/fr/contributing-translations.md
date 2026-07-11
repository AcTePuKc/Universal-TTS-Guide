<a id="translation-guide"></a>
# Guide de traduction {#translation-guide}

## Structure actuelle de MkDocs

Les fichiers anglais de `docs/` sont la source canonique. Les traductions disponibles se trouvent dans `docs/bg/`, `docs/es/`, `docs/fr/` et `docs/it/`. Une nouvelle traduction doit utiliser `docs/[language_code]/index.md`, les six fichiers de `guides/`, ainsi que `glossary.md`, `licence.md` et `contributing-translations.md`. Utilisez les fichiers Markdown et les liens `.md` ; MkDocs génère les URL finales.

Nous accueillons favorablement les traductions de ce guide afin de le rendre accessible à un public plus large. Si vous souhaitez contribuer à une traduction, veuillez suivre ces étapes :

1. **Forkez le dépôt** vers votre propre compte GitHub
2. **Créez la structure de répertoires nécessaire** pour votre langue :
   ```
   docs/[language_code]/
   ├── index.md
   └── guides/
       ├── 1-data-preparation.md
       ├── 2-training-setup.md
       └── ... (tous les fichiers de guide)
   ```
   Où `[language_code]` est le code à deux lettres ISO 639-1 de votre langue (par exemple, `es` pour l'espagnol)

3. **Traduisez le contenu** en commençant par le index.md, puis les fichiers de guide individuels
   - Conservez la même structure de fichiers et le même formatage Markdown
   - Laissez tous les exemples de code inchangés (ils doivent rester en anglais)
   - Traduisez tout le texte explicatif, les en-têtes et les commentaires

4. **Mettez à jour les liens de navigation** pour qu'ils pointent vers les bons fichiers dans le répertoire de votre langue

5. **Soumettez une Pull Request** avec votre traduction

**Remarques importantes pour les traducteurs :**
- Les termes techniques peuvent être difficiles à traduire. En cas de doute, vous pouvez conserver le terme anglais suivi d'une brève explication dans votre langue.
- Essayez de conserver le même ton et le même niveau de détail technique que l'original.
- Si vous trouvez des erreurs ou des points à améliorer dans le contenu anglais original lors de la traduction, veuillez ouvrir une issue distincte pour les traiter.
