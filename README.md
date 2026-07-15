# kash-docs

See the main [kash](https://github.com/jlevy/kash) repo for general instructions.

To run kash with text, research, and general document actions enabled, ensure you have
uv set up then:

```shell
uv tool install kash-docs --upgrade --force
kash
```

PDF/DOCX conversion and AWS publishing are optional so downstream tools such as
Deep Transcribe do not install unrelated runtimes:

```shell
uv tool install "kash-docs[documents]" --upgrade --force
uv tool install "kash-docs[aws]" --upgrade --force
uv tool install "kash-docs[full]" --upgrade --force
```

The `documents` extra provides MarkItDown, PDF rendering, and DOCX conversion. The
`full` extra also includes layout-aware PDF conversion and browser automation.

Or for dev builds from within this git repo:

```shell
# Install all deps and run tests:
make
# Run kash with all docs kit features enabled:
uv run kash
```

## Model Configuration

Document actions inherit the careful, structured, standard, and fast model roles from
kash-shell rather than selecting a provider directly. See the main
[model configuration documentation](https://github.com/jlevy/kash#model-configuration)
for the current Anthropic defaults and equivalent OpenAI settings.

For how to install uv and Python, see [installation.md](docs/installation.md).

For development workflows, see [development.md](docs/development.md).

For instructions on publishing to PyPI, see [publishing.md](docs/publishing.md).

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*

<!-- This document follows common-doc-guidelines.md.
See github.com/jlevy/practical-prose and review guidelines before editing.
-->
