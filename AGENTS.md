# Research code working rules

- Write Python-first, well-documented and reusable functions in .py files and demonstrate each stage with real visible output in numbered Jupyter notebooks.
- Follow the maintainer's existing style: clear section comments and Args/Returns/Shapes docstrings in English.
- Generic readers and inspection live in code/python/src/core; dataset-specific scientific logic belongs in studies/<study>/src.
- Avoid speculative abstractions, oversized packages, redundant inspection scripts and unnecessary dependencies.
- Validate code with unit tests and actual dataset runs where possible; do not invent model results.
- Preserve source data rights, scientific provenance and group-independent validation. Never upload raw clinical/participant data or credentials.
