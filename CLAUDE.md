# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

This project uses `uv`. All dependency and environment management goes through it.

```bash
uv sync                  # install dependencies into .venv
uv run python main.py    # run a script in the managed environment
uv add <package>         # add a dependency
```

Python version is pinned to 3.14 (`.python-version`).

## Project Purpose

This is a RAG (Retrieval-Augmented Generation) project being built on top of the English Wikipedia corpus. The Wikipedia dataset is loaded via HuggingFace `datasets` (`wikimedia/wikipedia`, `20231101.en` split).

## Key Scripts

- `download-wikipedia.py` — downloads the full English Wikipedia dataset from HuggingFace Hub using `load_dataset`. This dataset is large; streaming or batching may be needed.
- `main.py` — project entry point, currently a placeholder.
