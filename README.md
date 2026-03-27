# RRA Trading Swarm

Bootstrap scaffold for the Reflexive Reputation Arb (RRA) system.

This repository is being built against the RRA System Specification, with the following architectural constraints already in force:

- Python 3.12 backend managed with `uv`
- LangGraph for durable orchestration and persistent state
- Groq-backed inference for latency-sensitive agent decisions
- ERC-8004-aligned trust and registry integration
- Kraken CLI-native execution patterns for hardened order flow

## Current Status

The repository is in the initial scaffold phase. Core environment management is in place, while strategy nodes, persistence wiring, and execution logic will be added in later phases.

## Repository Layout

```text
Project-Optima/
|-- backend/
|   |-- agent/
|   |   `-- nodes/
|   |-- data/
|   `-- main.py
|-- frontend/
|   |-- app/
|   `-- components/
|-- scripts/
|-- security/
|-- .env
|-- .python-version
|-- pyproject.toml
`-- uv.lock
```

## Environment

The backend runtime is pinned to Python 3.12 and uses `uv` for dependency and virtual environment management.

Typical bootstrap flow:

```powershell
Set-Location D:\APPS\Project-Optima
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$PYTHON_VERSION = "3.12.10"
$PYENV_BAT = "$HOME\.pyenv\pyenv-win\bin\pyenv.bat"

& $PYENV_BAT install $PYTHON_VERSION
& $PYENV_BAT local $PYTHON_VERSION

uv venv .venv --python "$HOME\.pyenv\pyenv-win\versions\$PYTHON_VERSION\python.exe"
.\.venv\Scripts\Activate.ps1
uv sync --all-groups
```

## Dependency Management

Project dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

- Use `uv sync --all-groups` to install runtime and development dependencies.
- Avoid direct `pip install` drift unless there is a specific packaging reason.
- Keep exchange execution logic on the Kraken CLI path, not generic REST wrappers.

## Security Notes

- Do not commit populated `.env` files or exchange credentials.
- Keep local persistence stores, logs, and execution traces out of version control unless explicitly designated as fixtures.
- Preserve deterministic environment files such as `.python-version`, `pyproject.toml`, and `uv.lock`.

## Next Build Areas

- Backend package skeleton and typed settings
- LangGraph state schema and persistent checkpoint integration
- Groq inference node wiring
- ERC-8004 trust and registry adapters
- Hardened Kraken CLI execution node
