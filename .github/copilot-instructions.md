# Copilot Instructions for `rl-xbandit-demo`

## Build, test, and lint commands

This repository currently does not define dedicated build, lint, or test
scripts.

- Install dependencies:
  - `pip install -r requirements.txt`
- Run the project interactively:
  - `jupyter lab`

There is no automated test suite configured yet, so there is no single-test
command at this time.

## High-level architecture

- The project is currently notebook-first: the code lives in
  `src/bandit.ipynb` rather than a Python package/module layout.
- `requirements.txt` defines a data-science stack (`numpy`, `pandas`,
  `matplotlib`) plus `jupyterlab`, indicating local exploratory workflows in
  notebooks.
- The root `README.md` is intentionally minimal; operational guidance should be
  inferred from the dependency set and notebook-centric structure.

## Key conventions in this repo

- Python-specific conventions are defined in
  `.github/instructions/python.instructions.md` and should be followed whenever
  `.py` files are added or edited:
  - PEP 8 formatting (including 4-space indentation and line length guidance)
  - PEP 257 docstrings for functions/classes
  - Type annotations using `typing`
  - Clear comments for algorithm logic and edge-case handling
