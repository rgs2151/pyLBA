# Contributing

Install a development environment:

```bash
git clone https://github.com/rgs2151/pyLBA.git
cd pyLBA
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,docs]'
```

Format Python code with `black pyLBA`. Keep changes focused and include a small
reproduction when reporting a bug. Verification should be local and targeted;
the documentation workflow builds the site without running model fitting.

Build the documentation:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

New model classes inherit from `AccumulatorModel`. Parameter classes are frozen
dataclasses derived from `ModelParameters`. Implement the model density,
simulation, parameter class, and prior construction interfaces.
