# Privacy packing project

This repo contains code to simulate and evaluate different privacy packing policies, in different settings (offline/online, multi-block/single-block). We use Rényi DP only (not epsilon-delta DP).

## Warnings

- We need AutoDP's last version, which is not available on Pip. Clone their repo and run `pip install -e autodp`

## Organization

-  See [privacypacking/](privacypacking/) for some documentation about the code
-  Let's try to add some tests in [tests/](tests/)
-  [notebooks/](notebooks/) for the Jupyter notebooks (that can import the code with `import privacypacking` when the virtual environment is activated)

## Contributing


- Code style: flake8, Black, Google-style docstrings, type annotations (ideally)

- Install [poetry](https://python-poetry.org/) to set up the environment. The `poetry.lock` file will ensure that we all have the exact same packages. Useful commands:
    + `poetry add plotly` to install a package (e.g. Plotly) and update the requirements
    + `poetry update` to update the dependencies, `poetry lock --no-update` to just refresh the lockfile
    + `poetry install` to install the dependencies
    + `poetry shell` to activate the virtual environment

- Use `nb-clean` to clean heavy notebooks if necessary:
    + `nb-clean clean notebooks/*.ipynb` to run by hand
    + `nb-clean add-filter` to run at each Git commit
    + `nb-clean remove-filter` to switch back to manual cleaning (good if the metadata in some notebooks is actually useful, e.g. logs or graphs)
 
