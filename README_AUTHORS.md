## Installation

### Docker

Launch a Docker container with DOLFINx installed

```shell
docker run -ti -p 3000 dolfinx/dolfinx:stable
```

Install npm

```shell
apt-get -y update
apt-get -y install npm
```

Install extra Python dependencies

```shell
pip install -r requirements.txt
```

## For authors

### Instructions for the authors

1. In the appropriate `session` folder add your `.md` or `.py` files.
2. Add the corresponding page to `myst.yml`. **NOTE:** In `myst.yml`, your
   files with the `.py` extension must be replaced with the `.ipynb` extension.
3. Github CI will automatically convert all `.py` files into their `.ipynb`
   counterparts via `jupytext` before rendering the book.

To build your site locally:

```shell
jupytext --to ipynb session*/*.py
jupyter-book build --strict --execute
```

and to serve as a website:

```shell
jupyter-book start --execute
```
