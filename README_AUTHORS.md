# For authors

## Installation

Launch a Docker container with DOLFINx installed

```shell
docker run -ti -p 8888:8888 -e PORT=8888 --name=workshop2026 -v $(pwd):/root/shared -w /root/shared  --entrypoint=/bin/bash ghcr.io/fenics/dolfinx/lab:stable
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

## Local build

1. In the appropriate `session` folder add your `.md` or `.py` files.
2. Add the corresponding page to `myst.yml`. In `myst.yml`, your files with the
   `.py` extension should have `.ipynb` extension.
3. Github CI will automatically convert all `.py` files into their `.ipynb`
   counterparts via `jupytext` before rendering the book.

To build the site locally:

```shell
./build.sh
```

and to serve as a website:

```shell
python -m http.server 8888 -d _build/html/ &
```

## Generate PDF from sources

In the container install minimal `texlive` to support LaTeX

```shell
apt-get update
apt-get install -y texlive-xetex texlive-latex-extra texlive-fonts-recommended latexmk
```

Generate PDF

```shell
export LANG=C.UTF-8 LC_ALL=C.UTF-8 LC_CTYPE=C.UTF-8
jupyter-book build --pdf --execute
```