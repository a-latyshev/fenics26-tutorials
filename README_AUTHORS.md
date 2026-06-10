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

### Instructions for authors

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
