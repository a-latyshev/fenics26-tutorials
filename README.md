# FEniCS 2026 - Advanced Tutorial Session 

Welcome to the repository with tutorials for the hands-on session at the [FEniCS
2026](https://fenicsproject.org/fenics-2026/) conference! This session is
designed for participants who are already familiar with the basic usage of
[FEniCSx](https://fenicsproject.org/documentation/). We will explore advanced
features and best practices for using [FEniCSx
v0.10.0](https://github.com/FEniCS/dolfinx/releases/tag/v0.10.0.post0).

## Schedule 

The Advanced Tutorial Session is tailored for FEniCS 2026 participants and is scheduled for the first day of the conference, June 17, 2026, before the main talks begin. It features two hands-on practice sessions, followed by a Q&A where you can ask FEniCS developers any questions you may have about using FEniCSx. You can find the detailed schedule below:

| Time    | Segment                                   |
| ------- | ----------------------------------------- |
| 8:45 - 9:20 | Arrival/Registration of participants |
| 9:20 - 10:10 | [Practice session 1](./session1/session1.md)                |
| 10:10 - 10:40 | Coffee break                              |
| 10:40 - 11:30 | [Practice session 2](./session2/session2.md)          |
| 11:30 - 12:00 | Open QA session.      |
| 1h20   | Free lunch                                |

The conference officially starts at 13:20, see the conference schedule.

## Installation

### Docker

Install docker container with `dolfinx/dolfinx:v0.10.0` image. See
[instructions](https://github.com/FEniCS/dolfinx). Then install the following dependencies.

Install npm
```shell
apt update
apt install npm -y
```

Install Python dependencies
```shell
pip install -r requirements.txt
```
OR
```shell
pip install "jupyter-book>=2.0.0" "pyvista[jupyter]" jupytext jupyter-server
```
Optional:
```shell
pip install dolfinx-external-operator>=0.10.0
```
TODO: Remove dolfinx-external-operator dependence.


## For authors

:::{tip} Summary 
Jupyter Book 2 (JB2) is based on MyST markdown files and replaces jupytext `.py`
files. JB2's pages are based on either `.ipynb` or `.md` files, which makes the
workflow a bit less straightforward compared to Jupyter Book 1. To preserve
version control, continue working with `.py` files. The current workflow
generates `.ipynb` notebooks automatically. Don't forget to add the
corresponding `.ipynb` files to [myst.yml](./myst.yml).
:::

### Instructions for the authors

1. In the appropriate `session` folder add your `.md` or `.py` files.
2. Add the corresponding JB page to `myst.yml`. **NOTE:** In `myst.yml`, your
   files with the `.py` extension must be replaced with the `.ipynb` extension.
3. Github CI will automatically convert all `.py` files into their `.ipynb`
   counterparts via `jupytext`.

To test your site locally, use the following supplementary function:

```shell
python tools/local_book_build.py --serve --serve-port 8001
```
and access via:
```
http://127.0.0.1:8001/
```
The script `local_book_build.py` converts jupytext files into notebooks,
launches a jupyter server, build the book and serves site from `\html`.
Launching the jupyter server is required to **build** the notebook, because
`a-latyshev` did not find another way to do it. Furthermore, currently
`local_book_build.py` **does not** compile PyVista static scenes.

Authors are encouraged to take a look at [Jupyter Book 2 - key
features](jb2_key_features.md) to see main features of JB2 and JB2-based
standard FEniCSx demos look like.

More about jupyter book execution: https://jupyterbook.org/stable/execution/execution/.

There are issues when the site is statically hosted:
https://github.com/orgs/jupyter-book/projects/1/views/1?pane=issue&itemId=122114744&issue=jupyter-book%7Cmystmd%7C2000.

## License

Which one?