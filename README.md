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

## Building the book and running code

Make sure that you installed Jupyter Book 2
```shell
pip install "jupyter-book>=2.0.0"
```
If you build from a docker container, make sure that you have a running Jupyter
server

```shell
export JUPYTER_BASE_URL="http://127.0.0.1:8888/"
export JUPYTER_TOKEN="my-jupyter-token"
jupyter server --IdentityProvider.token="${JUPYTER_TOKEN}" --ServerApp.port="8888" --allow-root &
jupyter book start --execute
```

TODO

## For authors

:::{tip} Summary
Basically MyST markdown files replace jupytext `.py` files, which makes the
workflow a bit less straightforward.
:::

Jupyter Book 2 supports only notebooks `.ipynb` and MyST Markdown `.md` files.

Convert jupytext files to MyST Markdown:
```bash
jupytext --from py --to md:myst example.py
```

To preserve version control, AL suggests to continue working with jupytext `.py`
files and then convert them to MyST Markdown.

Authors are encouraged to take a look at [Jupyter Book 2 playground](./jb2_playground.md) to see how
demos look like.