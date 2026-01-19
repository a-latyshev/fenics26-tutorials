# FEniCS 2026 - Advanced Tutorial Session 

This is a repository with tutorials for the hands-on session of the [FEniCS
2026](https://fenicsproject.org/fenics-2026/) conference. The session is
intended for participants familiar with the basic usage of
[FEniCSx](https://fenicsproject.org/documentation/). We will explore advanced
features and best practices of using [FEniCSx
v0.10.0](https://github.com/FEniCS/dolfinx/releases/tag/v0.10.0.post0).

## Schedule 

The Advanced Tutorial Session is designed for the participants of FEniCS 2026 in
the first day of the conference, June, 17, 2026, before main talks. It consists
of 2 Practice sessions followed up by a QA session where participants may ask
FEniCS developer their questions regarding the use of FEniCSx. Here, below one
can find a schedule:

| Time    | Segment                                   |
| ------- | ----------------------------------------- |
| 8:45 - 9:20 | Arrival/Registration of participants |
| 9:20 - 10:10 | [Practice session 1](./session1/)                |
| 10:10 - 10:40 | Coffee break                              |
| 10:40 - 11:30 | [Practice session 2](./session2/)          |
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

## For authors

Jupyter Book 2 supports only notebooks `.ipynb` and MyST Markdown `.md` files.

Convert jupytext files to MyST Markdown:
```bash
jupytext --from py --to md:myst example.py
```

To preserve version control, AL suggests to continue working with jupytext `.py`
files and then convert them to MyST Markdown.

Authors are encouraged to take a look at [Jupyter Book 2 playground](./jb2_playground.md) to see how
demos look like.