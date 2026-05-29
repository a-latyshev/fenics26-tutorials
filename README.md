# FEniCS 2026 - Advanced Tutorial Session 

Welcome to the repository with tutorials for the hands-on session at the [FEniCS
2026](https://fenicsproject.org/fenics-2026/) conference! This session is
designed for participants who are already familiar with the basic usage of
[FEniCSx](https://fenicsproject.org/documentation/). We will explore advanced
features and best practices for using [FEniCSx
v0.10.0](https://github.com/FEniCS/dolfinx/releases/tag/v0.10.0.post0).

## Schedule 

The Advanced Tutorial Session is tailored for FEniCS 2026 participants and is
scheduled for the first day of the conference, June 17, 2026, before the main
talks begin. It features two hands-on practice sessions, followed by a Q&A
where you can ask FEniCS developers any questions you may have about using
FEniCSx. You can find the detailed schedule below:

| Time    | Segment                                   |
| ------- | ----------------------------------------- |
| 8:45 - 9:20 | Arrival/Registration of participants |
| 9:20 - 10:10 | [Practice session 1](./session1/session1.md)                |
| 10:10 - 10:40 | Coffee break                              |
| 10:40 - 11:30 | [Practice session 2](./session2/session2.md)          |
| 11:30 - 12:00 | Open QA session.      |
| 1h20   | Free lunch                                |

The conference officially starts at 13:20, see the [conference
schedule](https://fenicsproject.org/fenics-2026/).

## Installation

### Docker

Install docker container with `dolfinx/dolfinx:v0.10.0` image. See
[instructions](https://github.com/FEniCS/dolfinx). Then install the following
dependencies in the container:

Install npm
```shell
apt-get -y update
apt-get -y install npm
```

Install Python dependencies
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
jupytext --to md:myst session*/*.py
jupyter-book build --strict
```

and to serve as a website:

```shell
jupyter-book start
```

## License


