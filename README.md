# FEniCS 2026 - Advanced Tutorial Session 

Welcome to the repository with tutorials for the hands-on session at the [FEniCS
2026](https://fenicsproject.org/fenics-2026/) conference! This session is
designed for participants who are already familiar with the basic usage of
[FEniCSx](https://fenicsproject.org/documentation/).

## Schedule 

The Advanced Tutorial Session is tailored for FEniCS 2026 participants and is
scheduled for the first day of the conference, June 17, 2026, before the main
sessions begin.

It features two hands-on practice sessions, followed by a Q&A where you can ask
FEniCS developers any questions you may have about using FEniCSx. You can find
the detailed schedule below:

| Time    | Segment                                   |
| ------- | ----------------------------------------- |
| 8:45 - 9:20 | Arrival/Registration of tutorial session participants |
| 9:20 - 10:10 | [Tutorial session 1 with Jørgen S. Dokken: An exploration of advanced features in DOLFINx through the shifted boundary method](./session1/shifted_fem_intro.ipynb)                |
| 10:10 - 10:40 | Coffee break                              |
| 10:40 - 11:30 | [Tutorial session 2 with Jack S. Hale: A guide to building and running FEniCSx on HPC systems](./session2/session2.md)          |
| 11:30 - 12:00 | Open QA session.      |
| 12:00 - 13:20   | Lunch outside centre/Arrival/Registration of participants                              |

The conference officially starts at 13:20, see the [conference
programme](https://fenicsproject.org/fenics-2026/programme/).

## How to use

### Tutorial session 1

To run the scripts from tutorial 1 it is easiest to download a container image
with FEniCSx v0.11 and JupyterLab using `docker` or `podman`

```shell
docker run -ti -p 8888:8888 ghcr.io/fenics/dolfinx/lab:v0.11.0
```

The lab environment can be accessed at `localhost:8888`.

Install the required dependencies in a Terminal:

```shell
apt-get update
pip install scifem>=0.20
```

and then clone this repository

```shell
git clone https://github.com/a-latyshev/fenics26-tutorials.git
```

The scripts in `session1/` and can be run with `python` or opened in the
JupyterLab environment.

(Optional) to build the book via Jupyter Book 2 (MyST), follow instructions:
[README_AUTHORS](./README_AUTHORS.md).

### Tutorial session 2

Ensure that you have a container runtime installed (e.g. `docker` or `podman`)
and pull the following image with Spack pre-installed:

```bash
docker pull spack/ubuntu-noble:develop
```

## Authors

- Jørgen S. Dokken, Simula Research Laboratory
- Jack S. Hale, University of Luxembourg
- Andrey Latyshev, University of Luxembourg and Sorbonne Université

## License

[Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
