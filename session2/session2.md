---
authors:
- jshale
---

# Ten years of FEniCS on HPC systems 

## Introduction

This guide is an attempt to distill some knowledge built up at the University
of Luxembourg over the past decade around building and running FEniCS on HPC
systems.

During the tutorial session I will go through this material and at the end I
will give a brief interactive demo on installing FEniCS using the
[Spack](https://spack-tutorial.readthedocs.io/en/latest/) package manager, with
a particular focus on the specific aspects important for working with FEniCS.

The most important points will be called out and summarised as a TLDR; at the
end:

:::{important} Important point
A very important point about FEniCS on HPC!
:::

### Prerequisites

Ensure that you have a container runtime installed (e.g. `docker` or `podman`)
and pull the following image with Spack pre-installed:

```bash
docker pull spack/ubuntu-noble:develop
```

### Credits

My thanks to the following people for their many days/weeks fiddling with
FEniCS on HPC systems over the past decade or so:

- Martin Rehor (University of Luxembourg, UL)
- Raphaël Bulle (UL, INRIA: talk on $\phi$-FEM on...)
- Andrey Latyshev (UL, Sorbonne Université, co-organiser)
- Sona Salehian Ghamsari (UL)
- Georgios Kafanas (UL, talk on EasyBuild on ...)
- Jahid Hassan (UL, talk on Easybuild on ...)
- Chris Richardson (Cambridge University, talk on ...)

whose shared knowledge has made this guide possible.

## Overview

### Assumptions

1. You are familiar with the basics (launching jobs, module systems etc.) on
   your HPC.
2. You have built software from source, although perhaps not on your HPC.

### Learning outcomes

1. Understand the main methods for installing FEniCSx on HPC systems.
2. Be able to configure FEniCSx at runtime for optimal performance.
3. Assess if an installation provides *reasonable* performance and scalability. 

## Installing

### Methods

Although FEniCS/DOLFINx can be installed in many ways, the only ones relevant
for good performance on HPC are:

1. **Source**. Build from source. With a reasonable set of system-provided
   modules this is possible, but if your HPC system ships with out-of-date
   modules, or you need many additional complex dependencies built alongside
   DOLFINx, it can quickly become painful.
2. [Easybuild](https://easybuild.io). Easybuild is a software build and
   installation framework that allows you to manage (scientific) software on
   High Performance Computing (HPC) systems in an efficient way. Focus on
   high-quality, well-tested package sets released twice a year (e.g. `2024a`,
   `2024b`).
3. [Spack](https://spack.io) is a powerful build and installation framework for
   managing complex and custom scientific software stacks across languages, compilers
   and microarchitectures. For a full tutorial see [Tutorial Spack
   101](https://spack-tutorial.readthedocs.io/). Focus on automatically
   building complex and custom software stacks from scratch.
4. [The European Scientific Software Initiative (EESSI)](https://www.eessi.io)
   pronounced 'easy' aims to build a common stack of scientific software
   installations. Focus on providing uniform set of binaries across European
   HPC sites. Support for FEniCS since early 2026.

:::{important} Always use the system-provided MPI 
For good performance DOLFINx requires an optimal MPI implementation tuned to
the underlying interconnect of *your* HPC, used within a queue-aware job
launcher e.g. `srun`. This rules out using pre-built DOLFINx binaries aimed at
desktop computers (Conda) and generic launchers e.g. `mpiexec`.
:::

### Decision tree

1. Does my HPC provide an up-to-date set of basic dependencies? A
   C++20-compliant compiler, MPI, Python, BLAS, CMake, HDF5, PETSc (rare!).
   - **Yes**: Source build, or *partial stack* Spack build.
   - **No**: Contact your HPC system administrators, or *full stack* Spack
     build, but can be tricky.
2. Does my HPC centre offer pre-built FEniCS Easybuild packages, or support the
   [EESSI](https://www.eessi.io) **and** are my requirements met by the binary
   builds on offer?
   - **Yes**: Easybuild or EESSI.
   - **No**: Another option. 
3. Do I have extensive custom requirements, e.g. need to integrate at runtime
   with other complex packages like gmsh, JAX, pytorch, or use exotic compiler
   toolchains (Intel, AOCC, NVIDIA)?:
   - **Yes**: Spack build.
   - **No**: Another option.
4. Do I have strict reproducibility requirements?
   - **Yes**: Containers (e.g. Apptainer/Singularity), wrapping one of the
     above build approaches.

:::{important} Avoid source builds if you can
Source builds are hard - if in doubt, choose partial stack Spack,
Easybuild/EESSI binaries, or as a last resort, full stack Spack. 
:::

:::{important} Our opinion
Today all of our internal projects use the *partial stack Spack* approach, with
around fifteen dependencies brought in pre-built from UL's Easybuild-based
module system. Everything else is built by Spack.
:::

### Source build

One of the main design goals with FEniCSx has been to transition to
standards-based build tooling, in particular, CMake for C++ and
[scikit-build-core](https://scikit-build-core.readthedocs.io) for Python
wrappers.

Standards-compliant build tooling means FEniCSx is reasonably easy to build
from source on any platform with a good set of dependencies, by proceeding
roughly as follows:

1. CMake - Install the C++ Basix, UFCx header and DOLFINx libraries.
2. Python/pip - Install Basix Python wrapper.
3. Python/pip - Install UFL and FFCx.
4. Python/pip - Install DOLFINx Python wrapper. 

#### Ubuntu container

As an example, on a clean Ubuntu 26.04 Docker image, it is possible to install
FEniCSx into a Python virtual environment `~/fenics` in around 50 nearly
standard `apt`, `cmake` and `pip` commands:

:::{literalinclude} source-install/Dockerfile
:lang: dockerfile
:lineno-match:
:caption: Installing FEniCSx 
:::

#### Typical HPC system

However, additional DOLFINx dependencies (multiple partitioners, adios2),
complex runtime dependencies (gmsh, JAX, TensorFlow), and critical dependencies
installed in non-standard ways (HPC module systems) can lead to brittle builds
and lots of trial-and-error.

As an example, I logged onto the University of Luxembourg HPC `aion`, which has
a good set of modules organised according to the easybuild `year{a,b}` system,
e.g. `2024a`. I found using `module spider` (search) and by cross-referencing
against the above Ubuntu build I loaded:

```bash
module load env/development/2024a
module load devel/Boost mpi/OpenMPI devel/CMake math/SCOTCH \
  math/ParMETIS data/HDF5 \
  lib/FlexiBLAS tools/petsc4py \
  lang/Python lib/mpi4py
```

I was pretty happy, as some of these dependencies are tricky and time-consuming
to build. However, I could not find `pkgconfig`, `spdlog`, `pugixml`,
`nanobind` or `scikit-build-core`. I then tried the newer `2025a` release which
did not have `petsc4py`,  although it did have `scikit-build-core` and
`pkgconfig`.

So in the end, I decided to go with the `2024a` release, 'knowing' that both
`spdlog` and `pugixml` are relatively easy to build, and that I could
(hopefully) build `nanobind` and `scikit-build-core` from source using `pip`.

I then copy and pasted `RUN` commands out from the `Dockerfile` above and
recorded my successes/failures:

- :white_check_mark: Basix C++ build. Easy!
- :white_check_mark: UFCx header. Easy!
- :white_check_mark: DOLFINx C++ build; worked after manually installing
  `spdlog` and `pugixml` from source using CMake. The first time I forgot to
  build both with shared library support `.so`, so I had to manually inspect
  the `CMakeLists.txt` for the `-DBUILD_SHARED_LIBS=ON` option and build again.
  Then I configured with:

      cmake -B build-dir/ -S . -DDOLFINX_UFCX_PYTHON=OFF \
        -DCMAKE_PREFIX_PATH=~/fenics

- :white_check_mark: Basix Python wrapper; Here I began running into an
  issue. Recall that I wanted to use some Easybuild-provided Python modules;
  this requires that the Python be allowed to 'see' the Easybuild
  Python `site-packages`:

      python -m venv --system-site-packages ~/fenics/
      ...
      python -m pip install scikit-build-core[pyproject] nanobind
      python -m build --no-build-isolation --check-build-dependencies .

- :white_check_mark: UFL and FFCx install. Easy!
- :white_check_mark: DOLFINx Python wrapper. Easy!

:::{seealso} The Future? The MPI ABI Initiative.
:class: dropdown
:open: false
An ABI compatibility guarantee allows a piece of software to be compiled
against one library (e.g. MPICH), and for the implementation to be swapped out
at runtime via dynamic linking (e.g. Intel MPI, Cray MPI, MVAPICH2).

The recent MPI5 ABI Initiative [](https://doi.org/10.1145/3615318.3615319)
ensures that all MPI5-compliant implementations *must* be ABI compatible -- in
the future it may be possible to ship DOLFINx binaries and then 'swap out' to
the platform-specific MPI implementation at runtime.
:::

### With Easybuild

### With Spack



## Runtime configuration

### The Python `import` problem 

### JIT compilation

## Testing and benchmarking

### FEniCS unit tests

### FEniCS performance tests


