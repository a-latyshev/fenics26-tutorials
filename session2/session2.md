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
a particular focus on the specifics aspects important for working with FEniCS.

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

- Martin Rehor
- Raphaël Bulle (talk on $\phi$-FEM on...)
- Andrey Latyshev (co-organiser)
- Sona Salehian Ghamsari
- Georgious Kafanas (talk on EasyBuild on ...)
- Jahid Hassan (talk on Easybuild on ...)
- Chris Richardson (talk on ...)

whose shared knowledge has made this guide possible.

## Overview


### Learning outcomes

1. Understand the main methods for installing FEniCSx on HPC systems.
2. Recognise MPI 
3. Be able to 

## Installing

### Methods

1. **Source**. Build from source. With a reasonable set of system-provided
   modules this can be OK, but if your HPC system ships with out-of-date
   software or you need many additional complex dependencies, it can quickly
   become painful.
2. [Easybuild](https://easybuild.io). Easybuild is a software build and installation framework that
   allows you to manage (scientific) software on High Performance Computing
   (HPC) systems in an efficient way. Focus on high-quality, well-tested
   package sets released twice a year (e.g. `2024a`, `2024b`).
3. [Spack](https://spack.io) is a very powerful build and installation
   framework complex and custom scientific software stacks across languages,
   compilers and microarchitectures. For a full tutorial see [Tutorial Spack
   101](https://spack-tutorial.readthedocs.io/). Focus on automatically
   building complex and custom software stacks from scratch.
4. [The European Scientific Software Initiative (EESSI)](https://www.eessi.io)
   pronounced 'easy' aims to build a common stack of scientific software
   installations. Focus on providing uniform set of binaries across European
   HPC sites. Support for FEniCS since early 2026.

:::{important} Always use the system-provided MPI 
For good performance DOLFINx requires an optimal MPI implementation tuned to
the underlying interconnect of *your* HPC, and a queue-aware launcher e.g.
`srun`. This rules out using pre-built DOLFINx binaries aimed at desktop
computers (Conda) and generic launchers e.g. `mpiexec`.
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
   with other complex packages like gmsh, JAX, pytorch? or use exotic compiler
   toolchains (Intel, AOCC, NVIDIA)?:
   - **Yes**: Spack build.
   - **No**: Another option.
4. Do I have strict reproducibility requirements?
   - **Yes**: Containers (e.g. Apptainer/Singularity), wrapping one of the
     above build approaches.

:::{important} Avoid source builds
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

Standards-compliance build tooling means FEniCSx is reasonably easy to build
from source on any platform. However, additional DOLFINx dependencies (MPI,
PETSc, partitioners), additional complex runtime dependencies (gmsh, JAX,
TensorFlow), installed in non-standard ways (HPC module systems) can lead to
brittle builds and lots of trial-and-error required.

#### FEniCSx C++ components

DOLFINx, the problem-solving environment, and Basix, the basis tabulator, are
pure C++ libraries and can be installed and run on a system without Python.
The Python wrappers can then be built separately.

```bash
docker run -ti ubuntu
```

#### FEniCSx Python components

FFCx and UFL are pure Python and are 'nearly trivial' to install.


```bash
apt update
apt 
```

#### FEniCSx Python wrappers


:::{seealso} The Future? The MPI ABI Initiative.
:class: dropdown
:open: false
An ABI compatibility guarantee allows one a piece of software to be compiled
against one library (e.g. MPICH), and for the implementation to be swapped out
at runtime via dynamic linking (e.g. Intel MPI, Cray MPI, MVAPICH2).

The recent MPI5 ABI Initiative [](https://doi.org/10.1145/3615318.3615319)
ensures that all MPI5-compliant implementations *must* be ABI compatible -- in
the future it may be possible to ship DOLFINx binaries and then 'swap out' to
the platform-specific MPI implementation at runtime.
:::


### Benchmarking



#### Writing DOLFINx solvers



#### Running



#### The Python 

#### JIT compilation

DOLFINx Python uses just-in-time (JIT) compilation

## TLDR
