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

## Installing



### Decision tree

1. Does my HPC provide an up-to-date set of basic dependencies? A
   C++20-compliant compiler, MPI, Python, BLAS, CMake, HDF5, PETSc (rare!).
   - **Yes**: Source build, or *partial stack* Spack build.
   - **No**: *Full stack* Spack build, or contact your HPC system administrators.
2. Does my HPC centre offer FEniCS Easybuild packages, or support the [European
   Scientific Software Initiative (EESSI)](https://www.eessi.io) **and** are my
   requirements met by the binary builds on offer?
   - **Yes**: Easybuild or EESSI.
   - **No**: Source build or Spack build. 
3. Do I have extensive custom requirements, e.g. need to integrate at runtime
   with other complex packages like gmsh, JAX, pytorch:
   - **Yes**: Spack build.
   - **No**: Source.

:::{important} Avoid source builds
Source builds are hard - if in doubt, choose partial stack Spack,
Easybuild/EESSI binaries, or as a last resort, full stack Spack. 
:::

:::{important} Always use the system-provided MPI 
For good performance DOLFINx requires an optimal MPI implementation tuned to
the underlying interconnect of *your* HPC, and a queue-aware launcher e.g.
`srun`. This rules out using pre-built DOLFINx binaries aimed at desktop
computers (Conda) and generic launchers e.g. `mpiexec`.
:::

### Source build

One of the main design goals with FEniCSx has been to transition to
standards-based build tooling, in particular, CMake for C++ and
[scikit-build-core](https://scikit-build-core.readthedocs.io) for Python
wrappers.

Standards-compliance means FEniCSx is reasonably easy to build from source on
any platform. However, additional dependencies (MPI, PETSc, partitioners),
additional complex runtime dependencies (gmsh, JAX, TensorFlow), installed in
non-standard ways (HPC module systems) can lead to brittle builds and lots of
trial-and-error required.

```bash
docker run -ti ubuntu
```

This container is effectively clean, allowing us to see 


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
