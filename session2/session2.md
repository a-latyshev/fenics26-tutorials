---
authors:
- jshale
---

# Ten years of FEniCS on HPC systems 

## Introduction


This guide distills knowledge built up at the University of Luxembourg over the
past decade on building and running FEniCS on HPC systems.

:::{figure} images/aion_compute_racks.jpg
:width: 50%
:align: left
Aion supercomputer compute racks, University of Luxembourg HPC.
:::

During the session I will present this material and give a brief interactive
demo on installing FEniCS with
[Spack](https://spack-tutorial.readthedocs.io/en/latest/).

The most important points will be called out and summarised as a TLDR; at the
end:

:::{important} Important point
A very important point about FEniCS on HPC!
:::

### Before the session

Ensure that you have a container runtime installed (e.g. `docker` or `podman`)
and pull the following image with Spack pre-installed:

```bash
docker pull spack/ubuntu-noble:develop
```


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

1. **Source**. Build directly from source using system-provided modules. Focus
   on full control over the build, at the cost of manual dependency management.
2. [Easybuild](https://easybuild.io). A build and installation framework for
   scientific software on HPC. Focus on high-quality, well-tested package sets
   released twice a year (e.g. `2024a`, `2024b`).
3. [Spack](https://spack.io). A flexible build and installation framework for
   complex scientific software stacks. For a full tutorial see [Spack
   101](https://spack-tutorial.readthedocs.io/). Focus on custom stacks across
   compilers and microarchitectures.
4. [EESSI](https://www.eessi.io) (European Scientific Software Initiative,
   pronounced 'easy'). Focus on providing a uniform set of binaries across
   European HPC sites. Support for FEniCS since early 2026.

:::{important} Always use the system-provided MPI 
For good performance DOLFINx requires an optimal MPI implementation tuned to
the underlying interconnect of *your* HPC, used within a scheduler-integrated
job launcher e.g. `srun`. This rules out using pre-built DOLFINx binaries aimed
at desktop computers (Conda) and non-scheduler-integrated launchers e.g.
`mpiexec`.
:::

### Decision tree

1. Does my HPC centre offer pre-built FEniCS via Easybuild or
   [EESSI](https://www.eessi.io), **and** are my requirements met by the binary
   builds on offer?
   - **Yes**: Use Easybuild or EESSI — stop here.
   - **No**: Continue to step 2.
2. Does my HPC provide an up-to-date set of basic dependencies? A
   C++20-compliant compiler, MPI, Python, BLAS, CMake, HDF5, PETSc (rare!).
   - **Yes**: Source build, or *partial stack* Spack build — stop here.
   - **No**: *Full stack* Spack build. Contact your HPC administrators first;
     a full stack build can be tricky.
3. Do I have extensive custom requirements, e.g. integration with gmsh, JAX,
   pytorch, or exotic compiler toolchains (Intel, AOCC, NVIDIA)?
   - **Yes**: Spack build (partial or full stack).
4. Do I have strict reproducibility requirements?
   - **Yes**: Wrap your chosen approach in a container (e.g. Apptainer/Singularity).

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

As an example, on a clean Ubuntu 26.04 Docker image, FEniCSx can be installed
into a virtual environment `~/fenics` in around 50 `apt`, `cmake` and `pip`
commands:

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
did not have `petsc4py`, although it did have `scikit-build-core` and
`pkgconfig`.

So in the end, I decided to go with the `2024a` release, 'knowing' that both
`spdlog` and `pugixml` are relatively easy to build, and that I could
(hopefully) install `nanobind` and `scikit-build-core` from PyPI using `pip`.

I then copied and pasted `RUN` commands out from the `Dockerfile` above and
recorded my successes/failures:

- ✅ Basix C++ build. Easy!
- ✅ UFCx header. Easy!
- 🟡 DOLFINx C++ build; worked after manually installing
  `spdlog` and `pugixml` from source using CMake. The first time I forgot to
  build both with shared library support `.so`, so I had to manually inspect
  the `CMakeLists.txt` for the `-DBUILD_SHARED_LIBS=ON` option and build again.
  Then I configured with:

      cmake -B build-dir/ -S . -DDOLFINX_UFCX_PYTHON=OFF \
        -DCMAKE_PREFIX_PATH=~/fenics

- 🟡 Basix Python wrapper; Here I began running into an
  issue. Recall that I wanted to use some Easybuild-provided Python modules;
  this requires that the Python be allowed to 'see' the Easybuild
  Python `site-packages`:

      python -m venv --system-site-packages ~/fenics/
      ...
      python -m pip install scikit-build-core[pyproject] nanobind
      python -m build --no-build-isolation --check-build-dependencies .

- ✅ UFL and FFCx install. Easy!
- ✅ DOLFINx Python wrapper. Easy!

:::{seealso} The Future? The MPI ABI Initiative.
:class: dropdown
:open: false
An ABI compatibility guarantee allows software compiled against one MPI
implementation (e.g. MPICH) to have it swapped out at runtime via dynamic
linking (e.g. Intel MPI, Cray MPI, MVAPICH2).

The recent MPI5 ABI Initiative [](https://doi.org/10.1145/3615318.3615319)
guarantees ABI compatibility across all MPI5-compliant implementations — in
the future it may be possible to ship DOLFINx binaries and swap in the
platform-specific MPI at runtime.
:::

### With Easybuild

:::{figure} images/easybuild_logo.png
:width: 250px
:align: left
:::

#### Using pre-built modules

Only some HPC centres will have FEniCSx available as a pre-built Easybuild
module. If yours does, search for it with:

```bash
module spider FEniCS-DOLFINx-Python
```

Then load the module and its dependencies:

```bash
module load FEniCS-DOLFINx-Python/0.9.0-foss-2023b
```

:::{important} Check the toolchain
Easybuild packages are built against a specific toolchain (e.g. `foss-2023b`,
`intel-2023b`). Ensure any other modules you load use the same toolchain to
avoid ABI mismatches.
:::

#### Building with `eb`

If no pre-built module is available, you can build FEniCSx yourself. First,
load EasyBuild from the module system (the exact name varies by site — use
`module spider EasyBuild` to find it):

```bash
module load tools/EasyBuild
```

Next, clone the easyconfigs repository to get the FEniCSx easyconfig:

```bash
git clone https://github.com/easybuilders/easybuild-easyconfigs
```

The easyconfig is at:

```
easybuild-easyconfigs/easybuild/easyconfigs/f/FEniCS-DOLFINx-Python/FEniCS-DOLFINx-Python-0.9.0-foss-2023b.eb
```

Do a dry run first to check what will be built:

```bash
eb FEniCS-DOLFINx-Python-0.9.0-foss-2023b.eb --robot --dry-run
```

Then build (this will take a while):

```bash
eb FEniCS-DOLFINx-Python-0.9.0-foss-2023b.eb --robot
```

`--robot` automatically resolves and builds all missing dependencies. Once
complete, make the new modules visible and load:

```bash
module use $EASYBUILD_INSTALLPATH/modules/all
module load FEniCS-DOLFINx-Python/0.9.0-foss-2023b
```

:::{note} Easyconfig availability
Each version of FEniCSx against each toolchain requires a dedicated easyconfig
file, written and reviewed by the community. As a result, support for new
FEniCSx releases can lag behind, and not every version will have an easyconfig
available. If your work requires a specific recent version, Spack may be a more
practical choice.
:::

### With Spack

:::{figure} images/spack_logo.svg
:width: 200px
:align: left
:::



## Runtime configuration

### The Python `import` problem 

### JIT compilation

## Testing and benchmarking

### FEniCS unit tests

### FEniCS performance tests

## Credits

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

## AI use statement

The document draft was written without AI. Claude Sonnet 4.6 was used for
proof-reading, suggestions on improving the flow, and adding some visual
elements (logos etc.).
