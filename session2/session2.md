---
authors:
- jshale
---

# A guide to building and running FEniCSx on HPC systems

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
   - **Yes**: Wrap your chosen approach in a container image and execute in an
     HPC-aware container runtime (e.g. Apptainer/Singularity).

:::{important} Avoid source builds if you can
Source builds are hard - if in doubt, choose partial stack Spack,
Easybuild/EESSI binaries, or as a last resort, full stack Spack. 
:::

:::{important} My opinion
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
from source on any platform with a 'good enough' set of dependencies, by
proceeding roughly as follows:

1. Install and/or build the necessary dependencies.
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
:caption: Installing FEniCSx within a clean Ubuntu 26.04 Docker image. 
:::

#### Typical HPC system

However, additional DOLFINx dependencies (multiple partitioners, adios2),
complex runtime dependencies (gmsh, JAX, TensorFlow), and critical dependencies
installed in non-standard ways (HPC module systems) can lead to brittle builds
and lots of trial-and-error.

As an example, I logged onto the University of Luxembourg HPC aion cluster,
which has a good set of modules organised according to the easybuild
`year{a,b}` system, e.g. `2024a`. I found using `module spider` (search) and by
cross-referencing against the above Ubuntu build I loaded:

```bash
module load env/development/2024a
module load devel/Boost mpi/OpenMPI devel/CMake math/SCOTCH \
  math/ParMETIS data/HDF5 \
  lib/FlexiBLAS tools/petsc4py \
  lang/Python lib/mpi4py
```

I was pretty happy, as some of these dependencies are tricky and time-consuming
to build. However, I could not find pkgconfig, spdlog, pugixml, nanobind or
scikit-build-core. I then tried the newer `2025a` release which did not have
petsc4py, although it did have scikit-build-core and pkgconfig.

So in the end, I decided to go with the 2024a release, 'knowing' that both
spdlog and pugixml are relatively easy to build, and that I could
(hopefully) install nanobind and scikit-build-core from PyPI using `pip`.

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

How smoothly this will depend on how well-aligned your cluster's modules are
with the requirements of FEniCS - only three years ago, I had to build CMake,
PETSc and PugiXML from source, and in the past I recall building Boost and HDF5
from source too!

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

Then build (this can take a while):

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
available. If your work requires a specific configuration or more recent
version, Spack may be a more practical choice.
:::

### European Enviroment for Scientific Software Installations (EESSI)

The EESSI aims to set up a shared binary stack of scientific software
installations, and so avoid a lot of duplicate work across HPC sites. In
particular, EESSI aims to provide a *uniform* experience across all sites,
while focusing on performance. EESSI uses Easybuild to generate this shared
binary stack.

:::{figure} images/overview_layers.png
:width: 50%
:align: left
:::

EESSI is available on several of the EuroHPC JU systems including Karolina,
Vega, Deucalion ARM and GPU parititions and MareNostrum 5, for a full list
see [Systems where EESSI is available](https://www.eessi.io/docs/systems/).

Once installed by your site admin, EESSI is nearly trivial to use:

1. Check that EESSI is available.

```bash
ls /cvmfs/software.eessi.io
```

should show:

```
defaults  host_injections  init  README.eessi  versions
```

and then:

```bash
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
```

giving (abbreviated):

```
Found EESSI repo @ /cvmfs/software.eessi.io/versions/2023.06!
archdetect says x86_64/amd/zen2
archdetect could not detect any accelerators
Using x86_64/amd/zen2 as software subdirectory.
...
Prepending site path /cvmfs/software.eessi.io/host_injections/2023.06/software/linux/x86_64/amd/zen2/modules/all to $MODULEPATH...
Environment set up to use EESSI (2023.06), have fun!
```

then load the module for e.g. DOLFINx Python:

```bash
module load FEniCS-DOLFINx-Python/0.9.0-foss-2023b
```

and run e.g.:

```bash
mpiexec python -c "from mpi4py import MPI; import dolfinx"  
```

:::{important} Some rough edges
:class: dropdown
:open: false
I encountered two rough edges related to MPI in the EESSI 2023.06 set on the aion
cluster at ULHPC in mid-2026. Both of these points link back to the guidance
"Always using the system-provided MPI" - as EESSI provides a full binary stack,
it cannot follow this guidance.


1. A [known
   issue](https://www.eessi.io/docs/known_issues/eessi-2023.06/#eessi-production-repository-v202306)
   when using `mpirun` leading to the failure:

```bash
Failed to modify UD QP to INIT on mlx5_0: Operation not permitted
```
  
  It is possible to fix this by instructing OpenMPI to not use libfabric
  and turn off UCX uct:

```bash
mpiexec -mca pml ucx -mca btl '^uct,ofi' -mca mtl '^ofi'
```

  Whether libfabric or ucx provides higher performance communication
  depends on the interconnect used in your cluster.

2. Inability to use system `srun` to launch jobs. This is perhaps a bigger
   issue; launching MPI jobs with a scheduler-integrated launcher e.g. `srun`
   currently fails:

```bash
--------------------------------------------------------------------------
A requested component was not found, or was unable to be opened.  This
means that this component is either not installed or is unable to be
used on your system (e.g., sometimes this means that shared libraries
that the component requires are unable to be found/loaded).  Note that
PMIx stopped checking at the first component that it did not find.

Host:      aion-0086
Framework: psec
Component: munge
--------------------------------------------------------------------------
```

   Using a scheduler-integrated launcher like `srun` over `mpiexec` greatly
   improves the HPC experience and many of our workflows are built around
   `srun`, so this is not ideal.
:::

:::{seealso} The Future? The MPI ABI Initiative.
:class: dropdown
:open: false
An ABI compatibility guarantee allows software compiled against one MPI
implementation (e.g. MPICH) to have it swapped out at runtime via dynamic
linking (e.g. Intel MPI, Cray MPI, MVAPICH2).

The recent MPI5 ABI Initiative [](https://doi.org/10.1145/3615318.3615319)
guarantees ABI compatibility across all MPI5-compliant implementations — in the
future it may be possible to ship DOLFINx binaries (via EESSI, conda, pypi.org
etc.) and swap in a platform-tuned MPI at runtime.
:::

### With Spack

:::{figure} images/spack_logo.svg
:width: 200px
:align: left
:::

Spack can build an entire software stack — compilers, MPI, PETSc, ADIOS2, gmsh
etc. — in a single shot. Particularly powerful is Spack's concretisation
algorithm that acts as a very smart constraint solver: constraints from package
definitions, already-installed specs, and the user's request are compiled into
a logical encoding, and the concretisation algorithm finds the optimal
'concrete' solution satisfying all of them.

:::{important} Spack documentation
Spack is a complex and powerful piece of software; I recommend following the
[Spack Tutorial](https://spack-tutorial.readthedocs.io/en/latest/). Here I will
cover only some aspects related to installing and running FEniCS.
:::

On a cluster, the *partial stack* approach works well in practice: we tell
Spack to reuse the scheduler-integrated and interconnect-tuned MPI along with
the compiler from the module system, and then build everything else itself.
This is what we use for most internal projects at the University of Luxembourg.

#### Setting up Spack

Spack has a very minimal dependency set and can be installed by checking the source
out using `git`:

```bash
cd ~
git clone --depth=2 https://github.com/spack/spack.git
source ~/spack/share/spack/setup-env.sh
```

The key step for a *partial stack* build is telling Spack which dependencies to
take from the module system rather than building them itself. On an 'unknown'
HPC system I typically explore with `module spider` to find all compiler and
interconnect/MPI related components, e.g.:

```bash
module spider OpenMPI
module spider PMIx
module spider GCC
module spider compiler/GCCcore
```

While making notes of version numbers. I then `module load` all of the modules
and check for warning messages related to e.g. compatability.

Then, create `~/.spack/packages.yaml` with entries for each system-provided
package. The *abbreviated* example below is for the University of Luxembourg
`aion` cluster (GCC 13.2.0, OpenMPI 4.1.6, SLURM):

```yaml
packages:
  gcc:
    externals:
    - spec: gcc@13.2.0+binutils languages:='c,c++,fortran'
      modules:
      - compiler/GCC/13.2.0
      extra_attributes:
        compilers:
          c: /opt/apps/easybuild/.../GCCcore/13.2.0/bin/gcc
          cxx: /opt/apps/easybuild/.../GCCcore/13.2.0/bin/g++
          fortran: /opt/apps/easybuild/.../GCCcore/13.2.0/bin/gfortran
    buildable: false
  openmpi:
    variants: fabrics=ofi,ucx schedulers=slurm
    externals:
    - spec: openmpi@4.1.6
      modules:
      - mpi/OpenMPI/4.1.6-GCC-13.2.0
    buildable: false
  mpi:
    buildable: false
  slurm:
    externals:
    - spec: slurm@23.11.10 sysconfdir=/etc/slurm
      prefix: /usr
    buildable: false
  # ... plus binutils, libevent, libfabric, hwloc, ucx, pmix, libxml2, zlib etc.
```

The full `packages.yaml` can be found
[here](https://gist.github.com/jhale/23d4d7646e2dc05d0adc0395767d044a).

:::{important} Pin partial stack dependencies as not buildable
Setting `mpi: buildable: false` etc. together with the specific `openmpi` entry
guarantees Spack always uses the scheduler-integrated MPI from the module system,
not a self-built one that may lack the native fabric or Slurm support.
:::

:::{important} Checking dynamic linking
:class: dropdown
:open: false
Inspecting the full `packages.yaml` file you will see entries for `libxml2` and
`zlib`. Why do we need these? The first time I tried this Spack configuration
without `libxml2` and `zlib`, Spack chose to build `zlib` and `libxml2`.
However, `libxml2` produced warnings at runtime due to OpenMPI dynamically
linking against potentially incompatible (Spack-provided) version. To avoid
this, it is good practice to inspect the linking of at least MPI and ask Spack
not to build all the lower-level dependencies as well.

```bash
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
ldd $EBROOTOPENMPI/lib64/libmpi.so
```

gives:

```
linux-vdso.so.1 (0x00007fffb1bf7000)
libopen-rte.so.40 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/OpenMPI/4.1.6-GCC-13.2.0/lib/libopen-rte.so.40 (0x00007f25507ab000)
libopen-pal.so.40 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/OpenMPI/4.1.6-GCC-13.2.0/lib/libopen-pal.so.40 (0x00007f25506b9000)
librt.so.1 => /lib64/librt.so.1 (0x00007f25504b1000)
libutil.so.1 => /lib64/libutil.so.1 (0x00007f25502ad000)
libhwloc.so.15 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/hwloc/2.9.2-GCCcore-13.2.0/lib64/libhwloc.so.15 (0x00007f255024c000)
libpciaccess.so.0 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/libpciaccess/0.17-GCCcore-13.2.0/lib64/libpciaccess.so.0 (0x00007f2550240000)
libxml2.so.2 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/libxml2/2.11.5-GCCcore-13.2.0/lib64/libxml2.so.2 (0x00007f25500db000)
libdl.so.2 => /lib64/libdl.so.2 (0x00007f254fed7000)
libz.so.1 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/zlib/1.2.13-GCCcore-13.2.0/lib64/libz.so.1 (0x00007f254febc000)
liblzma.so.5 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/XZ/5.4.4-GCCcore-13.2.0/lib64/liblzma.so.5 (0x00007f254fe8d000)
libevent_core-2.1.so.7 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/libevent/2.1.12-GCCcore-13.2.0/lib64/libevent_core-2.1.so.7 (0x00007f254fe55000)
libevent_pthreads-2.1.so.7 => /opt/apps/easybuild/systems/aion/rhel810-20250803/2023b/epyc/software/libevent/2.1.12-GCCcore-13.2.0/lib64/libevent_pthreads-2.1.so.7 (0x00007f254fe50000)
libm.so.6 => /lib64/libm.so.6 (0x00007f254face000)
libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f254f8ae000)
libc.so.6 => /lib64/libc.so.6 (0x00007f254f4d7000)
/lib64/ld-linux-x86-64.so.2 (0x00007f255076b000)
```

While writing this tutorial, I realised I should probably also add `xz` to my `packages.yaml` file.
:::

#### Building FEniCS

Create a Spack environment and add the FEniCSx specs:

```bash
spack env create -d ~/fenicsx-env/
spack env activate ~/fenicsx-env/
spack add py-fenics-dolfinx@0.10 ^fenics-dolfinx+adios2 ^adios2+python ^petsc+mumps
spack concretize
spack install
```

#### Testing and using the build

Quickly verify the build works under MPI:

```bash
srun python -c "from mpi4py import MPI; import dolfinx"
```

## Runtime configuration

Two of the most common and impactful runtime performance problems on HPC
systems are caused not by the numerical computation itself, but by disk access
patterns during program startup and just-in-time compilation (JIT). HPC storage
systems are optimised for high aggregate throughput on large sequential reads
and writes - the kind generated by parallel I/O libraries such as HDF5 or
ADIOS2. They perform poorly under workloads that issue many small, random, or
metadata-heavy operations, which is precisely the access pattern generated when
Python initialises and loads modules, and when FEniCSx performs just-in-time
compilation of finite element kernels.

### The Python `import` problem 

The performance issues related to Python initialising and loading modules on
HPC has become infamous enough to warrant a specific name: "The Python `import`
problem". Infact, the issue is not specific to Python, and has been observed on
very large MPI runs (10000+ MPI ranks) using compiled C/C++/Fortran
applications as well.

#### Avoid the problem?

The first piece of advice is to try and avoid the problem! Put all FEniCS
installation files, for example `$SPACK_HOME`, on the most performant HPC
storage system for small file accesses - this is typically `$HOME`, not
`$SCRATCH`.

Then run a very simple script on an geometrically increasing number of nodes,
up to the maximum number needed for your analysis, e.g.:

```bash
#!/bin/bash -l
# SBATCH directives
# Initialisation

SCRIPT_START=$(date +%s)
srun python -c "from mpi4py import MPI; import dolfinx"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$(( SCRIPT_END - SCRIPT_START ))

echo "Script elapsed (seconds): ${SCRIPT_ELAPSED}"
```

In short, if you start to see a huge blow up (minutes, or even hanging jobs) in
`$SCRIPT_ELAPSED`, you likely have the Python `import` problem. If everything
looks OK, then no solution is needed.

#### Containers

Containers (e.g. [Apptainer/Singularity](https://apptainer.org),
[Docker](https://docker.io)) bundle the entire software stack into a single
binary image file stored on shared storage. At job startup, the container
runtime makes one large sequential copy into local memory, so each MPI rank
reads a large local file rather than issuing thousands of independent metadata
requests to the parallel filesystem for individual `.py` and `.so` files. This
dramatically reduces the metadata load on the parallel filesystem and, in
practice, eliminates the `import` problem even at large node counts. This was
demonstrated in [](https://doi.org/10.1109/MCSE.2017.2421459) using the
[Shifter](https://github.com/NERSC/shifter) runtime, and this applies more
common [Apptainer/Singularity](https://apptainer.org) runtime.

#### Spindle

[Spindle](https://github.com/hpc/Spindle) replaces the dynamic linker and
Python import machinery at runtime with an MPI-aware load. When one MPI rank
reads a shared library or Python module for the first time, Spindle broadcasts
the file contents to all other ranks over MPI. All subsequent ranks satisfy the
request from an in-memory cache rather than hitting the filesystem. The net
effect is that each file is read from disk exactly once per job, regardless of
node count, which eliminates the per-rank metadata request that causes the
`import` problem. Spindle requires no changes to the application or the
installation; it is invoked by prepending `spindle` to the usual `srun`
command:

```bash
spindle srun python my_fenicsx_script.py
```

Spindle can be installed using Spack or from source, and does not require
special permissions. We have used it with success to execute jobs with 10000s
of MPI ranks.

### JIT compilation

#### Performance

Each time DOLFINx encounters a new variational form, FFCx compiles it to a
shared library and writes it to a cache directory (default `~/.cache/fenics` or
`$XDG_CACHE_HOME` if set). On HPC systems, simultaneous JIT cache reads and
writes from thousands of ranks cause the same filesystem pressure as the
`import` problem.

DOLFINx mitigates this, to an extent, via the optional `mpi_jit_decorator`
keyword argument to the JIT compilation functions e.g. `dolfinx.fem.form`: rank
0 compiles the form and writes to the cache; all other ranks block on an MPI
broadcast, which rank 0 sends when it succeeds. So when using `MPI_COMM_WORLD`,
the default, each form is compiled once per job regardless of the size.

However, the cache read on non-root ranks still touches the parallel
filesystem. Pointing the cache at a node-local path such as an SSD-backed
`$TMPDIR`:

```bash
export XDG_CACHE_HOME=$TMPDIR/$USER/fenics-cache-$SLURM_JOB_ID
```

*and* performing the JIT-compilation + cache lookup on a communicator split
along the shared memory boundaries (i.e. one communicator per node):

```python
...
a = ufl.inner(u, v)*ufl.dx
shared_mem_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED, key=MPI.COMM_WORLD.rank)
a_dolfinx = form(a, jit_comm=shared_mem_comm)
```

solves the parallel file system bottleneck at the expense of requiring JIT
compilation on every node within a job, and on every job start, as typically
`$TMPDIR` is is cleared on job exit.

#### Compiler flags



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
