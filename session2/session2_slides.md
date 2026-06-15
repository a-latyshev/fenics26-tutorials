---
title: "FEniCSx on HPC Systems"
author: "Jack S. Hale"
date: "FEniCS Conference 2026"
---

# Introduction

## Installing software on someone else's computer

> The second hardest thing in scientific computing is installing software on
> someone else's computer.
>
> **Hans Petter Langtangen, FEniCS Workshop 2005.**

## Three stages

1. **Building** — source, Easybuild/EESSI, Spack
2. **Runtime configuration** — Python `import` problem, JIT compilation
3. **Testing and benchmarking** — unit tests, weak scaling

## The golden rule

**Always use the system-provided MPI and job launcher.**

- DOLFINx requires MPI tuned to the HPC interconnect
- Use `srun` (or equivalent), not `mpiexec`
- Rules out Conda and most pre-built desktop binaries

# Building

## Decision tree

1. Pre-built via **Easybuild** or **EESSI**? → use it
2. HPC has C++20 compiler, MPI, PETSc, HDF5? → **source build** or partial-stack **Spack**
3. Missing dependencies or exotic requirements? → full-stack **Spack**
4. Strict reproducibility? → wrap in a **container** (Apptainer/Singularity)

> Avoid source builds if you can.
> We prefer partial-stack Spack.

## Source build

Install order:

1. System modules (MPI, BLAS, HDF5, PETSc …)
2. Basix C++ → UFCx header → DOLFINx C++ (`cmake`)
3. Basix Python wrapper → UFL → FFCx → DOLFINx Python (`pip`)

Common missing pieces: `pkgconfig`, `spdlog`, `pugixml`, `nanobind`,
`scikit-build-core`.

## Easybuild

```bash
# If a pre-built module is available:
module spider FEniCS-DOLFINx-Python
module load FEniCS-DOLFINx-Python/0.9.0-foss-2023b
```

```bash
# Build it yourself:
module load tools/EasyBuild
eb FEniCS-DOLFINx-Python-0.9.0-foss-2023b.eb --robot
```

Each FEniCSx version × toolchain requires a dedicated easyconfig;
support for new releases can lag.

## EESSI

Uniform binary stack across European HPC sites (Karolina, Vega,
MareNostrum 5 …).

```bash
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
module load FEniCS-DOLFINx-Python/0.9.0-foss-2023b
```

**Rough edge:** EESSI ships its own MPI — `srun` integration and
interconnect tuning may not work out of the box.

## Spack: partial-stack approach

Reuse system MPI and compiler; let Spack build everything else.

```yaml
# ~/.spack/packages.yaml (abbreviated)
packages:
  openmpi:
    variants: fabrics=ofi,ucx schedulers=slurm
    externals:
    - spec: openmpi@4.1.6
      modules: [mpi/OpenMPI/4.1.6-GCC-13.2.0]
    buildable: false
  mpi:
    buildable: false     # never build a self-contained MPI
```

## Spack: building FEniCSx

```bash
spack env create -d ~/fenicsx-env/
spack env activate ~/fenicsx-env/
spack add fenics-dolfinx@0.10
spack concretize        # inspect output: [e] = external, [-] = to build
spack install           # ~30 min; parallel: -p2 -j4
```

# Runtime Configuration

## The Python `import` problem

Many small metadata requests to shared filesystem from all MPI ranks
→ slow startup or hanging jobs at scale.

**Diagnose:**

```bash
srun python -c "from mpi4py import MPI; import dolfinx"
# time across 1, 2, 4, … nodes; watch for blow-up
```

**Solutions:** local `$HOME` install, containers, Spindle.

## Spindle

MPI-aware loader: broadcasts each file to all ranks after one read.

```bash
spindle srun python my_fenicsx_script.py
```

- No application changes required
- Each file read from the parallel filesystem exactly once per job
- Used successfully at 10 000+ MPI ranks

## JIT compilation

FFCx compiles variational forms to shared libraries at runtime; cache
reads/writes stress the filesystem at scale.

**Node-local cache:**

```bash
export XDG_CACHE_HOME=$TMPDIR/$USER/fenics-cache-$SLURM_JOB_ID
```

**Compile once per node:**

```python
shared_mem_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
a_dolfinx = form(a, jit_comm=shared_mem_comm)
```

## JIT compiler flags

`~/.config/dolfinx/dolfinx_jit_options.json`:

```json
{ "cffi_extra_compile_args": ["-march=native", "-O3"] }
```

- `-ffast-math` is worth experimenting with but can cause correctness issues
- Add `-mtune=native` alongside `-march=native` on ARM

# Testing and Benchmarking

## Unit tests

Run the DOLFINx test suite on your HPC before production use.

Support for automatic test execution is being integrated into Easybuild
and Spack package recipes.

## Performance tests

`github.com/fenics/performance-test` — two C++ PETSc-based solvers
(Poisson, Elasticity).

```bash
git clone https://github.com/fenics/performance-test
cmake -B build-dir/ -S src/ && cmake --build build-dir/
srun -n 8 ./dolfinx-scaling-test \
  --problem_type poisson --scaling_type weak --ndofs 500000 \
  -pc_type hypre -pc_hypre_type boomeramg ...
```

## Interpreting weak scaling results

- Run from 1 to 8 nodes at **50 % core utilisation** per node
- Expect **roughly constant** assembly and solve times with node count
- Slight deterioration from 1 → 2 nodes is normal (shared memory → interconnect)

# Closing

## Closing thoughts

> The third hardest thing in scientific computing is installing software on
> someone else's computer.
>
> **Jack S. Hale, FEniCS Conference 2026.**

- Build tooling has improved enormously since 2005
- Growing challenge: heterogeneous hardware (ARM, NVIDIA, AMD) and mixed stacks (DOLFINx + PyTorch)
- MPI5 ABI initiative may eventually allow portable binaries with swappable MPI at runtime
